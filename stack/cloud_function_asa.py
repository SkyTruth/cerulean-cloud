"""cloud function to find slick culprits"""

import base64
import json
import os
import time

import database
import git
import pulumi
from pulumi_gcp import (
    cloudfunctionsv2,
    cloudrun,
    cloudscheduler,
    cloudtasks,
    projects,
    serviceaccount,
    storage,
)
from utils import construct_name, pulumi_create_zip
from vpc_connector import vpc_connector

stack = pulumi.get_stack()
project_id = pulumi.Config("gcp").require("project")
region = pulumi.Config("gcp").require("region")
# Function timeout and Scheduler attempt deadline. 1800 s is Scheduler's maximum
# for HTTP targets. The daily batch ran 25 min on 2026-11-23 (1,000 runs).
timeout_seconds = 1800

# We will store the source code to the Cloud Function in a Google Cloud Storage bucket.
bucket = storage.Bucket(
    construct_name("bucket-cf-asa"),
    location="EU",
    lifecycle_rules=[
        storage.BucketLifecycleRuleArgs(
            action=storage.BucketLifecycleRuleActionArgs(type="Delete"),
            condition=storage.BucketLifecycleRuleConditionArgs(
                age=7, matches_prefixes=["ais/"]
            ),
        ),
    ],
    labels={"pulumi": "true", "environment": pulumi.get_stack()},
)

# Create the Queue for tasks
queue = cloudtasks.Queue(
    construct_name("queue-cloud-tasks-asa-analysis"),
    location=region,
    rate_limits=cloudtasks.QueueRateLimitsArgs(
        max_concurrent_dispatches=200,
        max_dispatches_per_second=1,
    ),
    retry_config=cloudtasks.QueueRetryConfigArgs(
        # Allow limited retries for transient downstream failures (e.g., BigQuery hiccups).
        max_attempts=3,
        max_backoff="300s",
        max_doublings=3,
        max_retry_duration="600s",
        min_backoff="10s",
    ),
    stackdriver_logging_config=cloudtasks.QueueStackdriverLoggingConfigArgs(
        sampling_ratio=0.9,
    ),
)

repo = git.Repo(search_parent_directories=True)
git_sha = repo.head.object.hexsha
head_tags = [tag for tag in repo.tags if tag.commit.hexsha == git_sha]

if len(head_tags) > 0:
    git_tag = head_tags[0].name
else:  # Unshallow the repository to get full commit history
    shallow_path = os.path.join(repo.git_dir, "shallow")
    if os.path.exists(shallow_path):
        repo.git.fetch("--unshallow")
    repo.git.fetch("--tags")
    git_tag = next(
        tag.name
        for commit in repo.iter_commits()
        for tag in repo.tags
        if tag.commit.hexsha == commit.hexsha
    )

function_name = construct_name("cf-asa")

# Build the URL without circular dependency
computed_function_url = pulumi.Output.concat(
    "https://",
    region,
    "-",
    project_id,
    ".cloudfunctions.net/",
    function_name,
)

# Batch runs start a GFW query, so the function accepts them only with this
# account's OIDC token, which the Scheduler job below sends. The API key used by
# Cloud Tasks cannot start a batch.
scheduler_service_account = serviceaccount.Account(
    construct_name("cf-asa-scheduler-sa"),
    account_id=f"{stack}-cf-asa-sched",
    display_name="Cloud Scheduler identity for ASA batch runs",
)

config_values = {
    "DB_URL": database.sql_instance_url_with_ip_asyncpg,
    "GIT_HASH": git_sha,
    "GIT_TAG": git_tag,
    "PROJECT_ID": project_id,
    "GCPREGION": region,
    "ASA_QUEUE": queue.name,
    "ASA_BUCKET": bucket.name,
    "FUNCTION_URL": computed_function_url,
    "ASA_SCHEDULER_SA": scheduler_service_account.email,
    "ASA_IS_DRY_RUN": pulumi.Config("cerulean-cloud").require("dryrun_asa"),
}

# The Cloud Function source code itself needs to be zipped up into an
# archive, which we create using the pulumi.AssetArchive primitive.
PATH_TO_SOURCE_CODE = "../cerulean_cloud/cloud_function_asa"
package = pulumi_create_zip(
    dir_to_zip=PATH_TO_SOURCE_CODE,
    zip_filepath="../cloud_function_asa.zip",
)
archive = package.apply(lambda x: pulumi.FileAsset(x))

# Create the single Cloud Storage object, which contains all of the function's
# source code.
source_archive_object = storage.BucketObject(
    construct_name("source-cf-asa"),
    name=f"handler.py-asa-{time.time():f}",
    bucket=bucket.name,
    source=archive,
)

# Assign access to cloud SQL
cloud_function_service_account = serviceaccount.Account(
    construct_name("cf-asa"),
    account_id=f"{stack}-cf-asa",
    display_name="Service Account for cloud function.",
)

cloud_function_service_account_iam = projects.IAMMember(
    construct_name("cf-asa-iam"),
    project=project_id,
    role="projects/cerulean-338116/roles/cloudfunctionasarole",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

# GFW's account (the `ais:credentials` key, shared by every stack) runs the batch's
# AIS query in GFW's project but can't open BigQuery read sessions there, so it
# opens them here to stream the result. Reads bill to this project.
gfw_read_sessions = projects.IAMMember(
    construct_name("cf-asa-gfw-read-sessions"),
    project=project_id,
    role="roles/bigquery.readSessionUser",
    member="serviceAccount:skytruth-cerulean@world-fishing-827.iam.gserviceaccount.com",
)

asa_bucket_access = storage.BucketIAMMember(
    construct_name("cf-asa-bucket-access"),
    bucket=bucket.name,
    role="roles/storage.objectUser",
    member=pulumi.Output.concat(
        "serviceAccount:", cloud_function_service_account.email
    ),
)

# Define secret environment variables
gfw_credentials = {
    "key": "GOOGLE_APPLICATION_CREDENTIALS",
    "secret": pulumi.Config("ais").require("credentials"),
    "version": "latest",
    "project_id": project_id,
}
infra_api_key = {
    "key": "INFRA_API_TOKEN",
    "secret": pulumi.Config("cerulean-cloud").require("infra_keyname"),
    "version": "latest",
    "project_id": project_id,
}
api_key = {
    "key": "API_KEY",
    "secret": pulumi.Config("cerulean-cloud").require("keyname"),
    "version": "latest",
    "project_id": project_id,
}


# Create the Cloud Function (Gen2)
fxn = cloudfunctionsv2.Function(
    function_name,
    name=function_name,
    location=region,
    description="Cloud Function for ASA",
    build_config={
        "runtime": "python311",
        "entry_point": "main",
        "source": {
            "storage_source": {
                "bucket": bucket.name,
                "object": source_archive_object.name,
            },
        },
    },
    service_config={
        "environment_variables": config_values,
        "available_memory": "4096M",
        "timeout_seconds": timeout_seconds,
        "service_account_email": cloud_function_service_account.email,
        "secret_environment_variables": [gfw_credentials, infra_api_key, api_key],
        "vpc_connector": vpc_connector.id,
    },
    opts=pulumi.ResourceOptions(
        depends_on=[cloud_function_service_account_iam, asa_bucket_access],
    ),
)

invoker = cloudfunctionsv2.FunctionIamMember(
    construct_name("cf-asa-invoker"),
    project=fxn.project,
    location=fxn.location,
    cloud_function=fxn.name,
    role="roles/cloudfunctions.invoker",
    member="allUsers",
)

cloud_run_invoker = cloudrun.IamMember(
    "cf-asa-run-invoker",
    project=fxn.project,
    location=fxn.location,
    service=fxn.name,
    role="roles/run.invoker",
    member="allUsers",
)


def batch_target(lookback_days):
    body = {
        "batch": True,
        "lookback_days": lookback_days,
        "pivot": None,
        "retry_incomplete": False,
    }
    return cloudscheduler.JobHttpTargetArgs(
        http_method="POST",
        uri=computed_function_url,
        body=base64.b64encode(json.dumps(body).encode()).decode(),
        headers={"Content-Type": "application/json"},
        oidc_token=cloudscheduler.JobHttpTargetOidcTokenArgs(
            service_account_email=scheduler_service_account.email,
            audience=computed_function_url,
        ),
    )


# cf-alerts reads this job's last attempt (stack/cloud_function_alerts.py).
scheduler_job = cloudscheduler.Job(
    construct_name("cf-asa-scheduler"),
    region=region,
    schedule="0 5 * * *",
    time_zone="Etc/UTC",
    # Each deploy leaves staging's job paused. Resume it in the console to run
    # batches there.
    paused=stack == "staging",
    attempt_deadline=f"{timeout_seconds}s",
    http_target=batch_target(32),
    # A busy batch returns 503, and a timed-out attempt can keep running and hold
    # the lock. Retry 30 min after each failure, giving either time to finish.
    retry_config=cloudscheduler.JobRetryConfigArgs(
        retry_count=4, min_backoff_duration="1800s", max_backoff_duration="1800s"
    ),
    opts=pulumi.ResourceOptions(depends_on=[cloud_run_invoker]),
)

cloudscheduler.Job(
    construct_name("cf-asa-backlog-scheduler"),
    region=region,
    schedule="30 * * * *",
    time_zone="Etc/UTC",
    paused=True,
    attempt_deadline=f"{timeout_seconds}s",
    http_target=batch_target(2557),  # About seven years, including time to drain.
    opts=pulumi.ResourceOptions(
        depends_on=[cloud_run_invoker],
        # Operators choose the campaign window and when to resume or pause it.
        ignore_changes=["paused", "httpTarget.body"],
    ),
)
