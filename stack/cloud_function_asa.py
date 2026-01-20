"""cloud function to find slick culprits"""

import os
import time

import database
import git
import pulumi
from pulumi_gcp import (
    cloudfunctionsv2,
    cloudrun,
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

# We will store the source code to the Cloud Function in a Google Cloud Storage bucket.
bucket = storage.Bucket(
    construct_name("bucket-cf-asa"),
    location="EU",
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
        max_attempts=1,
        max_backoff="300s",
        max_doublings=1,
        max_retry_duration="4s",
        min_backoff="60s",
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

config_values = {
    "DB_URL": database.sql_instance_url_with_ip_asyncpg,
    "GIT_HASH": git_sha,
    "GIT_TAG": git_tag,
    "PROJECT_ID": project_id,
    "GCPREGION": region,
    "ASA_QUEUE": queue.name,
    "FUNCTION_URL": computed_function_url,
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
        "timeout_seconds": 540,
        "service_account_email": cloud_function_service_account.email,
        "secret_environment_variables": [gfw_credentials, infra_api_key, api_key],
        "vpc_connector": vpc_connector.id,
    },
    opts=pulumi.ResourceOptions(
        depends_on=[cloud_function_service_account_iam],
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
