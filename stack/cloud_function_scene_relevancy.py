"""cloud function to select appropriate scenes (over water and IW) from SNS notification"""
import os
import time

import cloud_run_orchestrator
import database
import pulumi
from pulumi_gcp import cloudfunctions, cloudtasks, projects, serviceaccount, storage
from utils import construct_name

stack = pulumi.get_stack()
# We will store the source code to the Cloud Function in a Google Cloud Storage bucket.
bucket = storage.Bucket(
    construct_name("bucket-cloud-function"),
    location="EU",
    labels={"pulumi": "true", "environment": pulumi.get_stack()},
)

# Create the Queue for tasks
queue = cloudtasks.Queue(
    construct_name("queue-cloud-run-orchestrator"),
    location=pulumi.Config("gcp").require("region"),
    rate_limits=cloudtasks.QueueRateLimitsArgs(
        max_concurrent_dispatches=3,
        max_dispatches_per_second=2,
    ),
    retry_config=cloudtasks.QueueRetryConfigArgs(
        max_attempts=5,
        max_backoff="3s",
        max_doublings=1,
        max_retry_duration="4s",
        min_backoff="2s",
    ),
    stackdriver_logging_config=cloudtasks.QueueStackdriverLoggingConfigArgs(
        sampling_ratio=0.9,
    ),
)

config_values = {
    "DB_URL": database.sql_instance_url,
    "GCP_PROJECT": pulumi.Config("gcp").require("project"),
    "GCP_LOCATION": pulumi.Config("gcp").require("region"),
    "QUEUE": queue.name,
    "ORCHESTRATOR_URL": cloud_run_orchestrator.default.statuses[0].url,
}

# The Cloud Function source code itself needs to be zipped up into an
# archive, which we create using the pulumi.AssetArchive primitive.
PATH_TO_SOURCE_CODE = "../cerulean_cloud/cloud_function_scene_relevancy"
assets = {}
for file in os.listdir(PATH_TO_SOURCE_CODE):
    location = os.path.join(PATH_TO_SOURCE_CODE, file)
    asset = pulumi.FileAsset(path=location)
    assets[file] = asset

archive = pulumi.AssetArchive(assets=assets)

# Create the single Cloud Storage object, which contains all of the function's
# source code. ("main.py" and "requirements.txt".)
source_archive_object = storage.BucketObject(
    construct_name("source-cloud-function-scene-relevancy"),
    name="handler.py-%f" % time.time(),
    bucket=bucket.name,
    source=archive,
)

# Assign access to cloud SQL
cloud_function_service_account = serviceaccount.Account(
    construct_name("cloud-function"),
    account_id=f"{stack}-cloud-function",
    display_name="Service Account for cloud function.",
)
cloud_function_service_account_iam_sql = projects.IAMBinding(
    construct_name("cloud-function-iam-sql"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/cloudsql.client",
    members=[
        cloud_function_service_account.email.apply(
            lambda email: f"serviceAccount:{email}"
        )
    ],
)
cloud_function_service_account_iam_tasks = projects.IAMBinding(
    construct_name("cloud-function-iam-tasks"),
    project=pulumi.Config("gcp").require("project"),
    role="roles/cloudtasks.enqueuer",
    members=[
        cloud_function_service_account.email.apply(
            lambda email: f"serviceAccount:{email}"
        )
    ],
)

fxn = cloudfunctions.Function(
    construct_name("cloud-function-scene-relevancy"),
    entry_point="main",
    environment_variables=config_values,
    region=pulumi.Config("gcp").require("region"),
    runtime="python38",
    source_archive_bucket=bucket.name,
    source_archive_object=source_archive_object.name,
    trigger_http=True,
    service_account_email=cloud_function_service_account.email,
)

invoker = cloudfunctions.FunctionIamMember(
    construct_name("cloud-function-scene-relevancy-invoker"),
    project=fxn.project,
    region=fxn.region,
    cloud_function=fxn.name,
    role="roles/cloudfunctions.invoker",
    member="allUsers",
)
