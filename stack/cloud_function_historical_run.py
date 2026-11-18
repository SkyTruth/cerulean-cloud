"""Historical scene discovery and the stack's dedicated Cloud Tasks queue."""

import sys
import time
from pathlib import Path

import cloud_function_scene_relevancy
import cloud_run_orchestrator
import database
import pulumi
from pulumi_gcp import (
    cloudfunctionsv2,
    cloudrun,
    cloudtasks,
    storage,
)
from utils import construct_name, pulumi_create_zip
from vpc_connector import vpc_connector

# Pulumi starts this program in stack/, while the queue name is shared with the runner.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cerulean_cloud.cloud_function_historical_run.queues import (  # noqa: E402
    historical_queue_name,
)

stack = pulumi.get_stack()
gcp_config = pulumi.Config("gcp")
queue = cloudtasks.Queue(
    construct_name("queue-cr-orchestrator-historical"),
    name=historical_queue_name(stack),
    project=gcp_config.require("project"),
    location=gcp_config.require("region"),
    rate_limits=cloudtasks.QueueRateLimitsArgs(
        max_concurrent_dispatches=30,
        max_dispatches_per_second=1,
    ),
    retry_config=cloudtasks.QueueRetryConfigArgs(
        max_attempts=3,
        max_backoff="300s",
        max_doublings=1,
        max_retry_duration="4s",
        min_backoff="60s",
    ),
    stackdriver_logging_config=cloudtasks.QueueStackdriverLoggingConfigArgs(
        sampling_ratio=0.9,
    ),
)

function_name = construct_name("cf-historical-run")
config_values = {
    "DB_URL": database.sql_instance_url_with_ip,
    "GCPPROJECT": gcp_config.require("project"),
    "GCPREGION": gcp_config.require("region"),
    "HISTORICAL_RUN_STAGE": stack,
    "ORCHESTRATOR_URL": cloud_run_orchestrator.default.statuses[0].url,
    "FUNCTIONNAME": function_name,
    "SCIHUB_USERNAME": pulumi.Config("scihub").require("username"),
    "SCIHUB_PASSWORD": pulumi.Config("scihub").require("password"),
}

# The Cloud Function source code itself needs to be zipped up into an
# archive.
PATH_TO_SOURCE_CODE = "../cerulean_cloud/cloud_function_historical_run"
package = pulumi_create_zip(
    dir_to_zip=PATH_TO_SOURCE_CODE,
    zip_filepath="../cloud_function_historical_run.zip",
)
archive = package.apply(lambda x: pulumi.FileAsset(x))

# Create the Cloud Storage object containing the function's source code.
source_archive_object = storage.BucketObject(
    construct_name("source-cf-historical-run"),
    name=f"handler.py-hr-{time.time():f}",
    bucket=cloud_function_scene_relevancy.bucket.name,
    source=archive,
)

apikey = {
    "key": "API_KEY",
    "secret": pulumi.Config("cerulean-cloud").require("keyname"),
    "version": "latest",
    "project_id": pulumi.Config("gcp").require("project"),
}

# Create the Cloud Function (Gen2)
fxn = cloudfunctionsv2.Function(
    function_name,
    name=function_name,
    location=pulumi.Config("gcp").require("region"),
    description="Cloud Function for Historical Run",
    build_config={
        "runtime": "python311",
        "entry_point": "main",
        "source": {
            "storage_source": {
                "bucket": cloud_function_scene_relevancy.bucket.name,
                "object": source_archive_object.name,
            },
        },
    },
    service_config={
        "available_memory": "512Mi",
        "environment_variables": config_values,
        "timeout_seconds": 1800,
        "service_account_email": cloud_function_scene_relevancy.cloud_function_service_account.email,
        "secret_environment_variables": [apikey],
        "vpc_connector": vpc_connector.id,
    },
    opts=pulumi.ResourceOptions(
        depends_on=[
            queue,
            cloud_function_scene_relevancy.cloud_function_service_account_iam,
        ],
    ),
)

invoker = cloudfunctionsv2.FunctionIamMember(
    construct_name("cf-historical-run-invoker"),
    project=fxn.project,
    location=fxn.location,
    cloud_function=fxn.name,
    role="roles/cloudfunctions.invoker",
    member="allUsers",
)

cloud_run_invoker = cloudrun.IamMember(
    "cf-historical-run-run-invoker",
    project=fxn.project,
    location=fxn.location,
    service=fxn.name,
    role="roles/run.invoker",
    member="allUsers",
)
