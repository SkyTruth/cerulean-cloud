"""cloud function to select appropriate scenes (over water and IW) from SNS notification"""

import time

import cloud_function_scene_relevancy
import cloud_run_orchestrator
import database
import pulumi
import vpc_connector
from pulumi_gcp import (
    cloudfunctionsv2,
    cloudrun,
    storage,
)
from utils import construct_name, pulumi_create_zip

stack = pulumi.get_stack()

function_name = construct_name("cf-historical-run")
config_values = {
    "DB_URL": database.sql_instance_url,
    "GCPPROJECT": pulumi.Config("gcp").require("project"),
    "GCPREGION": pulumi.Config("gcp").require("region"),
    "QUEUE": cloud_function_scene_relevancy.queue.name,
    "ORCHESTRATOR_URL": cloud_run_orchestrator.default.statuses[0].url,
    "FUNCTIONNAME": function_name,
    "SCIHUB_USERNAME": pulumi.Config("scihub").require("username"),
    "SCIHUB_PASSWORD": pulumi.Config("scihub").require("password"),
    "IS_DRY_RUN": pulumi.Config("cerulean-cloud").require("dryrun_historical"),
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
        "runtime": "python39",
        "entry_point": "main",
        "source": {
            "storage_source": {
                "bucket": cloud_function_scene_relevancy.bucket.name,
                "object": source_archive_object.name,
            },
        },
    },
    service_config={
        "environment_variables": config_values,
        "timeout_seconds": 500,
        "service_account_email": cloud_function_scene_relevancy.cloud_function_service_account.email,
        "secret_environment_variables": [apikey],
        "vpc_connector": vpc_connector.id,
    },
    opts=pulumi.ResourceOptions(
        depends_on=[cloud_function_scene_relevancy.cloud_function_service_account_iam],
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
