"""cloud function to select appropriate scenes (over water and IW) from SNS notification"""

import time

import cloud_run_orchestrator
import database
import pulumi
from pulumi_gcp import cloudfunctions, cloudtasks, projects, serviceaccount, storage
from utils import construct_name, pulumi_create_zip

stack = pulumi.get_stack()
# We will store the source code to the Cloud Function in a Google Cloud Storage bucket.
bucket = storage.Bucket(
    construct_name("bucket-cf-sr"),
    location="EU",
    labels={"pulumi": "true", "environment": pulumi.get_stack()},
)

# Create the Queue for tasks
queue = cloudtasks.Queue(
    construct_name("queue-cr-orchestrator"),
    location=pulumi.Config("gcp").require("region"),
    rate_limits=cloudtasks.QueueRateLimitsArgs(
        max_concurrent_dispatches=40,
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

function_name = construct_name("cf-sr")
config_values = {
    "DB_URL": database.sql_instance_url,
    "GCPPROJECT": pulumi.Config("gcp").require("project"),
    "GCPREGION": pulumi.Config("gcp").require("region"),
    "QUEUE": queue.name,
    "ORCHESTRATOR_URL": cloud_run_orchestrator.default.statuses[0].url,
    "FUNCTIONNAME": function_name,
    "IS_DRY_RUN": pulumi.Config("cerulean-cloud").require("dryrun_relevancy"),
}

# The Cloud Function source code itself needs to be zipped up into an
# archive, which we create using the pulumi.AssetArchive primitive.
PATH_TO_SOURCE_CODE = "../cerulean_cloud/cloud_function_scene_relevancy"
package = pulumi_create_zip(
    dir_to_zip=PATH_TO_SOURCE_CODE,
    zip_filepath="../cloud_function_scene_relevancy.zip",
)
archive = package.apply(lambda x: pulumi.FileAsset(x))

# Create the single Cloud Storage object, which contains all of the function's
# source code. ("main.py" and "requirements.txt".)
source_archive_object = storage.BucketObject(
    construct_name("source-cf-sr"),
    name=f"handler.py-sr-{time.time():f}",
    bucket=bucket.name,
    source=archive,
)

# Assign access to cloud SQL
cloud_function_service_account = serviceaccount.Account(
    construct_name("cf-sr"),
    account_id=f"{stack}-cf-sr",
    display_name="Service Account for cloud function.",
)
cloud_function_service_account_iam = projects.IAMMember(
    construct_name("cf-sr-iam"),
    project=pulumi.Config("gcp").require("project"),
    role="projects/cerulean-338116/roles/cloudfunctionscenerelevancyrole",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

apikey = cloudfunctions.FunctionSecretEnvironmentVariableArgs(
    key="API_KEY",
    secret=pulumi.Config("cerulean-cloud").require("keyname"),
    version="latest",
    project_id=pulumi.Config("gcp").require("project"),
)


fxn = cloudfunctions.Function(
    function_name,
    name=function_name,
    entry_point="main",
    environment_variables=config_values,
    region=pulumi.Config("gcp").require("region"),
    runtime="python39",
    source_archive_bucket=bucket.name,
    source_archive_object=source_archive_object.name,
    trigger_http=True,
    service_account_email=cloud_function_service_account.email,
    secret_environment_variables=[apikey],
    opts=pulumi.ResourceOptions(
        depends_on=[cloud_function_service_account_iam],
    ),
)

invoker = cloudfunctions.FunctionIamMember(
    construct_name("cf-sr-invoker"),
    project=fxn.project,
    region=fxn.region,
    cloud_function=fxn.name,
    role="roles/cloudfunctions.invoker",
    member="allUsers",
)
