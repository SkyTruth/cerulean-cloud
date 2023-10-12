"""cloud function to find slick culprits from AIS tracks"""
import time

import database
import pulumi
from pulumi_gcp import (  # noqa
    cloudfunctions,
    cloudtasks,
    projects,
    serviceaccount,
    storage,
)
from utils import construct_name

stack = pulumi.get_stack()
# We will store the source code to the Cloud Function in a Google Cloud Storage bucket.
bucket = storage.Bucket(
    construct_name("bucket-cloud-function-ais"),
    location="EU",
    labels={"pulumi": "true", "environment": pulumi.get_stack()},
)

# Create the Queue for tasks
queue = cloudtasks.Queue(
    construct_name("queue-cloud-tasks-ais-analysis"),
    location=pulumi.Config("gcp").require("region"),
    rate_limits=cloudtasks.QueueRateLimitsArgs(
        max_concurrent_dispatches=1,
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

function_name = construct_name("cloud-function-ais")
config_values = {
    "DB_URL": database.sql_instance_url,
    "GCP_PROJECT": pulumi.Config("gcp").require("project"),
    "GCP_LOCATION": pulumi.Config("gcp").require("region"),
    "QUEUE": queue.name,
    "FUNCTION_NAME": function_name,
    "API_KEY": pulumi.Config("cerulean-cloud").require("apikey"),
    "IS_DRY_RUN": pulumi.Config("cerulean-cloud").require("dryrun_ais"),
    "BQ_PROJECT_ID": "world-fishing-827",
}

# The Cloud Function source code itself needs to be zipped up into an
# archive, which we create using the pulumi.AssetArchive primitive.
PATH_TO_SOURCE_CODE = "../cerulean_cloud/cloud_function_ais_analysis"
archive = pulumi.FileArchive(PATH_TO_SOURCE_CODE)

# Create the single Cloud Storage object, which contains all of the function's
# source code. ("main.py" and "requirements.txt".)
source_archive_object = storage.BucketObject(
    construct_name("source-cloud-function-ais"),
    name="handler.py-%f" % time.time(),
    bucket=bucket.name,
    source=archive,
)

# Assign access to cloud SQL
cloud_function_service_account = serviceaccount.Account(
    construct_name("cloud-function-ais"),
    account_id=f"{stack}-cloud-function-ais",
    display_name="Service Account for cloud function.",
)

cloud_function_service_account_iam = projects.IAMMember(
    construct_name("cloud-function-ais-iam"),
    project=pulumi.Config("gcp").require("project"),
    role="projects/cerulean-338116/roles/cloudfunctionaisanalysisrole",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

fxn = cloudfunctions.Function(
    function_name,
    name=function_name,
    entry_point="main",
    environment_variables=config_values,
    region=pulumi.Config("gcp").require("region"),
    runtime="python38",
    source_archive_bucket=bucket.name,
    source_archive_object=source_archive_object.name,
    trigger_http=True,
    service_account_email=cloud_function_service_account.email,
)

# invoker = cloudfunctions.FunctionIamMember(
#     construct_name("cloud-function-ais-invoker"),
#     project=fxn.project,
#     region=fxn.region,
#     cloud_function=fxn.name,
#     role="roles/cloudfunctions.invoker",
#     member="allUsers",
# )

config_values["FUNCTION_URL"] = fxn.https_trigger_url
