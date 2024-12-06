"""cloud function to find slick culprits from AIS tracks"""

import time

import database
import git
import pulumi
from pulumi_gcp import cloudfunctions, cloudtasks, projects, serviceaccount, storage
from utils import construct_name, pulumi_create_zip

stack = pulumi.get_stack()
# We will store the source code to the Cloud Function in a Google Cloud Storage bucket.
bucket = storage.Bucket(
    construct_name("bucket-cf-ais"),
    location="EU",
    labels={"pulumi": "true", "environment": pulumi.get_stack()},
)

# Create the Queue for tasks
queue = cloudtasks.Queue(
    construct_name("queue-cloud-tasks-ais-analysis"),
    location=pulumi.Config("gcp").require("region"),
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

function_name = construct_name("cf-ais")
config_values = {
    "DB_URL": database.sql_instance_url_with_asyncpg,
    "GIT_HASH": git_sha,
}

# The Cloud Function source code itself needs to be zipped up into an
# archive, which we create using the pulumi.AssetArchive primitive.
PATH_TO_SOURCE_CODE = "../cerulean_cloud/cloud_function_ais_analysis"
package = pulumi_create_zip(
    dir_to_zip=PATH_TO_SOURCE_CODE,
    zip_filepath="../cloud_function_ais_analysis.zip",
)
archive = package.apply(lambda x: pulumi.FileAsset(x))

# Create the single Cloud Storage object, which contains all of the function's
# source code. ("main.py" and "requirements.txt".)
source_archive_object = storage.BucketObject(
    construct_name("source-cf-ais"),
    name=f"handler.py-{time.time():f}",
    bucket=bucket.name,
    source=archive,
)

# Assign access to cloud SQL
cloud_function_service_account = serviceaccount.Account(
    construct_name("cf-ais"),
    account_id=f"{stack}-cf-ais",
    display_name="Service Account for cloud function.",
)

cloud_function_service_account_iam = projects.IAMMember(
    construct_name("cf-ais-iam"),
    project=pulumi.Config("gcp").require("project"),
    role="projects/cerulean-338116/roles/cloudfunctionaisanalysisrole",
    member=cloud_function_service_account.email.apply(
        lambda email: f"serviceAccount:{email}"
    ),
)

gfw_credentials = cloudfunctions.FunctionSecretEnvironmentVariableArgs(
    key="GOOGLE_APPLICATION_CREDENTIALS",
    secret=pulumi.Config("ais").require("credentials"),
    version="latest",
    project_id=pulumi.Config("gcp").require("project"),
)


api_key = cloudfunctions.FunctionSecretEnvironmentVariableArgs(
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
    available_memory_mb=4096,
    timeout=540,
    secret_environment_variables=[gfw_credentials, api_key],
    opts=pulumi.ResourceOptions(
        depends_on=[cloud_function_service_account_iam],
    ),
)

invoker = cloudfunctions.FunctionIamMember(
    construct_name("cf-ais-invoker"),
    project=fxn.project,
    region=fxn.region,
    cloud_function=fxn.name,
    role="roles/cloudfunctions.invoker",
    member="allUsers",
)
