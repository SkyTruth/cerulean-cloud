"""cloud function to select appropriate scenes (over water and IW) from SNS notification"""

import cloud_function_scene_relevancy
import cloud_run_orchestrator
import database
import pulumi
from pulumi_gcp import cloudfunctions, storage
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
# archive, which we create using the pulumi.AssetArchive primitive.
PATH_TO_SOURCE_CODE = "../cerulean_cloud/cloud_function_historical_run"
package = pulumi_create_zip(
    dir_to_zip=PATH_TO_SOURCE_CODE,
    zip_filepath="../cloud_function_historical_run.zip",
)
archive = package.apply(lambda x: pulumi.FileAsset(x))

# Create the single Cloud Storage object, which contains all of the function's
# source code. ("main.py" and "requirements.txt".)
source_archive_object = storage.BucketObject(
    construct_name("source-cf-historical-run"),
    name="handler.py",
    bucket=cloud_function_scene_relevancy.bucket.name,
    source=archive,
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
    source_archive_bucket=cloud_function_scene_relevancy.bucket.name,
    source_archive_object=source_archive_object.name,
    trigger_http=True,
    service_account_email=cloud_function_scene_relevancy.cloud_function_service_account.email,
    timeout=500,
    secret_environment_variables=[apikey],
    opts=pulumi.ResourceOptions(
        depends_on=[cloud_function_scene_relevancy.cloud_function_service_account_iam],
    ),
)

invoker = cloudfunctions.FunctionIamMember(
    construct_name("cf-historical-run-invoker"),
    project=fxn.project,
    region=fxn.region,
    cloud_function=fxn.name,
    role="roles/cloudfunctions.invoker",
    member="allUsers",
)
