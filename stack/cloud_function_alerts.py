import pulumi
import pulumi_gcp as gcp
import time
from utils import construct_name, pulumi_create_zip
import cloud_run_tipg


PATH_TO_SOURCE_CODE = "../cerulean_cloud/cloud_function_alerts"
resource_name = "cf-alerts"

stack = pulumi.get_stack()

# Create service account for this stack
service_account = gcp.serviceaccount.Account(
    construct_name(f"{resource_name}-service-account"),
    account_id=f"{stack}-{resource_name}",
    display_name="Service account for Slack alert function",
)

# Create bucket to store handler
bucket = gcp.storage.Bucket(
    construct_name(f"{resource_name}-bucket"),
    location=pulumi.Config("gcp").require("region"),
    labels={"pulumi": "true", "environment": pulumi.get_stack()},
)

# Create and zip package to be basis of CloudFunction
package = pulumi_create_zip(
    dir_to_zip=PATH_TO_SOURCE_CODE,
    zip_filepath="../cloud_function_alerts.zip",
)
archive = package.apply(lambda x: pulumi.FileAsset(x))

# Create the Cloud Storage object containing the zipped CloudFunction
source_archive_object = gcp.storage.BucketObject(
    construct_name(f"{resource_name}-source"),
    name=construct_name(f"{resource_name}-source-{time.time():f}"),
    bucket=bucket.name,
    source=archive,
)

# Define secret environment variables
slack_webhooks = {
    "key": "SLACK_ALERTS_WEBHOOK",
    "secret": pulumi.Config("alerts").require("keyname"),
    "version": "latest",
    "project_id": pulumi.Config("gcp").require("project"),
}

# Create the CloudFunction (v2)
fxn = gcp.cloudfunctionsv2.Function(
    construct_name(resource_name),
    name=construct_name(resource_name),
    location=pulumi.Config("gcp").require("region"),
    description="Cloud Function for Pipeline Failure Alerting",
    build_config=gcp.cloudfunctionsv2.FunctionBuildConfigArgs(
        runtime="python311",
        entry_point="main",
        source=gcp.cloudfunctionsv2.FunctionBuildConfigSourceArgs(
            storage_source=gcp.cloudfunctionsv2.FunctionBuildConfigSourceStorageSourceArgs(
                bucket=bucket.name,
                object=source_archive_object.name,
            ),
        ),
    ),
    service_config=gcp.cloudfunctionsv2.FunctionServiceConfigArgs(
        max_instance_count=1,
        available_memory="128Mi",
        timeout_seconds=60,
        ingress_settings="ALLOW_INTERNAL_ONLY",
        environment_variables={
            "GCP_PROJECT": gcp.config.project,
            "TIPG_URL": cloud_run_tipg.default.statuses[0].url,
        },
        secret_environment_variables=[slack_webhooks],
    ),
)

# IAM entry for all users to invoke the function
invoker = gcp.cloudfunctionsv2.FunctionIamMember(
    construct_name(f"{resource_name}-invoker"),
    project=fxn.project,
    location=fxn.location,
    cloud_function=fxn.name,
    role="roles/cloudfunctions.invoker",
    member=pulumi.Output.concat("serviceAccount:", service_account.email),
    opts=pulumi.ResourceOptions(replace_on_changes=["member"]),
)

cloud_run_invoker = gcp.cloudrun.IamMember(
    construct_name(f"{resource_name}-cr-invoker"),
    service=fxn.name,
    location=fxn.location,
    role="roles/run.invoker",
    member=pulumi.Output.concat("serviceAccount:", service_account.email),
    opts=pulumi.ResourceOptions(replace_on_changes=["member"]),
)


http_target = gcp.cloudscheduler.JobHttpTargetArgs(
    http_method="GET",
    uri=fxn.service_config.uri,
    oidc_token=gcp.cloudscheduler.JobHttpTargetOidcTokenArgs(
        service_account_email=service_account.email,
    ),
)


job = gcp.cloudscheduler.Job(
    construct_name(f"{resource_name}-scheduler"),
    description="Run test daily",
    schedule="0 8 * * *",  # 8 AM
    # schedule="every 30 minutes",
    time_zone="America/New_York",
    http_target=http_target,
    opts=pulumi.ResourceOptions(replace_on_changes=["schedule"]),
)
