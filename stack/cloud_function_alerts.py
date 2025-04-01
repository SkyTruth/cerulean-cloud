import pulumi
import pulumi_gcp as gcp
from utils import construct_name, pulumi_create_zip


PATH_TO_SOURCE_CODE = "../cerulean_cloud/cloud_function_alerts"
secret_name = "cerulean-slack-alerts-webhook"
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

# archive = pulumi.AssetArchive({".": pulumi.FileArchive(PATH_TO_SOURCE_CODE)})

# Create the Cloud Storage object containing the zipped CloudFunction
source_archive_object = gcp.storage.BucketObject(
    construct_name(f"{resource_name}-source"),
    name=construct_name(f"{resource_name}-source"),
    bucket=bucket.name,
    source=archive,
)


# Create the CloudFunction (v2)
fxn = gcp.cloudfunctionsv2.Function(
    f"{construct_name(resource_name)}-v2",
    name=f"{construct_name(resource_name)}-v2",
    location=pulumi.Config("gcp").require("region"),
    description="Cloud Function for Pipeline Failure Alerting",
    build_config=gcp.cloudfunctionsv2.FunctionBuildConfigArgs(
        runtime="python311",
        entry_point="main",  # must match `def main(request):`
        source=gcp.cloudfunctionsv2.FunctionBuildConfigSourceArgs(
            storage_source=gcp.cloudfunctionsv2.FunctionBuildConfigSourceStorageSourceArgs(
                bucket=bucket.name,
                object=source_archive_object.name,
            ),
            # source=pulumi.AssetArchive({".": pulumi.FileArchive(PATH_TO_SOURCE_CODE)}
        ),
    ),
    service_config=gcp.cloudfunctionsv2.FunctionServiceConfigArgs(
        max_instance_count=1,
        available_memory="128Mi",
        timeout_seconds=60,
        ingress_settings="ALLOW_ALL",
        environment_variables={
            "SECRET_NAME": secret_name,
            "GCP_PROJECT": gcp.config.project,
        },
    ),
    opts=pulumi.ResourceOptions(replace_on_changes=["http_target", "member"]),
)


secret_access = gcp.secretmanager.SecretIamMember(
    construct_name(f"{resource_name}-secret-access"),
    secret_id=secret_name,
    role="roles/secretmanager.secretAccessor",
    member=pulumi.Output.concat("serviceAccount:", service_account.email),
)


# IAM entry for all users to invoke the function
invoker = gcp.cloudfunctionsv2.FunctionIamMember(
    construct_name(f"{resource_name}-invoker"),
    project=fxn.project,
    location=fxn.location,
    cloud_function=fxn.name,
    role="roles/cloudfunctions.invoker",
    # member=pulumi.Output.concat("serviceAccount:", service_account.email),
    member="allUsers",
)


http_target = gcp.cloudscheduler.JobHttpTargetArgs(
    http_method="GET",
    uri=fxn.service_config.uri,
    # oidc_token=gcp.cloudscheduler.JobHttpTargetOidcTokenArgs(
    #     audience=fxn.service_config.apply(lambda service_config: f"{service_config.uri}/"),
    #     service_account_email=service_account.email,
    # ),
)


# job = gcp.cloudscheduler.Job(
#     construct_name(f"{resource_name}-scheduler-frequent"),
#     description="Run test daily",
#     schedule="0 8 * * *",  # 8 AM
#     time_zone="America/New_York",
#     http_target=http_target,
# )

# TODO: Frequent alert just for testing, remove when finished
job = gcp.cloudscheduler.Job(
    construct_name(f"{resource_name}-scheduler-frequent"),
    description="Run test frequently",
    schedule="every 1 hour",
    time_zone="America/New_York",
    http_target=http_target,
)
