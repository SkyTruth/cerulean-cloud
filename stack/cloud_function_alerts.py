import pulumi
import pulumi_gcp as gcp
from utils import construct_name


secret_name = "cerulean-slack-alerts-webhook"
function_name = "cf-alerts"

service_account = gcp.serviceaccount.Account(
    f"{function_name}-service-account",
    account_id="function-sa",
    display_name="Service account for Slack alert function",
)

cloud_function = gcp.cloudfunctions.Function(
    construct_name(function_name),
    name=construct_name(function_name),
    location=pulumi.Config("gcp").require("region"),
    description="Cloud Function for Pipeline Failure Alerting",
    runtime="python39",
    entry_point="main",
    available_memory_mb=128,
    trigger_http=True,
    service_account_email=service_account.email,
    environment_variables={
        "SECRET_NAME": secret_name,
        "GCP_PROJECT": gcp.config.project,
    },
)


secret_access = gcp.secretmanager.SecretIamMember(
    f"{function_name}-secret-access",
    secret_id=secret_name,
    role="roles/secretmanager.secretAccessor",
    member=pulumi.Output.concat("serviceAccount:", service_account.email),
)


# IAM entry for all users to invoke the function
invoker = gcp.cloudfunctions.FunctionIamMember(
    construct_name(f"{function_name}-invoker"),
    project=cloud_function.project,
    region=cloud_function.region,
    cloud_function=cloud_function.name,
    role="roles/cloudfunctions.invoker",
    member="allUsers",
)

# job = gcp.cloudscheduler.Job(
#     construct_name(f"{function_name}-scheduler-frequent"),
#     description="Run test daily",
#     schedule="0 8 * * *",  # 8 AM
#     time_zone="US/Eastern",
#     http_target=gcp.cloudscheduler.JobHttpTargetArgs(
#         http_method="GET",
#         uri=cloud_function.https_trigger_url,
#     ),
# )


# TODO: Frequent alert just for testing, remove when finished
job = gcp.cloudscheduler.Job(
    construct_name(f"{function_name}-scheduler-frequent"),
    description="Run test frequently",
    schedule="every 5 minutes",
    time_zone="US/Eastern",
    http_target=gcp.cloudscheduler.JobHttpTargetArgs(
        http_method="GET",
        uri=cloud_function.https_trigger_url,
    ),
)
