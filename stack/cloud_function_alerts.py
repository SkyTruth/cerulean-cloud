import pulumi
import pulumi_gcp as gcp
from utils import construct_name


cloud_function = gcp.cloudfunctions.Function(
    "function",
    name=construct_name("cf-alerts"),
    location=pulumi.Config("gcp").require("region"),
    description="Cloud Function for Pipeline Failure Alerting",
    runtime="python39",
    entry_point="main",
    available_memory_mb=128,
    trigger_http=True,
)


# IAM entry for all users to invoke the function
invoker = gcp.cloudfunctions.FunctionIamMember(
    construct_name("cf-alerts-invoker"),
    project=cloud_function.project,
    region=cloud_function.region,
    cloud_function=cloud_function.name,
    role="roles/cloudfunctions.invoker",
    member="allUsers",
)

job = gcp.cloudscheduler.Job(
    "daily-alerts-job",
    description="Run test daily",
    schedule="0 8 * * *",  # 8 AM
    time_zone="US/Eastern",
    http_target=gcp.cloudscheduler.JobHttpTargetArgs(
        http_method="GET",
        uri=cloud_function.https_trigger_url,
    ),
)

secret_access = gcp.secretmanager.SecretIamMember(
    "alerts-secret-access",
    secret_id="slack-webhook",
    role="roles/secretmanager.secretAccessor",
    member=pulumi.Output.concat(
        "serviceAccount:", cloud_function.service_account_email
    ),
)
