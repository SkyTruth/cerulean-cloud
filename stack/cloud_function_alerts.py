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
    "daily-test-job",
    description="Run test daily",
    schedule="0 8 * * *",  # 8 AM UTC
    time_zone="UTC",
    http_target=gcp.cloudscheduler.JobHttpTargetArgs(
        http_method="GET",
        uri=cloud_function.https_trigger_url,
    ),
)
