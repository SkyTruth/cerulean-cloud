"""titiler sentinel infra module"""

import asyncio

import pulumi
import pulumi_aws as aws
import pulumi_gcp as gcp
from utils import construct_name, create_package, filebase64sha256

titiler_keyname = pulumi.Config("cerulean-cloud").require("titiler_keyname")
secret = gcp.secretmanager.get_secret(secret_id=titiler_keyname)
titiler_api_key = gcp.secretmanager.get_secret_version_output(
    secret=titiler_keyname
).secret_data


s3_bucket = aws.s3.Bucket(construct_name("titiler-lambda-archive"))

lambda_package_path = pulumi.Output.from_input(asyncio.to_thread(create_package, "../"))
lambda_package_archive = lambda_package_path.apply(lambda x: pulumi.FileArchive(x))
lambda_package_hash = lambda_package_path.apply(lambda x: filebase64sha256(x))

lambda_obj = aws.s3.BucketObject(
    construct_name("titiler-lambda-archive"),
    key="package.zip",
    bucket=s3_bucket.id,
    source=lambda_package_archive,
)

# Role policy to fetch S3
iam_for_lambda = aws.iam.Role(
    construct_name("lambda-titiler-role"),
    assume_role_policy="""{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Effect": "Allow"
    }
  ]
}
""",
)

# Lambda function
lambda_titiler_sentinel = aws.lambda_.Function(
    resource_name=construct_name("lambda-titiler-sentinel"),
    s3_bucket=s3_bucket.id,
    s3_key=lambda_obj.key,
    source_code_hash=lambda_package_hash,
    runtime="python3.9",
    role=iam_for_lambda.arn,
    memory_size=3008,
    timeout=10,
    handler="handler.handler",
    environment=aws.lambda_.FunctionEnvironmentArgs(
        variables={
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
            "GDAL_CACHEMAX": "200",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "GDAL_INGESTED_BYTES_AT_OPEN": "32768",
            "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
            "GDAL_HTTP_MULTIPLEX": "YES",
            "GDAL_HTTP_VERSION": "2",
            "PYTHONWARNINGS": "ignore",
            "VSI_CACHE": "TRUE",
            "VSI_CACHE_SIZE": "5000000",
            "AWS_REQUEST_PAYER": "requester",
            "API_KEY": titiler_api_key,
            "RIO_TILER_MAX_THREADS": "1",
        },
    ),
)

lambda_s3_policy = aws.iam.Policy(
    construct_name("lambda-titiler-policy"),
    description="IAM policy for Lambda to interact with S3",
    policy="""{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::sentinel-s1-l1c/*",
      "Effect": "Allow"
    }
  ]}""",
)

_ = aws.iam.RolePolicyAttachment(
    construct_name("lambda-titiler-attachment"),
    policy_arn=lambda_s3_policy.arn,
    role=iam_for_lambda.name,
)
_ = aws.iam.RolePolicyAttachment(
    construct_name("lambda-titiler-attachment2"),
    policy_arn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    role=iam_for_lambda.name,
)

# API gateway
lambda_permission = aws.lambda_.Permission(
    construct_name("lambda-titiler-permission"),
    action="lambda:InvokeFunction",
    principal="apigateway.amazonaws.com",
    function=lambda_titiler_sentinel,
)
lambda_api = aws.apigatewayv2.Api(
    construct_name("lambda-titiler-api"), protocol_type="HTTP"
)
lambda_integration = aws.apigatewayv2.Integration(
    construct_name("lambda-titiler-integration"),
    api_id=lambda_api.id,
    integration_type="AWS_PROXY",
    integration_uri=lambda_titiler_sentinel.invoke_arn,
)
lambda_route = aws.apigatewayv2.Route(
    construct_name("lambda-titiler-route"),
    api_id=lambda_api.id,
    route_key="ANY /{proxy+}",
    target=pulumi.Output.concat("integrations/", lambda_integration.id),
)
lambda_stage = aws.apigatewayv2.Stage(
    construct_name("lambda-titiler-stage"),
    api_id=lambda_api.id,
    name="$default",
    auto_deploy=True,
    opts=pulumi.ResourceOptions(depends_on=[lambda_route]),
)

api_abuse_alerts_topic = aws.sns.Topic(construct_name("lambda-APIAbuseAlert"))

lambda_invocations_alarm = aws.cloudwatch.MetricAlarm(
    resource_name=construct_name("lambda-titiler-alarm"),
    comparison_operator="GreaterThanThreshold",
    evaluation_periods=1,
    metric_name="Invocations",
    namespace="AWS/Lambda",
    period=3600,  # 1 Hour in seconds
    statistic="Sum",
    threshold=30000,
    dimensions={"FunctionName": lambda_titiler_sentinel.name},
    alarm_description="Alarm when the function's invocations exceed 30,000 within 1 hour.",
    alarm_actions=[api_abuse_alerts_topic.arn],
    actions_enabled=True,
)

email_subscription = aws.sns.TopicSubscription(
    resource_name=construct_name("lambda-titiler-email-tech"),
    topic=api_abuse_alerts_topic.arn,
    protocol="email",
    endpoint="tech+cerulean@skytruth.org",
)

email_subscription = aws.sns.TopicSubscription(
    resource_name=construct_name("lambda-titiler-email-jona"),
    topic=api_abuse_alerts_topic.arn,
    protocol="email",
    endpoint="jona@skytruth.org",
)

email_subscription = aws.sns.TopicSubscription(
    resource_name=construct_name("lambda-titiler-email-aemon"),
    topic=api_abuse_alerts_topic.arn,
    protocol="email",
    endpoint="aemon@skytruth.org",
)

email_subscription = aws.sns.TopicSubscription(
    resource_name=construct_name("lambda-titiler-email-jason"),
    topic=api_abuse_alerts_topic.arn,
    protocol="email",
    endpoint="jason@skytruth.org",
)
