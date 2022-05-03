"""titiler sentinel infra module"""
import os

import docker
import pulumi
import pulumi_aws as aws
from utils import construct_name


# Build image
def create_package(code_dir: str) -> pulumi.FileArchive:
    """Build docker image and create package."""
    print("Creating lambda package [running in Docker]...")
    client = docker.from_env()

    print("Building docker image...")
    client.images.build(
        path=code_dir,
        dockerfile="Dockerfiles/Dockerfile.titiler",
        tag="titiler-lambda:latest",
        rm=True,
    )

    print("Copying package.zip ...")
    client.containers.run(
        image="titiler-lambda:latest",
        command="/bin/sh -c 'cp /tmp/package.zip /local/package.zip'",
        remove=True,
        volumes={os.path.abspath(code_dir): {"bind": "/local/", "mode": "rw"}},
        user=0,
    )
    return pulumi.FileArchive("../package.zip")


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
    code=create_package("../"),
    runtime="python3.8",
    role=iam_for_lambda.arn,
    memory_size=1024,
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
aws.iam.RolePolicyAttachment(
    construct_name("lambda-titiler-attachment"),
    policy_arn=lambda_s3_policy.arn,
    role=iam_for_lambda.name,
)

# API gateway
lambda_permission = aws.lambda_.Permission(
    construct_name("lambda-titiler-permission"),
    action="lambda:InvokeFunction",
    principal="apigateway.amazonaws.com",
    function=lambda_titiler_sentinel,
)

lambda_titiler_sentinel_url = aws.apigatewayv2.Api(
    construct_name("lambda-titiler-gateway"),
    protocol_type="HTTP",
    route_key="GET /",
    target=lambda_titiler_sentinel.invoke_arn,
)
