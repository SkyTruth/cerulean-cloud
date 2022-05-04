"""titiler sentinel infra module"""
import os

import docker
import pulumi
import pulumi_aws as aws
from utils import construct_name

s3_bucket = aws.s3.Bucket(construct_name("titiler-lambda-archive"))


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
    return pulumi.FileArchive(f"{code_dir}package.zip")


lambda_obj = aws.s3.BucketObject(
    construct_name("titiler-lambda-archive"),
    key="package.zip",
    bucket=s3_bucket.id,
    source=create_package("../"),
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
lambda_rest_api = aws.apigateway.RestApi(
    construct_name("lambda-titiler-rest-api"), name="Titiler Sentinel"
)
lambda_resource = aws.apigateway.Resource(
    construct_name("lambda-titiler-resource"),
    rest_api=lambda_rest_api.id,
    parent_id=lambda_rest_api.root_resource_id,
    path_part="{proxy+}",
)

lambda_method = aws.apigateway.Method(
    construct_name("lambda-titiler-method"),
    rest_api=lambda_rest_api.id,
    resource_id=lambda_resource.id,
    http_method="ANY",
    authorization="NONE",
)
lambda_integration = aws.apigateway.Integration(
    construct_name("lambda-titiler-integration"),
    rest_api=lambda_rest_api.id,
    resource_id=lambda_resource.id,
    http_method=lambda_method.http_method,
    integration_http_method="POST",
    type="AWS_PROXY",
    uri=lambda_titiler_sentinel.invoke_arn,
)
lambda_deployment = aws.apigateway.Deployment(
    construct_name("lambda-titiler-deployment"),
    rest_api=lambda_rest_api.id,
    stage_name="default",
    opts=pulumi.ResourceOptions(depends_on=[lambda_method, lambda_integration]),
)
