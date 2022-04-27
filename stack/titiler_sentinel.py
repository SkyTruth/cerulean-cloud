"""titiler sentinel infra module"""
import pulumi
import pulumi_aws as aws
import pulumi_docker as docker
from pulumi_aws import apigateway, ecr, iam, lambda_
from utils import construct_name

# Build image
repo = ecr.Repository(construct_name("registry"))
image_name = repo.repository_url
registry_info = None  # use ECR credentials helper.

image = docker.Image(
    construct_name("image-titiler-sentinel"),
    build="../cerulean_cloud/titiler_sentinel",
    image_name=image_name,
    registry=registry_info,
)

# Role policy to fetch S3
iam_for_lambda = iam.Role(
    "iamForLambda",
    assume_role_policy="""{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "s3:GetObject",
      "Effect": "Allow",
      "Sid": "",
      "Resources": ["arn:aws:s3:::sentinel-s1-l1c/*"]
    }
  ]
}
""",
)

# Lambda function
lambda_titiler_sentinel = lambda_.Function(
    name=construct_name("lambda-titiler-sentinel"),
    image_uri=image.uri,
    role=iam_for_lambda.arn,
    memory_size=1024,
    environment=lambda_.FunctionEnvironmentArgs(
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


# API gateway LambdaProxyIntegration
api = apigateway.RestAPI(construct_name("apigateway-titiler-sentinel"))
resource = apigateway.Resource(
    "resource", path_part="resource", parent_id=api.root_resource_id, rest_api=api.id
)
method = apigateway.Method(
    "method",
    rest_api=api.id,
    resource_id=resource.id,
    http_method="GET",
    authorization="NONE",
)
integration = aws.apigateway.Integration(
    "integration",
    rest_api=api.id,
    resource_id=resource.id,
    http_method=method.http_method,
    integration_http_method="POST",
    type="AWS_PROXY",
    uri=lambda_titiler_sentinel.invoke_arn,
)

current_identity = aws.get_caller_identity()
current_region = aws.get_region()
apigw_lambda = lambda_.Permission(
    "apigwLambda",
    action="lambda:InvokeFunction",
    function=lambda_.name,
    principal="apigateway.amazonaws.com",
    source_arn=pulumi.Output.all(api.id, method.http_method, resource.path).apply(
        lambda id, http_method, path: f"arn:aws:execute-api:{current_region.name}:{current_identity.account_id}:{id}/*/{http_method}{path}"
    ),
)
# Export URL
