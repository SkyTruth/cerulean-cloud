"""example infra module"""
import pulumi
from pulumi_aws import s3
from pulumi_gcp import storage

project = pulumi.get_project()
stack = pulumi.get_stack()

# Create a GCP resource (Storage Bucket)
bucket = storage.Bucket(
    f"{project}-{stack}-a-bucket",
    location="EU",
    labels={"pulumi": "true", "environment": stack},
)

# Create a AWS resource (Storage Bucket)
s3_bucket = s3.Bucket(
    f"{project}-{stack}-a-bucket", tags={"Pulumi": "true", "Environment": stack}
)
