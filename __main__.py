"""A Python Pulumi program"""

import pulumi
from pulumi_gcp import storage
#from pulumi_aws import s3


project = pulumi.get_project()
stack = pulumi.get_stack()

# Create a GCP resource (Storage Bucket)
bucket = storage.Bucket(f'{project}-{stack}-a-bucket', location="EU", labels={"pulumi": "true", "environment": stack})

# Get existing bucket in S3
#s3_bucket = s3.get_bucket(bucket="cerulean-cloud-test")

# Export the DNS name of the bucket
pulumi.export('bucket_name',  bucket.url)
#pulumi.export('s3_bucket_name', s3_bucket.bucket_domain_name)