"""A Python Pulumi program"""

import pulumi
from pulumi_gcp import storage

project = pulumi.get_project()
stack = pulumi.get_stack()

# Create a GCP resource (Storage Bucket)
bucket = storage.Bucket(f'{project}-{stack}-a-bucket', labels={"pulumi": "true", "environment": stack})

# Export the DNS name of the bucket
pulumi.export('bucket_name',  bucket.url)