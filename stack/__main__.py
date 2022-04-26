"""A Python Pulumi program"""

import pulumi
import infra

# Export the DNS name of the bucket
pulumi.export('bucket_name',  infra.bucket.url)
#pulumi.export('s3_bucket_name', s3_bucket.bucket_domain_name)