"""A Python Pulumi program"""

import infra
import pulumi
import titiler_sentinel

# Export the DNS name of the bucket
pulumi.export("bucket_name", infra.bucket.url)
pulumi.export("titiler_sentinel_url", titiler_sentinel.lambda_deployment.invoke_url)
