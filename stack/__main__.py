"""A Python Pulumi program"""

import pulumi

import stack.infra as infra
import stack.titiler_sentinel as titiler_sentinel

# Export the DNS name of the bucket
pulumi.export("bucket_name", infra.bucket.url)
pulumi.export("titiler_sentinel_url", titiler_sentinel.lambda_api.api_endpoint)
