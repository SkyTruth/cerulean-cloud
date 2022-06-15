"""A Python Pulumi program"""

import cloud_run_offset_tile
import cloud_run_orchestrator
import infra as infra
import pulumi
import titiler_sentinel as titiler_sentinel

# Export the DNS name of the bucket
pulumi.export("bucket_name", infra.bucket.url)
pulumi.export("titiler_sentinel_url", titiler_sentinel.lambda_api.api_endpoint)
pulumi.export(
    "cloud_run_offset_tile_url", cloud_run_offset_tile.default.statuses[0].url
)
pulumi.export(
    "cloud_run_orchestrator_url", cloud_run_orchestrator.default.statuses[0].url
)
