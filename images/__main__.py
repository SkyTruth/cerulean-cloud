"""A Python Pulumi program"""

import cloud_run_images
import pulumi

# Export the DNS name of the bucket
pulumi.export("offset_tile", cloud_run_images.cloud_run_offset_tile_image_url)
pulumi.export("orchestrtor", cloud_run_images.cloud_run_orchestrator_image_url)
