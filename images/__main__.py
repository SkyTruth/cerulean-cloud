"""A Python Pulumi program"""

import cloud_run_images
import pulumi

# Export the DNS name of the bucket
pulumi.export("infer", cloud_run_images.cloud_run_infer_image_url)
pulumi.export("orchestrator", cloud_run_images.cloud_run_orchestrator_image_url)
pulumi.export("tipg", cloud_run_images.cloud_run_tipg_image_url)
pulumi.export("model", cloud_run_images.weights_name)
