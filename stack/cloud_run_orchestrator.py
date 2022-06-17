"""infra for cloud run function for orchestration
Reference doc: https://www.pulumi.com/blog/build-publish-containers-iac/
"""
import cloud_run_images
import cloud_run_offset_tile
import pulumi
import pulumi_gcp as gcp
import titiler_sentinel
from utils import construct_name

default = gcp.cloudrun.Service(
    construct_name("cloud-run-orchestrator"),
    location=pulumi.Config("gcp").require("region"),
    template=gcp.cloudrun.ServiceTemplateArgs(
        spec=gcp.cloudrun.ServiceTemplateSpecArgs(
            containers=[
                gcp.cloudrun.ServiceTemplateSpecContainerArgs(
                    image=cloud_run_images.cloud_run_orchestrator_image_url,
                    envs=[
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="TITILER_URL",
                            value=titiler_sentinel.lambda_api.api_endpoint.apply(
                                lambda api_endpoint: api_endpoint
                            ),
                        ),
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="INFERENCE_URL",
                            value=cloud_run_offset_tile.default.statuses.apply(
                                lambda statuses: statuses[0].url
                            ),
                        ),
                    ],
                    resources=dict(limits=dict(memory="2Gi", cpu="4000m")),
                ),
            ],
        )
    ),
    metadata=gcp.cloudrun.ServiceMetadataArgs(
        annotations={
            "autoscaling.knative.dev/minScale": "1",
            "run.googleapis.com/launch-stage": "BETA",
        },
    ),
    traffics=[
        gcp.cloudrun.ServiceTrafficArgs(
            percent=100,
            latest_revision=True,
        )
    ],
    autogenerate_revision_name=True,
    opts=pulumi.ResourceOptions(
        depends_on=[
            titiler_sentinel.lambda_api,
            cloud_run_offset_tile.default,
        ]
    ),
)
