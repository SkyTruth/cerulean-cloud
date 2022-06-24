"""infra for cloud run function to perform inference on offset tiles
Reference doc: https://www.pulumi.com/blog/build-publish-containers-iac/
"""
import cloud_run_images
import pulumi
import pulumi_gcp as gcp
from utils import construct_name

default = gcp.cloudrun.Service(
    construct_name("cloud-run-offset-tiles"),
    location=pulumi.Config("gcp").require("region"),
    template=gcp.cloudrun.ServiceTemplateArgs(
        spec=gcp.cloudrun.ServiceTemplateSpecArgs(
            containers=[
                gcp.cloudrun.ServiceTemplateSpecContainerArgs(
                    image=cloud_run_images.cloud_run_offset_tile_image.name,
                    envs=[
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="SOURCE",
                            value="remote",
                        ),
                    ],
                    resources=dict(limits=dict(memory="4Gi", cpu="8000m")),
                ),
            ],
            container_concurrency=3,
        )
    ),
    metadata=gcp.cloudrun.ServiceMetadataArgs(
        annotations={
            "autoscaling.knative.dev/minScale": "1",
            "run.googleapis.com/launch-stage": "BETA",
        },
        name=cloud_run_images.cloud_run_offset_tile_sha,
    ),
    traffics=[
        gcp.cloudrun.ServiceTrafficArgs(
            percent=100,
            latest_revision=True,
        )
    ],
)
noauth_iam_policy_data = gcp.organizations.get_iam_policy(
    bindings=[
        gcp.organizations.GetIAMPolicyBindingArgs(
            role="roles/run.invoker",
            members=["allUsers"],
        )
    ]
)
noauth_iam_policy = gcp.cloudrun.IamPolicy(
    construct_name("cloud-run-noauth-iam-policy-offset"),
    location=default.location,
    project=default.project,
    service=default.name,
    policy_data=noauth_iam_policy_data.policy_data,
)
