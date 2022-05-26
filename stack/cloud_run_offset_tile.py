"""infra for cloud run function to perform inference on offset tiles
Reference doc: https://www.pulumi.com/blog/build-publish-containers-iac/
"""
import pulumi
import pulumi_docker as docker
import pulumi_gcp as gcp
from utils import construct_name, get_file_from_gcs

config = pulumi.Config()

weigths_bucket = config.require("weigths_bucket")
weigths_name = config.require("weigths_name")


registry = gcp.container.Registry(construct_name("registry"), location="EU")
registry_url = registry.id.apply(
    lambda _: gcp.container.get_registry_repository().repository_url
)
cloud_run_offset_tile_image_url = registry_url.apply(
    lambda url: f"{url}/cloud-run-offset-tile-image"
)
registry_info = None  # use gcloud for authentication.

model_weights = get_file_from_gcs(
    weigths_bucket,
    weigths_name,
    out_path="../cerulean_cloud/cloud_run_offset_tiles/model/model.pt",
)
cloud_run_offset_tile_image = docker.Image(
    construct_name("cloud-run-offset-tile-image"),
    build=docker.DockerBuild(
        context="../",
        dockerfile="../Dockerfiles/Dockerfile.cloud_run_offset",
        extra_options=["--no-cache"],
    ),
    image_name=cloud_run_offset_tile_image_url,
    registry=registry_info,
)

default = gcp.cloudrun.Service(
    construct_name("cloud-run-offset-tiles"),
    location=pulumi.Config("gcp").require("region"),
    template=gcp.cloudrun.ServiceTemplateArgs(
        spec=gcp.cloudrun.ServiceTemplateSpecArgs(
            containers=[
                gcp.cloudrun.ServiceTemplateSpecContainerArgs(
                    image=cloud_run_offset_tile_image.base_image_name,
                    envs=[
                        gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                            name="SOURCE",
                            value="remote",
                        ),
                    ],
                    resources=dict(limits=dict(memory="4Gi", cpu="4000m")),
                ),
            ],
        )
    ),
    metadata=gcp.cloudrun.ServiceMetadataArgs(
        annotations={
            "autoscaling.knative.dev/minScale": "1",
        },
    ),
    traffics=[
        gcp.cloudrun.ServiceTrafficArgs(
            percent=100,
            latest_revision=True,
        )
    ],
    autogenerate_revision_name=True,
    opts=pulumi.ResourceOptions(depends_on=[cloud_run_offset_tile_image]),
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
    construct_name("cloud-run-noauth-iam-policy"),
    location=default.location,
    project=default.project,
    service=default.name,
    policy_data=noauth_iam_policy_data.policy_data,
)
