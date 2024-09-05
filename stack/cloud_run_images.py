"""images for cloud run"""

import pulumi
import pulumi_docker as docker
import pulumi_gcp as gcp

project = pulumi.get_project()
stack = pulumi.get_stack()


def construct_name_images(resource_name: str) -> str:
    """construct resource names from project and stack"""
    return f"{project}-images-{stack}-{resource_name}"


registries = [
    "asia.gcr.io",
    "eu.gcr.io",
    "gcr.io",
    "marketplace.gcr.io",
    "us.gcr.io",
    "staging-k8s.gcr.io",
]

gcr_docker_provider = docker.Provider(
    construct_name_images("gcr"),
    registry_auth=[{"address": a} for a in registries],
)


cloud_run_offset_tile_registry_image = docker.get_registry_image(
    name=gcp.container.get_registry_image(
        name=construct_name_images("cr-offset-tile-image:latest"),
    ).image_url,
    opts=pulumi.InvokeOptions(
        provider=gcr_docker_provider,
    ),
)
cloud_run_orchestrator_registry_image = docker.get_registry_image(
    name=gcp.container.get_registry_image(
        name=construct_name_images("cr-orchestrator-image:latest"),
    ).image_url,
    opts=pulumi.InvokeOptions(
        provider=gcr_docker_provider,
    ),
)
cloud_run_tipg_registry_image = docker.get_registry_image(
    name=gcp.container.get_registry_image(
        name=construct_name_images("cr-tipg-image:latest"),
    ).image_url,
    opts=pulumi.InvokeOptions(
        provider=gcr_docker_provider,
    ),
)


cloud_run_offset_tile_image = docker.RemoteImage(
    construct_name_images("remote-offset"),
    name=cloud_run_offset_tile_registry_image.name,
    pull_triggers=[
        cloud_run_offset_tile_registry_image.sha256_digest,
    ],
    opts=pulumi.ResourceOptions(
        provider=gcr_docker_provider,
    ),
)


cloud_run_orchestrator_image = docker.RemoteImage(
    construct_name_images("remote-orchestrator"),
    name=cloud_run_orchestrator_registry_image.name,
    pull_triggers=[
        cloud_run_orchestrator_registry_image.sha256_digest,
    ],
    opts=pulumi.ResourceOptions(
        provider=gcr_docker_provider,
    ),
)

cloud_run_tipg_image = docker.RemoteImage(
    construct_name_images("remote-tipg"),
    name=cloud_run_tipg_registry_image.name,
    pull_triggers=[
        cloud_run_tipg_registry_image.sha256_digest,
    ],
    opts=pulumi.ResourceOptions(
        provider=gcr_docker_provider,
    ),
)


cloud_run_offset_tile_sha = cloud_run_offset_tile_registry_image.sha256_digest[8:20]
cloud_run_orchestrator_sha = cloud_run_orchestrator_registry_image.sha256_digest[8:20]
cloud_run_tipg_sha = cloud_run_tipg_registry_image.sha256_digest[8:20]
