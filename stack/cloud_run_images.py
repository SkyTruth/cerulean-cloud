"""images for cloud run"""

import pulumi
import pulumi_docker as docker
import pulumi_gcp as gcp

project = pulumi.get_project()
stack = pulumi.get_stack()


def construct_name_images(resource_name: str) -> str:
    """construct resource names from project and stack for images"""
    return f"{project}-images-{stack}-{resource_name}"


# Define the Artifact Registry domain and repository.
artifact_registry_domain = "europe-west1-docker.pkg.dev"
repository_name = construct_name_images("registry")
artifact_registry_url = (
    f"{artifact_registry_domain}/{gcp.config.project}/{repository_name}"
)

# Update the docker provider to use the Artifact Registry domain.
docker_provider = docker.Provider(
    construct_name_images("ar"),
    registry_auth=[{"address": artifact_registry_domain}],
)

# Compute the full image URLs (note no longer using gcp.container.get_registry_image).
cloud_run_offset_tile_image_url = (
    f"{artifact_registry_url}/{construct_name_images('cr-offset-tile-image')}:latest"
)
cloud_run_orchestrator_image_url = (
    f"{artifact_registry_url}/{construct_name_images('cr-orchestrator-image')}:latest"
)
cloud_run_tipg_image_url = (
    f"{artifact_registry_url}/{construct_name_images('cr-tipg-image')}:latest"
)

cloud_run_offset_tile_registry_image = docker.get_registry_image(
    name=cloud_run_offset_tile_image_url,
    opts=pulumi.InvokeOptions(provider=docker_provider),
)
cloud_run_orchestrator_registry_image = docker.get_registry_image(
    name=cloud_run_orchestrator_image_url,
    opts=pulumi.InvokeOptions(provider=docker_provider),
)
cloud_run_tipg_registry_image = docker.get_registry_image(
    name=cloud_run_tipg_image_url,
    opts=pulumi.InvokeOptions(provider=docker_provider),
)

cloud_run_offset_tile_image = docker.RemoteImage(
    construct_name_images("remote-offset"),
    name=cloud_run_offset_tile_registry_image.name,
    pull_triggers=[cloud_run_offset_tile_registry_image.sha256_digest],
    opts=pulumi.ResourceOptions(provider=docker_provider),
)

cloud_run_orchestrator_image = docker.RemoteImage(
    construct_name_images("remote-orchestrator"),
    name=cloud_run_orchestrator_registry_image.name,
    pull_triggers=[cloud_run_orchestrator_registry_image.sha256_digest],
    opts=pulumi.ResourceOptions(provider=docker_provider),
)

cloud_run_tipg_image = docker.RemoteImage(
    construct_name_images("remote-tipg"),
    name=cloud_run_tipg_registry_image.name,
    pull_triggers=[cloud_run_tipg_registry_image.sha256_digest],
    opts=pulumi.ResourceOptions(provider=docker_provider),
)
