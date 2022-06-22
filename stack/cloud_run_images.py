"""images for cloud run
"""
import pulumi
import pulumi_gcp as gcp

project = pulumi.get_project()
stack = pulumi.get_stack()


def construct_name_images(resource_name: str) -> str:
    """construct resource names from project and stack"""
    return f"{project}-images-{stack}-{resource_name}"


registry = gcp.container.Registry(construct_name_images("registry"), location="EU")
registry_url = registry.id.apply(
    lambda _: gcp.container.get_registry_repository().repository_url
)
cloud_run_offset_tile_image_url = registry_url.apply(
    lambda url: f"{url}/{construct_name_images('cloud-run-offset-tile-image:latest')}"
)
cloud_run_orchestrator_image_url = registry_url.apply(
    lambda url: f"{url}/{construct_name_images('cloud-run-orchestrator-image:latest')}"
)
