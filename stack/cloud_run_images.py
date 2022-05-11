"""images for cloud run
"""
import pulumi_docker as docker
import pulumi_gcp as gcp
from utils import construct_name

registry = gcp.container.Registry(construct_name("registry"), location="EU")
registry_url = registry.id.apply(
    lambda _: gcp.container.get_registry_repository().repository_url
)
image_name_with_stack = construct_name("cloud-run-offset-tile-image")
cloud_run_offset_tile_image_url = registry_url.apply(
    lambda url, image_name_with_stack=image_name_with_stack: f"{url}/{image_name_with_stack}"
)
registry_info = None  # use gcloud for authentication.

cloud_run_offset_tile_image = docker.Image(
    construct_name("cloud-run-offset-tile-image"),
    build=docker.DockerBuild(
        context="../",
        dockerfile="../Dockerfiles/Dockerfile.cloud_run_offset",
        extra_options=["--quiet"],
    ),
    image_name=cloud_run_offset_tile_image_url,
    registry=registry_info,
)
