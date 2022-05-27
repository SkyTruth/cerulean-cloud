"""images for cloud run
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
        extra_options=["--no-cache", "--quiet"],
    ),
    image_name=cloud_run_offset_tile_image_url,
    registry=registry_info,
)
