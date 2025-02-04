"""images for cloud run"""

import pulumi
import pulumi_docker as docker
import pulumi_gcp as gcp
from google.cloud import storage


def get_file_from_gcs(bucket: str, name: str, out_path: str) -> pulumi.FileAsset:
    """Gets a file from GCS and saves it to local, returning a pulumi file asset

    Args:
        bucket (str): a bucket name
        name (str): a object key
        out_path (str): an output path (from stack/)

    Returns:
        pulumi.FileAsset: The output file asset from local file downloaded from GCS
    """
    storage_client = storage.Client()
    gcp_bucket = storage_client.get_bucket(bucket)
    # Create a blob object from the filepath
    blob = gcp_bucket.blob(name)
    # Download the file to a destination
    blob.download_to_filename(out_path)
    return pulumi.FileAsset(out_path)


def construct_name_images(resource_name: str) -> str:
    """construct resource names from project and stack"""
    project = pulumi.get_project()
    stack = pulumi.get_stack()
    # If the project name already ends with "images", don’t add another.
    if project.endswith("images"):
        base = project
    else:
        base = f"{project}-images"
    return f"{base}-{stack}-{resource_name}"


config = pulumi.Config()

weights_bucket = config.require("weights_bucket")
weights_name = config.require("weights_name")

# Create an Artifact Registry repository (DOCKER format) in europe-west1.
repository = gcp.artifactregistry.Repository(
    construct_name_images("registry"),
    repository_id=construct_name_images("registry"),
    format="DOCKER",
    location="europe-west1",
)
# Artifact Registry Docker images are hosted at:
# {location}-docker.pkg.dev/{project}/{repository}
registry_url = pulumi.Output.concat(
    "europe-west1-docker.pkg.dev/", gcp.config.project, "/", repository.repository_id
)
cloud_run_offset_tile_image_url = registry_url.apply(
    lambda url: f"{url}/{construct_name_images('cr-offset-tile-image')}"
)
cloud_run_orchestrator_image_url = registry_url.apply(
    lambda url: f"{url}/{construct_name_images('cr-orchestrator-image')}"
)
cloud_run_tipg_image_url = registry_url.apply(
    lambda url: f"{url}/{construct_name_images('cr-tipg-image')}"
)
registry_info = None  # use gcloud for authentication.

model_weights = get_file_from_gcs(
    weights_bucket,
    weights_name,
    out_path="../cerulean_cloud/cloud_run_offset_tiles/model/model.pt",
)
cloud_run_offset_tile_image = docker.Image(
    construct_name_images("cr-offset-tile-image"),
    build=docker.DockerBuildArgs(
        context="../",
        dockerfile="../Dockerfiles/Dockerfile.cloud_run_offset",
        target="final",
    ),
    image_name=cloud_run_offset_tile_image_url,
    registry=registry_info,
)
cloud_run_orchestrator_image = docker.Image(
    construct_name_images("cr-orchestrator-image"),
    build=docker.DockerBuildArgs(
        context="../",
        dockerfile="../Dockerfiles/Dockerfile.cloud_run_orchestrator",
        target="final",
    ),
    image_name=cloud_run_orchestrator_image_url,
    registry=registry_info,
)
cloud_run_tipg_image = docker.Image(
    construct_name_images("cr-tipg-image"),
    build=docker.DockerBuildArgs(
        context="../",
        dockerfile="../Dockerfiles/Dockerfile.cloud_run_tipg",
        target="final",
    ),
    image_name=cloud_run_tipg_image_url,
    registry=registry_info,
)
