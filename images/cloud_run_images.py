"""images for cloud run
"""
import hashlib
import os

import pulumi
import pulumi_docker as docker
import pulumi_gcp as gcp
from google.cloud import storage

project = pulumi.get_project()
stack = pulumi.get_stack()


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


def construct_name(resource_name: str) -> str:
    """construct resource names from project and stack"""
    return f"{project}-{stack}-{resource_name}"


def file_hash(path):
    """Calculates a SHA256 hash for a file."""
    with open(path, "rb") as f:
        bytes = f.read()
        return hashlib.sha256(bytes).hexdigest()


def dir_hash(path, substring):
    """Calculates a hash for all files in a directory."""
    hashes = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath) and (substring in filepath):
                hashes.append(file_hash(filepath))
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


config = pulumi.Config()

weigths_bucket = config.require("weigths_bucket")
weigths_name = config.require("weigths_name")


registry = gcp.container.Registry(construct_name("registry"), location="EU")
registry_url = registry.id.apply(
    lambda _: gcp.container.get_registry_repository().repository_url
)
cloud_run_offset_tile_image_url = registry_url.apply(
    lambda url: f"{url}/{construct_name('cloud-run-offset-tile-image')}:{dir_hash('../', 'offset')[:32]}"
)
cloud_run_orchestrator_image_url = registry_url.apply(
    lambda url: f"{url}/{construct_name('cloud-run-orchestrator-image')}:{dir_hash('../', 'orchestrator')[:32]}"
)
cloud_run_tifeatures_image_url = registry_url.apply(
    lambda url: f"{url}/{construct_name('cloud-run-tifeatures-image')}:{dir_hash('../', 'tifeatures')[:32]}"
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
        extra_options=["--quiet"],
    ),
    image_name=cloud_run_offset_tile_image_url,
    registry=registry_info,
)
cloud_run_orchestrator_image = docker.Image(
    construct_name("cloud-run-orchestrator-image"),
    build=docker.DockerBuild(
        context="../",
        dockerfile="../Dockerfiles/Dockerfile.cloud_run_orchestrator",
        extra_options=["--quiet"],
        env={"MODEL": weigths_name},
    ),
    image_name=cloud_run_orchestrator_image_url,
    registry=registry_info,
)
cloud_run_tifeatures_image = docker.Image(
    construct_name("cloud-run-tifeatures-image"),
    build=docker.DockerBuild(
        context="../",
        dockerfile="../Dockerfiles/Dockerfile.cloud_run_tifeatures",
        extra_options=["--quiet"],
    ),
    image_name=cloud_run_tifeatures_image_url,
    registry=registry_info,
)
