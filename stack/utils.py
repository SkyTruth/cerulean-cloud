"""utils for stack building"""

import asyncio
import base64
import fnmatch
import hashlib
import os
import zipfile
from tempfile import mkstemp
from typing import List, Optional

import docker
import pulumi
from google.cloud import storage


def construct_name(resource_name: str) -> str:
    """construct resource names from project and stack"""
    return f"{pulumi.get_project()}-{pulumi.get_stack()}-{resource_name}"


def sha256sum(filename):
    """
    Helper function that calculates the hash of a file
    using the SHA256 algorithm
    Inspiration:
    https://stackoverflow.com/a/44873382
    NB: we're deliberately using `digest` instead of `hexdigest` in order to
    mimic Terraform.
    """
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.digest()


def filebase64sha256(filename):
    """
    Computes the Base64-encoded SHA256 hash of a file
    This function mimics its Terraform counterpart, therefore being compatible
    with Pulumi's provisioning engine.
    From https://gist.github.com/LouisAmon/ea395d39d80b28eb78181831fa523456
    """
    h = sha256sum(filename)
    b = base64.b64encode(h)
    return b.decode()


# Build image
def create_package(code_dir: str) -> str:
    """Build docker image and create package."""

    code_dir = os.path.abspath(code_dir)
    assert code_dir
    print(f"Creating lambda package in [{code_dir}] [running in Docker]...")

    client = docker.from_env()
    print("Checking Docker is available...")
    client.ping()

    print("Building container image...")
    image, _ = client.images.build(
        path=code_dir,
        dockerfile="Dockerfiles/Dockerfile.titiler",
        tag="titiler-lambda:latest",
        nocache=True,
        rm=True,
    )
    print(f"Sucessfully built container image with id {image.id}")

    print("Creating installation package.zip ...")
    _ = client.containers.run(
        image="titiler-lambda:latest",
        remove=True,
        volumes={
            code_dir: {
                "bind": "/local/",
                "mode": "rw",
            },
        },
        user=0,
    )

    result = os.path.join(code_dir, "package.zip")
    if not os.path.isfile(result):
        raise RuntimeError(f"Failed to create package.zip at {result}")
    print(f"Sucessfully created package.zip at {result}")
    return result


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


def create_zip(
    dir_to_zip: str,
    zip_filepath: Optional[str] = None,
    ignore_globs: Optional[List[str]] = None,
    compression: int = zipfile.ZIP_DEFLATED,
) -> str:
    """
    Creates a zip file containing the contents of a specified directory. Files matching any of the provided glob patterns will be ignored.

    :param zip_filepath: The path where the output zip file will be created.
    :param dir_to_zip: The directory to recursively add to the zip file.
    :param ignore_globs: A list of glob patterns specifying which files to ignore.
    :param compression: The compression type to use for the zip file (default is ZIP_DEFLATED)
    """
    if not zip_filepath:
        _, zip_filepath = mkstemp(
            suffix=".zip",
        )
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_to_zip):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, dir_to_zip)
                if not any(
                    fnmatch.fnmatch(relative_path, pattern)
                    for pattern in ignore_globs or []
                ):
                    # Store the file relative to the directory specified
                    archive_name = os.path.relpath(
                        full_path, os.path.dirname(dir_to_zip)
                    )
                    zipf.write(full_path, archive_name)
    return zip_filepath


def pulumi_create_zip(
    dir_to_zip: str,
    zip_filepath: Optional[str] = None,
    ignore_globs: Optional[List[str]] = None,
    compression: int = zipfile.ZIP_DEFLATED,
) -> pulumi.Output[str]:
    """
    Creates a zip file containing the contents of a specified directory. Files matching any of the provided glob patterns will be ignored.

    :param zip_filepath: The path where the output zip file will be created.
    :param dir_to_zip: The directory to recursively add to the zip file.
    :param ignore_globs: A list of glob patterns specifying which files to ignore.
    :param compression: The compression type to use for the zip file (default is ZIP_DEFLATED)
    """
    coro = asyncio.to_thread(
        create_zip,
        zip_filepath,
        dir_to_zip,
        ignore_globs,
        compression,
    )
    return pulumi.Output.from_input(coro)
