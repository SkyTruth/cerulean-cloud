"""utils for stack building"""
import base64
import hashlib

import pulumi

project = pulumi.get_project()
stack = pulumi.get_stack()


def construct_name(resource_name: str) -> str:
    """construct resource names from project and stack"""
    return f"{project}-{stack}-{resource_name}"


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
