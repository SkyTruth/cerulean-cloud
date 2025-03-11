# vpc_connector.py
import pulumi
from pulumi_gcp import vpcaccess


def reduced_construct_name(resource_name: str) -> str:
    """construct resource names from stack"""
    # This is reduced because the VPC Connector can only have 23 characters!!!
    return f"{pulumi.get_stack()}-{resource_name}"


# Create a shared VPC connector.
vpc_connector = vpcaccess.Connector(
    reduced_construct_name("vpca"),
    name=reduced_construct_name("vpca"),
    region=pulumi.Config("gcp").require("region"),
    network="default",
    ip_cidr_range=pulumi.Config("vpc").require("cidr_range"),
)

pulumi.export("vpc_connector_id", vpc_connector.id)
