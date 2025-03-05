# vpc_connector.py
import pulumi
from pulumi_gcp import vpcaccess

# Create a shared VPC connector.
vpc_connector = vpcaccess.Connector(
    "vpc-connector",
    name="cf-vpc-connector",
    region=pulumi.Config("gcp").require("region"),
    network="default",
    ip_cidr_range="10.8.0.0/28",
)

# Optionally export the ID if needed.
pulumi.export("vpc_connector_id", vpc_connector.id)
