"""database configurations
"""
import pulumi
import pulumi_gcp as gcp
from utils import construct_name

config = pulumi.Config("gcp")

# See versions at https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/sql_database_instance#database_version
instance = gcp.sql.DatabaseInstance(
    construct_name("database"),
    region=config.require("region"),
    database_version="POSTGRES_14",
    settings=gcp.sql.DatabaseInstanceSettingsArgs(
        tier=config.require("db-instance"),
    ),
    deletion_protection=True,
)
database = gcp.sql.Database("database", instance=instance.name)
