"""database configurations
"""
import pulumi
import pulumi_gcp as gcp
from utils import construct_name

# See versions at https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/sql_database_instance#database_version
instance = gcp.sql.DatabaseInstance(
    construct_name("database"),
    region=pulumi.Config("gcp").require("region"),
    database_version="POSTGRES_14",
    settings=gcp.sql.DatabaseInstanceSettingsArgs(
        tier=pulumi.Config("db").require("db-instance"),
    ),
    deletion_protection=True,
)
database = gcp.sql.Database("database", instance=instance.name)
