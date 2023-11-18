"""database configurations
"""
import pulumi
import pulumi_gcp as gcp
from utils import construct_name

# See versions at https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/sql_database_instance#database_version
instance = gcp.sql.DatabaseInstance(
    construct_name("database-instance"),
    region=pulumi.Config("gcp").require("region"),
    database_version="POSTGRES_14",
    settings=gcp.sql.DatabaseInstanceSettingsArgs(
        tier=pulumi.Config("db").require("db-instance"),
        backup_configuration=dict(enabled=True),
        # Postgres tuning values ref: https://github.com/developmentseed/how/tree/main/dev/postgresql
        database_flags=[
            dict(name="pg_stat_statments.track", value="ALL"),
            # Should be 1/4 of total system memory (15Gb)
            dict(name="shared_buffers", value="468MB"),
            # Should be slightly higher than expected number of simultaneous connections
            dict(name="max_connections", value="500"),
            # Use for sorting and joining operations. work_mem * max_connections
            # should be less than shared buffers. However, this is the case if
            # we expect `max_connection` to relate to the number of users querying
            # the database. Since we know this is not likely the case we will leave
            # this value at the suggest 50MB
            dict(name="work_mem", value="50MB"),
            # Can be significantly higher than work_mem (and necessary
            # in our case due to the costly operations performed on insert)
            dict(name="maintenance_work_mem", value="512MB"),
            # Has to do with underlying hardwade (value 1.1 is for SSDs,
            # which almost all cloud Postgres instances run, 4 would be
            # for Postgres running on spinning Hard Disk Drive )
            dict(name="random_page_cost", value="1.1"),
            # Only used by temp tables
            dict(name="temp_buffers", value="512MB"),
            # Max number of concurrent i/o processes
            dict(name="effective_io_concurrency", value="100"),
            dict(name="min_wal_size", value="1GB"),
            dict(name="max_wal_size", value="4GB"),
        ],
    ),
    deletion_protection=pulumi.Config("db").require("deletion-protection"),
)
db_name = construct_name("database")
database = gcp.sql.Database(db_name, instance=instance.name, name=db_name)
users = gcp.sql.User(
    construct_name("database-users"),
    name=db_name,
    instance=instance.name,
    password=pulumi.Config("db").require_secret("db-password"),
)

sql_instance_url_with_asyncpg = pulumi.Output.concat(
    "postgresql+asyncpg://",
    db_name,
    ":",
    pulumi.Config("db").require_secret("db-password"),
    "@/",
    db_name,
    "?host=/cloudsql/",
    instance.connection_name,
)
sql_instance_url = pulumi.Output.concat(
    "postgresql://",
    db_name,
    ":",
    pulumi.Config("db").require_secret("db-password"),
    "@/",
    db_name,
    "?host=/cloudsql/",
    instance.connection_name,
)
sql_instance_url_alembic = pulumi.Output.concat(
    "postgresql://",
    db_name,
    ":",
    pulumi.Config("db").require_secret("db-password"),
    "@",
    "127.0.0.1",
    "/",
    db_name,
)
