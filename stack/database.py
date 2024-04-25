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
            # flag definitions and allowable values here: https://cloud.google.com/sql/docs/postgres/flags
            dict(name="pg_stat_statements.track", value="all"),
            # Should be 1/4 of total system memory (15Gb)
            # shared_buffers: Converted from 4680 MB to 599040 units of 8KB
            dict(name="shared_buffers", value=pulumi.Config("db").require("db-mem")),
            # Should be slightly higher than expected number of simultaneous connections
            # max_connections: Original value 500, compliant with Google Cloud
            dict(name="max_connections", value="500"),
            # Use for sorting and joining operations. work_mem * max_connections
            # should be less than shared buffers. However, this is the case if
            # we expect `max_connection` to relate to the number of users querying
            # the database. Since we know this is not likely the case we will leave
            # this value at the suggest 50MB
            # work_mem: Converted from 50 MB to 51200 KB
            dict(name="work_mem", value="51200"),
            # Can be significantly higher than work_mem (and necessary
            # in our case due to the costly operations performed on insert)
            # maintenance_work_mem: Converted from 512 MB to 524288 KB
            dict(name="maintenance_work_mem", value="524288"),
            # Has to do with underlying hardwade (value 1.1 is for SSDs,
            # which almost all cloud Postgres instances run, 4 would be
            # for Postgres running on spinning Hard Disk Drive )
            # random_page_cost: Original value 1.1, a float, compliant with Google Cloud
            dict(name="random_page_cost", value="1.1"),
            # Only used by temp tables
            # temp_buffers: Converted from 512 MB to 65536 units of 8KB
            dict(name="temp_buffers", value="65536"),
            # Max number of concurrent i/o processes
            # dict(name="effective_io_concurrency", value="100"),
            # min_wal_size: Converted from 1 GB to 1024 MB
            dict(name="min_wal_size", value="1024"),
            # max_wal_size: Converted from 4 GB to 4096 MB
            dict(name="max_wal_size", value="4096"),
        ],
    ),
)


db_name = construct_name("database")
database = gcp.sql.Database(
    db_name,
    instance=instance.name,
    name=db_name,
    opts=pulumi.ResourceOptions(protect=True),
)
users = gcp.sql.User(
    construct_name("database-users"),
    name=db_name,
    instance=instance.name,
    password=pulumi.Config("db").require_secret("db-password"),
    opts=pulumi.ResourceOptions(depends_on=[database]),
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
