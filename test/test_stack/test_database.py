"""Selecting a connection target must preserve the managed rollback database."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pulumi
import pytest

DB_NAME = "cerulean-cloud-test-database"


class DatabaseMocks(pulumi.runtime.Mocks):
    def __init__(self):
        self.resources = []

    def new_resource(self, args):
        self.resources.append(args)
        if args.typ == "gcp:sql/databaseInstance:DatabaseInstance":
            name = "clone" if args.resource_id else "original"
            return args.resource_id or name, {
                **args.inputs,
                "name": name,
                "connectionName": f"test-project:europe-west1:{name}",
                "ipAddresses": [
                    {
                        "type": "PRIVATE",
                        "ipAddress": "10.0.0.2" if args.resource_id else "10.0.0.1",
                    }
                ],
            }
        return args.name, args.inputs

    def call(self, args):
        return {}


@pytest.mark.parametrize(
    "name", ["", "with space", "db/other", "db?sslmode=disable", "percent%20"]
)
@pulumi.runtime.test
def test_unsupported_database_names_fail_before_registering_resources(
    monkeypatch, name
):
    mocks = DatabaseMocks()
    pulumi.runtime.set_mocks(mocks, project="cerulean-cloud", stack="test")
    pulumi.runtime.set_all_config({"db:connection-database": name})
    monkeypatch.setitem(
        sys.modules,
        "utils",
        SimpleNamespace(construct_name=lambda suffix: f"cerulean-cloud-test-{suffix}"),
    )
    path = Path(__file__).resolve().parents[2] / "stack" / "database.py"
    spec = importlib.util.spec_from_file_location("invalid_database_target", path)
    with pytest.raises(ValueError, match="db:connection-database"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert mocks.resources == []


@pytest.mark.parametrize("connection_instance", [None, "clone"])
@pytest.mark.parametrize("connection_database", [None, "selected-production-v2"])
@pulumi.runtime.test
def test_target_preserves_managed_resources(
    monkeypatch, connection_instance, connection_database
):
    mocks = DatabaseMocks()
    pulumi.runtime.set_mocks(mocks, project="cerulean-cloud", stack="test")
    config = {
        "gcp:project": "test-project",
        "gcp:region": "europe-west1",
        "db:db-instance": "db-custom-4-15360",
        "db:db-mem": "599040",
        "db:db-work-mem": "51200",
        "db:db-parallel-workers": "8",
        "db:db-worker-processes": "8",
        "db:db-password": "test-password",
        "db:deletion-protection": "true",
    }
    if connection_instance:
        config["db:connection-instance"] = connection_instance
    if connection_database:
        config["db:connection-database"] = connection_database
    pulumi.runtime.set_all_config(config)
    monkeypatch.setitem(
        sys.modules,
        "utils",
        SimpleNamespace(construct_name=lambda name: f"cerulean-cloud-test-{name}"),
    )
    path = Path(__file__).resolve().parents[2] / "stack" / "database.py"
    spec = importlib.util.spec_from_file_location("database_under_test", path)
    db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db)
    monkeypatch.setitem(sys.modules, "database", db)
    monkeypatch.setitem(
        sys.modules,
        "cloud_run_images",
        SimpleNamespace(cloud_run_tipg_image=SimpleNamespace(name="test-tipg-image")),
    )
    monkeypatch.setitem(
        sys.modules,
        "cloud_run_infer",
        SimpleNamespace(noauth_iam_policy_data=SimpleNamespace(policy_data="{}")),
    )
    tipg_spec = importlib.util.spec_from_file_location(
        "tipg_under_test", path.with_name("cloud_run_tipg.py")
    )
    tipg = importlib.util.module_from_spec(tipg_spec)
    tipg_spec.loader.exec_module(tipg)

    def check(values):
        name = connection_database or DB_NAME
        connection = f"test-project:europe-west1:{connection_instance or 'original'}"
        ip = "10.0.0.2" if connection_instance else "10.0.0.1"
        assert values["target"] == {
            "project": "test-project",
            "region": "europe-west1",
            "connection_name": connection,
            "private_ip": ip,
            "database": name,
            "port": 5432,
        }
        assert values["target_is_secret"] is False
        assert (
            values["socket"]
            == f"postgresql://{DB_NAME}:test-password@/{name}?host=/cloudsql/{connection}"
        )
        assert values["socket_async"] == values["socket"].replace(
            "postgresql://", "postgresql+asyncpg://", 1
        )
        assert values["ip"] == f"postgresql://{DB_NAME}:test-password@{ip}:5432/{name}"
        assert values["ip_async"] == values["ip"].replace(
            "postgresql://", "postgresql+asyncpg://", 1
        )
        assert (
            values["alembic"]
            == f"postgresql://{DB_NAME}:test-password@127.0.0.1/{name}"
        )
        assert (
            values["managed_user_instance"]
            == values["managed_database_instance"]
            == "original"
        )
        assert values["managed_user_name"] == values["managed_database_name"] == DB_NAME
        managed = [
            r
            for r in mocks.resources
            if r.typ == "gcp:sql/databaseInstance:DatabaseInstance"
            and not r.resource_id
        ]
        assert len(managed) == 1
        assert managed[0].name == "cerulean-cloud-test-database-instance"
        assert managed[0].inputs["databaseVersion"] == "POSTGRES_14"
        assert managed[0].inputs["deletionProtection"] is True
        lookups = [r for r in mocks.resources if r.resource_id]
        assert len(lookups) == bool(connection_instance)
        if lookups:
            assert lookups[0].resource_id == "projects/test-project/instances/clone"
        template = pulumi.runtime.rpc.unwrap_rpc_secret(
            next(
                r.inputs["template"]
                for r in mocks.resources
                if r.typ == "gcp:cloudrun/service:Service"
            )
        )
        assert (
            template["metadata"]["annotations"]["run.googleapis.com/cloudsql-instances"]
            == connection
        )
        env = {
            e["name"]: e.get("value") for e in template["spec"]["containers"][0]["envs"]
        }
        assert env["DATABASE_URL"] == values["socket"]

    return pulumi.Output.all(
        target=db.target,
        target_is_secret=pulumi.Output.is_secret(db.target),
        socket=db.sql_instance_url,
        socket_async=db.sql_instance_url_with_asyncpg,
        ip=db.sql_instance_url_with_ip,
        ip_async=db.sql_instance_url_with_ip_asyncpg,
        alembic=db.sql_instance_url_alembic,
        managed_user_instance=db.users.instance,
        managed_database_instance=db.database.instance,
        managed_user_name=db.users.name,
        managed_database_name=db.database.name,
        tipg=tipg.default.id,
    ).apply(check)
