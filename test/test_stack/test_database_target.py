"""Database cutover commands must verify serving configs and preserve UI images."""

import copy
import json

import pytest

from scripts import database_target as tool

TARGET = {
    "project": "test-project",
    "region": "europe-west1",
    "connection_name": "test-project:europe-west1:clone",
    "private_ip": "10.0.0.2",
    "database": "selected-production-v2",
    "port": 5432,
}


def ui_config(name, target=None):
    target = target or {**TARGET, "private_ip": "10.0.0.1", "database": "original"}
    return {
        "metadata": {
            "name": name,
            "annotations": {
                "run.googleapis.com/cloudsql-instances": target["connection_name"]
            },
        },
        "spec": {
            "containers": [
                {
                    "image": f"registry.invalid/{name}@sha256:old",
                    "env": [
                        {"name": k, "value": v}
                        for k, v in {
                            "DB_HOST": target["private_ip"],
                            "DB_NAME": target["database"],
                            "DB_PORT": str(target["port"]),
                            "DB_SECRET_NAME": "existing-role-secret",
                            "AUTH_SECRET": "must-not-appear-in-snapshots",
                        }.items()
                    ],
                }
            ]
        },
    }


class FakeCloud:
    def __init__(self):
        self.commands = []
        self.services = {}
        self.revisions = {}
        self.fail_preparation = None
        for name in tool.PROD_UIS:
            active = ui_config(name)
            active["metadata"]["name"] = f"{name}-old"
            self.revisions[f"{name}-old"] = active
            self.services[name] = {
                "metadata": {"name": name},
                "spec": {"template": ui_config(name)},
                "status": {
                    "traffic": [{"revisionName": f"{name}-old", "percent": 100}],
                    "latestCreatedRevisionName": f"{name}-old",
                    "latestReadyRevisionName": f"{name}-old",
                },
            }

    def __call__(self, target, *args):
        self.commands.append(args)
        if args[:3] == ("sql", "databases", "describe"):
            assert args[3] == TARGET["database"]
            return {}
        if args[:3] == ("run", "services", "describe"):
            return copy.deepcopy(self.services[args[3]])
        if args[:3] == ("run", "revisions", "describe"):
            return copy.deepcopy(self.revisions[args[3]])
        name = args[3]
        if args[:3] == ("run", "services", "update"):
            if self.fail_preparation == name:
                raise RuntimeError("new revision failed to start")
            assert "--no-traffic" in args
            env_flag = next(
                a.split("=", 1)[1] for a in args if a.startswith("--update-env-vars=")
            )
            delimiter, values = env_flag[1:].split("^", 1)
            values = dict(v.split("=", 1) for v in values.split(delimiter))
            assert values == {
                "DB_HOST": TARGET["private_ip"],
                "DB_NAME": TARGET["database"],
                "DB_PORT": "5432",
            }
            assert f"--set-cloudsql-instances={TARGET['connection_name']}" in args
            config = ui_config(name, TARGET)
            image = next(a.split("=", 1)[1] for a in args if a.startswith("--image="))
            assert image == f"registry.invalid/{name}@sha256:old"
            config["spec"]["containers"][0]["image"] = image
            self.revisions[f"{name}-new"] = {
                **copy.deepcopy(config),
                "metadata": {**config["metadata"], "name": f"{name}-new"},
            }
            self.services[name]["spec"]["template"] = config
            self.services[name]["status"].update(
                latestCreatedRevisionName=f"{name}-new",
                latestReadyRevisionName=f"{name}-new",
            )
            return {}
        if args[:3] == ("run", "services", "update-traffic"):
            assert all(
                s["status"]["latestReadyRevisionName"].endswith("-new")
                for s in self.services.values()
            )
            assert f"--to-revisions={name}-new=100" in args
            self.services[name]["status"]["traffic"] = [
                {"revisionName": f"{name}-new", "percent": 100}
            ]
            return {}
        raise AssertionError(args)


def test_apply_prepares_both_uis_preserves_images_and_records_previous_settings(
    monkeypatch, tmp_path
):
    cloud = FakeCloud()
    monkeypatch.setattr(tool, "gcloud", cloud)
    monkeypatch.setattr(tool, "backend_reports", lambda *_: [])
    snapshot = tmp_path / "before.json"
    tool.apply_ui(TARGET, tool.PROD_STACK, tool.PROD_UIS, "us-east4", snapshot)
    saved = json.loads(snapshot.read_text())
    assert len(saved["uis"]) == 2
    assert all(
        r["serving"]["environment"]["DB_NAME"] == "original" for r in saved["uis"]
    )
    assert "must-not-appear" not in snapshot.read_text()
    assert snapshot.stat().st_mode & 0o777 == 0o600
    assert (
        len(
            [
                c
                for c in cloud.commands
                if c[:3] == ("run", "services", "update-traffic")
            ]
        )
        == 2
    )


def test_failed_second_preparation_does_not_route_either_ui(monkeypatch, tmp_path):
    cloud = FakeCloud()
    cloud.fail_preparation = tool.PROD_UIS[1]
    monkeypatch.setattr(tool, "gcloud", cloud)
    monkeypatch.setattr(tool, "backend_reports", lambda *_: [])
    with pytest.raises(RuntimeError, match="failed to start"):
        tool.apply_ui(
            TARGET, tool.PROD_STACK, tool.PROD_UIS, "us-east4", tmp_path / "before.json"
        )
    assert not any(
        c[:3] == ("run", "services", "update-traffic") for c in cloud.commands
    )


def test_backend_mismatch_blocks_all_mutations(monkeypatch, tmp_path):
    cloud = FakeCloud()
    monkeypatch.setattr(tool, "gcloud", cloud)
    monkeypatch.setattr(
        tool,
        "backend_reports",
        lambda *_: [{"service": "backend", "errors": ["wrong database"]}],
    )
    with pytest.raises(ValueError, match="wrong database"):
        tool.apply_ui(
            TARGET, tool.PROD_STACK, tool.PROD_UIS, "us-east4", tmp_path / "before.json"
        )
    assert not cloud.commands


def test_verification_checks_serving_revision_even_when_template_is_correct(
    monkeypatch,
):
    cloud = FakeCloud()
    name = tool.PROD_UIS[0]
    cloud.services[name]["spec"]["template"] = ui_config(name, TARGET)
    monkeypatch.setattr(tool, "gcloud", cloud)
    _, _, report = tool.inspect(TARGET, name, "us-east4")
    assert report["errors"]
    assert all(e.startswith("serving revision:") for e in report["errors"])


@pytest.mark.parametrize("backend", tool.BACKENDS)
def test_backend_url_validation_checks_database_and_instance(backend):
    key, socket = tool.BACKENDS[backend]
    path = TARGET["database"]
    url = (
        f"postgresql://user:secret@/{path}?host=/cloudsql/{TARGET['connection_name']}"
        if socket
        else f"postgresql://user:secret@{TARGET['private_ip']}:5432/{path}"
    )
    config = {
        "metadata": {
            "annotations": {
                "run.googleapis.com/cloudsql-instances": TARGET["connection_name"]
            }
        },
        "spec": {"containers": [{"env": [{"name": key, "value": url}]}]},
    }
    assert tool.connection_errors(config, TARGET, backend) == []
    assert tool.connection_errors(config, {**TARGET, "database": "other"}, backend)
    assert tool.connection_errors(
        config,
        {
            **TARGET,
            "private_ip": "10.0.0.99",
            "connection_name": "test-project:europe-west1:other",
        },
        backend,
    )


def test_snapshot_is_never_overwritten(tmp_path):
    path = tmp_path / "before.json"
    path.write_text("original snapshot")
    with pytest.raises(FileExistsError):
        tool.save_snapshot(path, tool.PROD_STACK, TARGET, [])
    assert path.read_text() == "original snapshot"


@pytest.mark.parametrize(
    "changes",
    [
        {"private_ip": None},
        {"connection_name": "wrong-project:europe-west1:clone"},
        {"password": "secret"},
        {"port": 1234},
    ],
)
def test_invalid_target_is_rejected(monkeypatch, changes):
    monkeypatch.setattr(tool, "json_command", lambda _: {**TARGET, **changes})
    with pytest.raises(ValueError):
        tool.load_target(tool.PROD_STACK)


def test_prod_cannot_update_just_one_ui():
    with pytest.raises(SystemExit):
        tool.main(
            [
                "apply-ui",
                "--stack",
                tool.PROD_STACK,
                "--ui-service",
                tool.PROD_UIS[0],
                "--snapshot",
                "unused.json",
            ]
        )
