"""Historical-run client request contracts and local archive inventory."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from click.testing import CliRunner
from shapely.geometry import box, mapping

from scripts import historical_run as client
from cerulean_cloud.cloud_function_historical_run import main as historical
from cerulean_cloud.cloud_function_historical_run import local as local_config


def test_s3_command_is_removed(remote):
    result = invoke("s3", "--date-start", "2025-01-01", "--date-end", "2025-01-02")
    assert result.exit_code == 2
    assert "No such command 's3'" in result.output
    assert set(client.cli.commands) == {"eodag", "scenes"}
    remote.assert_not_called()


def test_local_eodag_discovery_filters_db_and_queues_selected_scenes(
    local_execution, remote, monkeypatch, tmp_path
):
    processed, trigger, queue = local_execution
    processed.side_effect = [{"completed"}, set()]
    dag = MagicMock()
    dag.search_iter_page.return_value = [
        [
            SimpleNamespace(geometry=box(0, 0, 1, 1), properties={"id": scene})
            for scene in ("new", "completed")
        ]
    ]
    gateway = MagicMock(return_value=dag)
    monkeypatch.setitem(
        sys.modules,
        "eodag",
        SimpleNamespace(EODataAccessGateway=gateway, setup_logging=MagicMock()),
    )
    monkeypatch.setattr(historical, "load_ocean_poly", lambda: box(-5, -5, 5, 5))
    path = tmp_path / "region.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": mapping(box(-5, -5, 5, 5))}
                ],
            }
        )
    )
    result = CliRunner().invoke(
        client.cli,
        [
            "--local",
            "eodag",
            "--date-start",
            "2025-01-01",
            "--date-end",
            "2025-01-02",
            "--geometry",
            str(path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["scene_count"] == 2
    assert summary["already_processed_scene_count"] == 1
    assert summary["queued_count"] == 1
    assert queue.call_args.args[0] == ["new"]
    trigger.assert_awaited_once_with(1, 1, logs_url="")
    dag.set_preferred_provider.assert_called_once_with("geodes")
    assert dag.search_iter_page.call_args.kwargs["polarizationMode"] == "VV VH"
    remote.assert_not_called()


@pytest.fixture
def remote(monkeypatch):
    response = httpx.Response(
        200,
        json={"trigger_id": 42},
        request=httpx.Request("POST", "https://historical.example"),
    )
    post = MagicMock(return_value=response)
    monkeypatch.setattr(client.httpx, "post", post)
    return post


def invoke(*args):
    return CliRunner().invoke(
        client.cli,
        ["--url", "https://historical.example", "--api-key", "test-key", *args],
    )


@pytest.fixture
def local_execution(monkeypatch, remote):
    monkeypatch.setattr(
        local_config,
        "resolve_orchestrator_queue",
        MagicMock(
            side_effect=lambda project,
            region,
            stage: f"cerulean-cloud-{stage}-queue-cr-orchestrator-deployed"
        ),
    )
    for key in ("STACK_SECRETS", "HISTORICAL_DB_URL", "CERULEAN_API_KEY"):
        # Track variables that dotenv may create as well as existing values.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "DB_URL": "postgresql://localhost/test",
        "GCPPROJECT": "project",
        "GCPREGION": "region",
        "QUEUE": "queue",
        "ORCHESTRATOR_URL": "https://orchestrator.example",
        "API_KEY": "test-key",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        client.subprocess,
        "check_output",
        MagicMock(side_effect=AssertionError("no Pulumi")),
    )
    processed = AsyncMock(return_value=set())
    trigger = AsyncMock(side_effect=[{"id": i} for i in range(1, 10)])
    queue = MagicMock(side_effect=lambda scenes, *args, **kwargs: len(scenes))
    monkeypatch.setattr(historical, "get_processed_scene_ids", processed)
    monkeypatch.setattr(historical, "add_trigger_row", trigger)
    monkeypatch.setattr(historical, "handler_queue", queue)

    async def save(record):
        record.id, record.trigger = 123, 1
        return record

    monkeypatch.setattr(client, "history_metadata", MagicMock(return_value={}))
    monkeypatch.setattr(client, "save_historical_run", AsyncMock(side_effect=save))
    return processed, trigger, queue


def test_local_preview_excludes_completed_scenes_without_queueing(
    local_execution, remote
):
    processed, trigger, queue = local_execution
    processed.return_value = {"completed"}
    result = CliRunner().invoke(
        client.cli,
        [
            "--local",
            "scenes",
            "--scene-id",
            "new",
            "--scene-id",
            "completed",
            "--list-only",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "scene_count": 2,
        "filtered_scene_count": 1,
        "already_processed_scene_count": 1,
        "scene_ids": ["new"],
    }
    trigger.assert_not_awaited()
    queue.assert_not_called()
    remote.assert_not_called()
    client.subprocess.check_output.assert_not_called()
    client.save_historical_run.assert_not_awaited()


@pytest.mark.parametrize("dry_run", [False, True])
def test_local_csv_batches_and_rechecks_db_before_each_batch(
    local_execution, remote, tmp_path, dry_run
):
    processed, trigger, queue = local_execution
    scenes = [f"scene-{i}" for i in range(205)]
    path = tmp_path / "scenes.csv"
    path.write_text("scene_id\n" + "\n".join([*scenes, "completed", scenes[0]]) + "\n")
    processed.side_effect = [{"completed"}, set(), {"scene-100"}, set()]
    result = CliRunner().invoke(
        client.cli,
        [
            "--local",
            "scenes",
            "--csv",
            str(path),
            "--manual-trigger",
            "--dry-run" if dry_run else "--no-dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["scene_count"] == 207
    assert summary["already_processed_scene_count"] == 2
    assert summary["filtered_scene_count"] == summary["queued_count"] == 204
    assert summary["dry_run"] is dry_run
    assert [batch["trigger_id"] for batch in summary["batches"]] == [1, 1, 1]
    record = client.save_historical_run.await_args.args[0]
    assert (record.dry_run, record.queued_task_count) == (dry_run, 204)
    assert (record.trigger_.scene_count, record.submission_success) == (207, True)
    assert summary["historical_run_id"] == 123
    assert [len(call.args[0]) for call in queue.call_args_list] == [100, 99, 5]
    assert [scene for call in queue.call_args_list for scene in call.args[0]] == [
        scene for scene in scenes if scene != "scene-100"
    ]
    assert processed.await_count == 4
    trigger.assert_not_awaited()
    assert all(call.kwargs["dry_run"] is dry_run for call in queue.call_args_list)
    assert all(call.kwargs["manual_trigger"] is True for call in queue.call_args_list)
    assert "Batch 3 confirmed:" in result.stderr
    remote.assert_not_called()


@pytest.mark.parametrize(
    "completed,error",
    [
        (0, RuntimeError("17/100 tasks confirmed")),
        (1, RuntimeError("17/100 tasks confirmed")),
        (1, TimeoutError("queue response lost")),
    ],
)
def test_local_submission_failure_stops_and_reports_confirmed_batches(
    local_execution, completed, error
):
    _, trigger, queue = local_execution
    queue.side_effect = [100] * completed + [error]
    args = ["--local", "scenes", "--dry-run"]
    for i in range(205):
        args.extend(["--scene-id", f"scene-{i}"])
    result = CliRunner().invoke(client.cli, args)
    assert result.exit_code != 0
    assert result.exception is error
    assert queue.call_count == completed + 1
    record = client.save_historical_run.await_args.args[0]
    assert record.queued_task_count is None
    assert record.trigger_.filtered_scene_count is None
    assert record.submission_success is False
    assert record.submission_error == repr(error)
    assert ("Batch 1 confirmed:" in result.stderr) is bool(completed)
    assert "Batch 2 confirmed:" not in result.stderr
    assert not result.stdout


def test_local_submission_requires_explicit_dry_run_before_discovery(local_execution):
    processed, trigger, queue = local_execution
    result = CliRunner().invoke(
        client.cli, ["--local", "scenes", "--scene-id", "scene"]
    )
    assert result.exit_code != 0
    assert "requires --dry-run or --no-dry-run" in result.output
    processed.assert_not_awaited()
    trigger.assert_not_awaited()
    queue.assert_not_called()


def test_local_eodag_preview_passes_geometry_and_dates(
    local_execution, tmp_path, monkeypatch
):
    path = tmp_path / "region.geojson"
    geometry = {"type": "FeatureCollection", "features": []}
    path.write_text(json.dumps(geometry))
    search = MagicMock(return_value=(1, ["scene"]))
    monkeypatch.setattr(historical, "handle_search", search)
    result = CliRunner().invoke(
        client.cli,
        [
            "--local",
            "eodag",
            "--date-start",
            "2025-01-01",
            "--date-end",
            "2025-01-02",
            "--geometry",
            str(path),
            "--list-only",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = search.call_args.args[0]
    assert payload["geometry"] == geometry
    assert payload["start"] == "2025-01-01"
    assert payload["end"] == "2025-01-02"
    assert json.loads(result.stdout)["scene_ids"] == ["scene"]


@pytest.mark.parametrize(
    "stage,stack_key",
    [("test", "TEST"), ("staging", "STAGING"), ("prod20240903", "PRODUCTION")],
)
def test_local_env_file_uses_selected_notebook_stack(
    local_execution, tmp_path, stage, stack_key
):
    path = tmp_path / ".env"
    config = {
        stack_key: {
            "DB_URL": "postgresql+asyncpg://localhost/selected-stack",
            "ORCHESTRATOR_URL": "https://selected.example",
            "API_KEY": "selected-key",
        }
    }
    path.write_text("STACK_SECRETS='" + json.dumps(config) + "'\n")
    result = CliRunner().invoke(
        client.cli,
        [
            "--local",
            "--stage",
            stage,
            "--env-file",
            str(path),
            "scenes",
            "--scene-id",
            "scene",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert os.environ["DB_URL"] == "postgresql://localhost/selected-stack"
    assert os.environ["ORCHESTRATOR_URL"] == "https://selected.example"
    assert os.environ["API_KEY"] == "selected-key"
    assert (
        os.environ["QUEUE"] == f"cerulean-cloud-{stage}-queue-cr-orchestrator-deployed"
    )
    assert "selected-key" not in result.output


def test_local_proxy_override_and_preview_need_only_database(
    local_execution, monkeypatch
):
    for key in (
        "DB_URL",
        "GCPPROJECT",
        "GCPREGION",
        "QUEUE",
        "ORCHESTRATOR_URL",
        "API_KEY",
    ):
        monkeypatch.delenv(key)
    monkeypatch.setenv("HISTORICAL_DB_URL", "postgresql+asyncpg://localhost/proxy-db")
    result = CliRunner().invoke(
        client.cli, ["--local", "scenes", "--scene-id", "scene", "--list-only"]
    )
    assert result.exit_code == 0, result.output
    assert os.environ["DB_URL"] == "postgresql://localhost/proxy-db"


def test_local_missing_configuration_fails_before_discovery(
    local_execution, monkeypatch
):
    processed, _, queue = local_execution
    monkeypatch.delenv("DB_URL")
    result = CliRunner().invoke(
        client.cli, ["--local", "scenes", "--scene-id", "scene", "--dry-run"]
    )
    assert result.exit_code != 0
    assert "Local execution requires: DB_URL" in result.output
    processed.assert_not_awaited()
    queue.assert_not_called()


def test_local_all_scenes_completed_submits_no_batches(local_execution):
    processed, trigger, queue = local_execution
    processed.return_value = {"completed"}
    result = CliRunner().invoke(
        client.cli,
        [
            "--local",
            "scenes",
            "--scene-id",
            "completed",
            "--no-dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["already_processed_scene_count"] == 1
    assert summary["queued_count"] == 0
    assert summary["batches"] == []
    trigger.assert_not_awaited()
    queue.assert_not_called()


def test_explicit_scene_ids_csv_and_authenticated_request(remote, tmp_path):
    csv = tmp_path / "scenes.csv"
    csv.write_text("scene_id\nsecond\nfirst\n")
    result = invoke(
        "scenes",
        "--scene-id",
        "first",
        "--csv",
        str(csv),
        "--no-dry-run",
        "--manual-trigger",
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"trigger_id": 42}
    assert remote.call_args.kwargs == {
        "json": {
            "scene_ids": ["first", "second", "first"],
            "action": "enqueue",
            "dry_run": False,
            "manual_trigger": True,
        },
        "headers": {"Authorization": "Bearer test-key"},
        "timeout": 1860,
    }


def test_existing_eodag_cli_selector_is_preserved(remote, tmp_path):
    path = tmp_path / "area.geojson"
    geometry = {"type": "FeatureCollection", "features": []}
    path.write_text(json.dumps(geometry))
    result = invoke(
        "eodag",
        "--date-start",
        "2025-01-01",
        "--date-end",
        "2025-01-02",
        "--geometry",
        str(path),
    )
    assert result.exit_code == 0, result.output
    assert remote.call_args.kwargs["json"] == {
        "start": "2025-01-01",
        "end": "2025-01-02",
        "geometry": geometry,
        "action": "enqueue",
        "manual_trigger": False,
    }


def test_arbitrary_stack_uses_exported_endpoint(remote, monkeypatch):
    pulumi = MagicMock(return_value="https://deployed.example\n")
    monkeypatch.setattr(client.subprocess, "check_output", pulumi)
    monkeypatch.delenv("HISTORICAL_RUN_URL", raising=False)
    result = CliRunner().invoke(
        client.cli,
        [
            "--stage",
            "prod20240903",
            "--api-key",
            "test-key",
            "scenes",
            "--scene-id",
            "first",
        ],
    )
    assert result.exit_code == 0, result.output
    pulumi.assert_called_once_with(
        ["pulumi", "stack", "output", "historical_run_url", "--stack", "prod20240903"],
        text=True,
    )
    assert remote.call_args.args == ("https://deployed.example",)


@pytest.mark.parametrize(
    "arguments",
    [
        ["scenes"],
        ["eodag"],
        ["s3", "--date-start", "2025-01-02", "--date-end", "2025-01-01"],
    ],
)
def test_invalid_cli_requests_do_not_submit(remote, arguments):
    assert invoke(*arguments).exit_code != 0
    remote.assert_not_called()


def test_bad_csv_header_is_reported(remote, tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("wrong\nscene\n")
    result = invoke("scenes", "--csv", str(path))
    assert result.exit_code != 0
    assert "scene_id column" in result.output
    remote.assert_not_called()


def test_remote_http_failure_is_not_reported_as_success(remote):
    remote.return_value = httpx.Response(
        503, request=httpx.Request("POST", "https://historical.example")
    )
    result = invoke("scenes", "--scene-id", "first")
    assert isinstance(result.exception, httpx.HTTPStatusError)


def test_cloud_function_source_imports_from_deployment_root(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    from stack.utils import create_zip
    import zipfile

    source = tmp_path / "deployment"
    source.mkdir()
    archive = create_zip(
        str(repo / "cerulean_cloud/cloud_function_historical_run"),
        str(tmp_path / "function.zip"),
    )
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(source)
    assert (source / "geodes.yml").is_file()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import main; assert main.load_ocean_poly.__module__ == 'relevance'; assert main.handle_search({'scene_ids': ['one', 'one']}) == (2, ['one'])",
        ],
        cwd=source,
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_history_insert_failure_queues_nothing(local_execution):
    processed, _, queue = local_execution
    client.save_historical_run.side_effect = RuntimeError("no historical_run table")
    result = CliRunner().invoke(
        client.cli, ["--local", "scenes", "--scene-id", "scene", "--dry-run"]
    )
    assert str(result.exception) == "no historical_run table"
    processed.assert_not_awaited()
    queue.assert_not_called()


@pytest.mark.parametrize("queue_error", [False, True])
def test_history_update_failure_never_hides_a_queue_error(local_execution, queue_error):
    async def save(record):
        if record.id:
            raise RuntimeError("database failed")
        record.id, record.trigger = 123, 1
        return record

    client.save_historical_run.side_effect = save
    if queue_error:
        local_execution[2].side_effect = RuntimeError("queue failed")
    result = CliRunner().invoke(
        client.cli, ["--local", "scenes", "--scene-id", "scene", "--dry-run"]
    )
    assert str(result.exception) == (
        "queue failed" if queue_error else "database failed"
    )


def test_invalid_local_request_creates_no_history(local_execution, tmp_path):
    geometry = tmp_path / "region.geojson"
    geometry.write_text("{}")
    result = CliRunner().invoke(
        client.cli,
        ["--local", "eodag", "--geometry", str(geometry), "--dry-run"]
        + ["--date-start", "2025-01-02", "--date-end", "2025-01-01"],
    )
    assert "end must be on or after start" in result.output
    client.save_historical_run.assert_not_awaited()
