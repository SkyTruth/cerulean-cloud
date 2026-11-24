"""Historical-run client request contracts and local archive inventory."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from click.testing import CliRunner
from shapely.geometry import box, mapping
from sqlalchemy.engine import make_url

from scripts import historical_run as client
from cerulean_cloud.cloud_function_historical_run import geodes_catalog
from cerulean_cloud.cloud_function_historical_run import main as historical


def test_s3_command_is_removed(remote):
    result = invoke("s3", "--date-start", "2025-01-01", "--date-end", "2025-01-02")
    assert result.exit_code == 2
    assert "No such command 's3'" in result.output
    assert set(client.cli.commands) == {"eodag", "scenes"}
    remote.assert_not_called()


@pytest.mark.parametrize("selector", [[], ["eodag"]])
@pytest.mark.parametrize("with_geometry", [False, True])
def test_local_eodag_discovery_filters_db_and_queues_selected_scenes(
    local_execution, remote, monkeypatch, tmp_path, selector, with_geometry
):
    processed, trigger, queue = local_execution
    processed.side_effect = [{"S1A_IW_GRDM_completed"}, set()]
    search = MagicMock(return_value=(3, ["S1A_IW_GRDH_new", "S1A_IW_GRDM_completed"]))
    monkeypatch.setattr(geodes_catalog, "search_catalog", search)
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
            *selector,
            "--date-start",
            "2025-01-01",
            "--date-end",
            "2025-01-02",
            *(["--geometry", str(path)] if with_geometry else []),
            "--mode",
            "EW",
            "--mode",
            "IW",
            "--mode",
            "SM",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["scene_count"] == 3
    assert summary["already_processed_scene_count"] == 1
    assert summary["queued_count"] == 1
    assert queue.call_args.args[0] == ["S1A_IW_GRDH_new"]
    trigger.assert_not_awaited()  # Local batches reuse the history record's trigger.
    assert queue.call_args.args[1] == summary["trigger_id"]
    assert client.save_historical_run.await_count == 2
    criteria = search.call_args.args[0]
    assert json.loads(criteria["polarizationMode"]) == ["VV", "VV VH"]
    assert json.loads(criteria["sensorMode"]) == [
        "EW",
        "IW",
        "SM",
    ]
    assert queue.call_args.kwargs["dry_run"] is True
    assert ("geom" in criteria) is with_geometry
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
    for key in (
        "STACK_SECRETS",
        "HISTORICAL_DB_URL",
        "CERULEAN_API_KEY",
        "HISTORICAL_BATCH_WORKERS",
        "HISTORICAL_RUN_STAGE",
    ):
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
    monkeypatch.setattr(
        historical, "cloud_tasks_session", MagicMock(side_effect=MagicMock)
    )

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
    assert "batches" not in summary
    record = client.save_historical_run.await_args.args[0]
    assert (record.dry_run, record.queued_task_count) == (dry_run, 204)
    assert (record.trigger_.scene_count, record.submission_success) == (207, True)
    assert summary["historical_run_id"] == 123
    calls = sorted(queue.call_args_list, key=lambda call: call.kwargs["batch_number"])
    assert {call.args[1] for call in calls} == {summary["trigger_id"]}
    assert [len(call.args[0]) for call in calls] == [100, 99, 5]
    assert [scene for call in calls for scene in call.args[0]] == [
        scene for scene in scenes if scene != "scene-100"
    ]
    assert processed.await_count == 4
    trigger.assert_not_awaited()
    assert all(call.kwargs["dry_run"] is dry_run for call in queue.call_args_list)
    assert all(call.kwargs["manual_trigger"] is True for call in queue.call_args_list)
    assert "Batch 3 confirmed:" in result.stderr
    remote.assert_not_called()


def test_local_throughput_spaces_batches_and_reports_window(local_execution, tmp_path):
    queue = local_execution[2]
    start = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=2)
    path = tmp_path / "scenes.csv"
    path.write_text("scene_id\n" + "\n".join(f"scene-{i}" for i in range(205)) + "\n")
    result = CliRunner().invoke(
        client.cli,
        [
            "--local",
            "scenes",
            "--csv",
            str(path),
            "--throughput",
            "4000",
            "--release-start",
            start.isoformat(),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["release_rate"] == 3600
    assert summary["release_start"] == start.isoformat()
    assert summary["release_end"] == (start + timedelta(seconds=205)).isoformat()
    calls = sorted(queue.call_args_list, key=lambda call: call.kwargs["batch_number"])
    assert [call.kwargs["schedule_time"] for call in calls] == [
        start + timedelta(seconds=seconds) for seconds in (0, 100, 200)
    ]


def test_local_release_defaults_to_submission_time(local_execution):
    before = datetime.now(timezone.utc)
    result = CliRunner().invoke(
        client.cli,
        ["--local", "scenes", "--scene-id", "scene", "--throughput", "3000"],
    )
    assert result.exit_code == 0, result.output
    released = local_execution[2].call_args.kwargs["schedule_time"]
    assert before.replace(microsecond=0) <= released <= datetime.now(timezone.utc)
    assert json.loads(result.stdout)["release_start"] == released.isoformat(
        timespec="seconds"
    )


def test_local_release_past_cloud_tasks_limit_queues_nothing(local_execution):
    result = CliRunner().invoke(
        client.cli,
        [
            "--local",
            "scenes",
            "--scene-id",
            "a",
            "--scene-id",
            "b",
            "--throughput",
            "0.001",
        ],
    )
    assert result.exit_code == 1
    assert "30-day scheduling limit" in result.output
    local_execution[2].assert_not_called()
    record = client.save_historical_run.await_args.args[0]
    assert record.submission_error == "ReleaseScheduleError"


def test_remote_release_options_are_sent(remote):
    result = invoke(
        "scenes",
        "--scene-id",
        "scene",
        "--throughput",
        "3200",
        "--release-start",
        "2026-11-25T06:00:00Z",
    )
    assert result.exit_code == 0, result.output
    payload = remote.call_args.kwargs["json"]
    assert payload["throughput"] == 3200
    assert payload["release_start"] == "2026-11-25T06:00:00Z"


@pytest.mark.parametrize(
    "flags",
    [
        ["--release-start", "2026-11-25T06:00:00Z"],
        ["--throughput", "0"],
        ["--throughput", "fast"],
    ],
)
def test_invalid_release_options_do_not_submit(remote, flags):
    assert invoke("scenes", "--scene-id", "scene", *flags).exit_code != 0
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
    local_execution, monkeypatch, completed, error
):
    monkeypatch.setenv("HISTORICAL_BATCH_WORKERS", "1")
    _, trigger, queue = local_execution
    queue.side_effect = [100] * completed + [error]
    args = ["--local", "scenes", "--dry-run"]
    for i in range(205):
        args.extend(["--scene-id", f"scene-{i}"])
    result = CliRunner().invoke(client.cli, args)
    assert result.exit_code != 0
    assert isinstance(result.exception, type(error))
    assert result.exception.args == error.args
    assert queue.call_count == completed + 1
    record = client.save_historical_run.await_args.args[0]
    assert record.queued_task_count is None
    assert record.trigger_.filtered_scene_count is None
    assert record.submission_success is False
    assert record.submission_error == type(error).__name__
    assert ("Batch 1 confirmed:" in result.stderr) is bool(completed)
    assert "Batch 2 confirmed:" not in result.stderr
    assert not result.stdout


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], True),
        (["--dry-run"], True),
        (["--dry-run=true"], True),
        (["--dry-run", "True"], True),
        (["--dry-run=false"], False),
        (["--dry-run", "False"], False),
        (["--no-dry-run"], False),
    ],
)
def test_local_dry_run_flags_control_history_and_queued_tasks(
    local_execution, flags, expected
):
    queue = local_execution[2]
    result = CliRunner().invoke(
        client.cli, ["--local", "scenes", "--scene-id", "scene", *flags]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["dry_run"] is expected
    assert queue.call_args.kwargs["dry_run"] is expected
    assert client.save_historical_run.await_args.args[0].dry_run is expected


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], True),
        (["--dry-run"], True),
        (["--dry-run=True"], True),
        (["--dry-run=false"], False),
        (["--no-dry-run"], False),
    ],
)
def test_remote_dry_run_flags_are_sent_explicitly(remote, flags, expected):
    result = invoke("scenes", "--scene-id", "scene", *flags)
    assert result.exit_code == 0, result.output
    assert remote.call_args.kwargs["json"]["dry_run"] is expected


@pytest.mark.parametrize(
    "flags",
    [
        ["--dry-run=invalid"],
        ["--dry-run="],
        ["--dry-run=false", "--no-dry-run"],
        ["--no-dry-run", "--dry-run"],
    ],
)
def test_invalid_dry_run_flags_stop_before_local_side_effects(local_execution, flags):
    result = CliRunner().invoke(
        client.cli, ["--local", "scenes", "--scene-id", "scene", *flags]
    )
    assert result.exit_code == 2
    client.save_historical_run.assert_not_awaited()
    local_execution[0].assert_not_awaited()
    local_execution[2].assert_not_called()


@pytest.mark.parametrize("selector", [[], ["eodag"]])
def test_local_eodag_preview_passes_geometry_and_dates(
    local_execution, tmp_path, monkeypatch, selector
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
            *selector,
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
@pytest.mark.parametrize(
    "legacy_queue", [None, "cerulean-cloud-test-queue-cr-orchestrator-live"]
)
def test_local_env_file_uses_selected_notebook_stack(
    local_execution, tmp_path, stage, stack_key, legacy_queue
):
    path = tmp_path / ".env"
    config = {
        stack_key: {
            "DB_URL": "postgresql+asyncpg://localhost/selected-stack",
            "ORCHESTRATOR_URL": "https://selected.example",
            "API_KEY": "selected-key",
        }
    }
    if legacy_queue is not None:
        config[stack_key].update(QUEUE=legacy_queue, HISTORICAL_QUEUE=legacy_queue)
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
        local_execution[2].call_args.kwargs["queue_name"]
        == f"orchestrator-backlog-{stage}"
    )
    assert os.environ["QUEUE"] == "queue"  # Legacy routing settings are ignored.
    assert "selected-key" not in result.output


def test_local_submission_needs_no_queue_configuration(local_execution, monkeypatch):
    monkeypatch.delenv("QUEUE")
    monkeypatch.delenv("HISTORICAL_QUEUE", raising=False)
    result = CliRunner().invoke(
        client.cli, ["--local", "scenes", "--scene-id", "scene"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["queued_count"] == 1


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


@pytest.mark.parametrize("source", ["STACK_SECRETS", "DB_URL", "HISTORICAL_DB_URL"])
@pytest.mark.parametrize("selector", [[], ["scenes"]])
@pytest.mark.parametrize("use_proxy", [False, True])
def test_local_proxy_flags_preserve_database_configuration(
    local_execution, monkeypatch, source, selector, use_proxy
):
    db_url = (
        "postgresql+asyncpg://reader:p%40ss%3Aword@/selected-stack"
        "?host=%2Fcloudsql%2Fproject%3Aregion%3Ainstance&port=5432"
        "&application_name=historical-cli"
    )
    if source in ("STACK_SECRETS", "HISTORICAL_DB_URL"):
        monkeypatch.setenv(
            "STACK_SECRETS",
            json.dumps(
                {
                    "TEST": {
                        "DB_URL": db_url
                        if source == "STACK_SECRETS"
                        else "postgresql://other:password@other-host/other-db",
                    }
                }
            ),
        )
    if source != "STACK_SECRETS":
        monkeypatch.setenv(source, db_url)
    monkeypatch.setattr(
        historical, "handle_search", MagicMock(return_value=(1, ["scene"]))
    )
    flags = ["--db-host", "127.0.0.1", "--db-port", "5434"] if use_proxy else []
    selection = (
        ["scenes", "--scene-id", "scene"]
        if selector
        else ["--date-start", "2025-01-01", "--date-end", "2025-01-02"]
    )
    result = CliRunner().invoke(
        client.cli, ["--local", "--stage", "test", *flags, *selection, "--list-only"]
    )
    assert result.exit_code == 0, result.output
    if use_proxy:
        url = make_url(os.environ["DB_URL"])
        assert (url.host, url.port) == ("127.0.0.1", 5434)
        assert (url.username, url.password, url.database) == (
            "reader",
            "p@ss:word",
            "selected-stack",
        )
        assert dict(url.query) == {"application_name": "historical-cli"}
    else:
        assert os.environ["DB_URL"] == db_url.replace(
            "postgresql+asyncpg://", "postgresql://", 1
        )
    assert "p@ss:word" not in result.output
    local_execution[2].assert_not_called()
    client.save_historical_run.assert_not_awaited()


@pytest.mark.parametrize(
    "flags,expected_host,expected_port",
    [
        (["--db-host", "127.0.0.1"], "127.0.0.1", 5432),
        (["--db-port", "5434"], "existing-host", 5434),
    ],
)
def test_local_database_flags_can_override_one_component(
    local_execution, monkeypatch, flags, expected_host, expected_port
):
    monkeypatch.setenv("DB_URL", "postgresql://reader:password@existing-host:5432/db")
    result = CliRunner().invoke(
        client.cli, ["--local", *flags, "scenes", "--scene-id", "scene"]
    )
    assert result.exit_code == 0, result.output
    url = make_url(os.environ["DB_URL"])
    assert (url.host, url.port) == (expected_host, expected_port)
    assert json.loads(result.stdout)["queued_count"] == 1


@pytest.mark.parametrize("flags", [["--db-host", "127.0.0.1"], ["--db-port", "5434"]])
def test_database_flags_require_local_execution(remote, flags):
    result = invoke(*flags, "scenes", "--scene-id", "scene")
    assert result.exit_code == 2
    assert "--db-host and --db-port require --local" in result.output
    remote.assert_not_called()


@pytest.mark.parametrize("port", ["0", "65536", "not-a-port"])
def test_invalid_database_port_stops_before_local_side_effects(local_execution, port):
    result = CliRunner().invoke(
        client.cli, ["--local", "--db-port", port, "scenes", "--scene-id", "scene"]
    )
    assert result.exit_code == 2
    client.save_historical_run.assert_not_awaited()
    local_execution[0].assert_not_awaited()
    local_execution[2].assert_not_called()


def test_invalid_database_url_override_does_not_expose_credentials(
    local_execution, monkeypatch
):
    monkeypatch.setenv("DB_URL", "postgresql://reader:secret@host:invalid/db")
    result = CliRunner().invoke(
        client.cli,
        ["--local", "--db-host", "127.0.0.1", "scenes", "--scene-id", "scene"],
    )
    assert result.exit_code == 1
    assert "Invalid local database URL" in result.output
    assert "secret" not in result.output
    client.save_historical_run.assert_not_awaited()
    local_execution[0].assert_not_awaited()
    local_execution[2].assert_not_called()


@pytest.mark.parametrize("flags", [[], ["--db-host", "127.0.0.1", "--db-port", "5434"]])
def test_local_missing_configuration_fails_before_discovery(
    local_execution, monkeypatch, flags
):
    processed, _, queue = local_execution
    monkeypatch.delenv("DB_URL")
    result = CliRunner().invoke(
        client.cli, ["--local", *flags, "scenes", "--scene-id", "scene", "--dry-run"]
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
    assert "batches" not in summary
    trigger.assert_not_awaited()
    queue.assert_not_called()


def test_local_overwrite_skips_history_in_discovery_and_every_batch(
    local_execution, monkeypatch
):
    processed, _, queue = local_execution
    processed.side_effect = AssertionError("overwrite must skip the history lookup")
    scenes = ["completed", *[f"scene-{i}" for i in range(100)]]
    handle_search = historical.handle_search

    def discover(payload):
        if "scene_ids" in payload:
            return handle_search(payload)
        return len(scenes), scenes

    monkeypatch.setattr(historical, "handle_search", discover)
    result = CliRunner().invoke(
        client.cli, ["--local", "--overwrite", "--dry-run=false"]
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["already_processed_scene_count"] == 0
    assert summary["queued_count"] == 101
    calls = sorted(queue.call_args_list, key=lambda call: call.kwargs["batch_number"])
    assert [call.args[0] for call in calls] == [
        scenes[:100],
        scenes[100:],
    ]
    assert all(call.kwargs["dry_run"] is False for call in queue.call_args_list)
    processed.assert_not_awaited()
    assert client.save_historical_run.await_count == 2


def test_local_overwrite_preview_needs_no_database(local_execution, monkeypatch):
    monkeypatch.delenv("DB_URL")
    result = CliRunner().invoke(
        client.cli,
        ["--local", "scenes", "--scene-id", "completed", "--overwrite", "--list-only"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["scene_ids"] == ["completed"]
    local_execution[0].assert_not_awaited()
    local_execution[2].assert_not_called()
    client.save_historical_run.assert_not_awaited()


def test_remote_overwrite_sends_boolean(remote):
    result = invoke("scenes", "--scene-id", "completed", "--overwrite")
    assert result.exit_code == 0, result.output
    assert remote.call_args.kwargs["json"]["overwrite"] is True


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


@pytest.mark.parametrize("selector", [[], ["eodag"]])
@pytest.mark.parametrize("with_geometry", [False, True])
def test_eodag_is_default_and_explicit_selector_is_preserved(
    remote, tmp_path, selector, with_geometry
):
    path = tmp_path / "area.geojson"
    geometry = {"type": "FeatureCollection", "features": []}
    path.write_text(json.dumps(geometry))
    result = invoke(
        *selector,
        "--date-start",
        "2025-01-01",
        "--date-end",
        "2025-01-02",
        *(["--geometry", str(path)] if with_geometry else []),
    )
    assert result.exit_code == 0, result.output
    assert remote.call_args.kwargs["json"] == {
        "start": "2025-01-01",
        "end": "2025-01-02",
        **({"geometry": geometry} if with_geometry else {}),
        "action": "enqueue",
        "manual_trigger": False,
        "dry_run": True,
    }


@pytest.mark.parametrize(
    "flags,dry_run", [(["--dry-run"], True), (["--dry-run=false"], False)]
)
def test_default_eodag_accepts_inline_search_options(remote, tmp_path, flags, dry_run):
    path = tmp_path / "region.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    result = invoke(
        f"--geometry={path}",
        "--date-start=2025-01-01",
        "--date-end=2025-02-01",
        "--mode=EW",
        "--mode=IW",
        "--mode=SM",
        *flags,
    )
    assert result.exit_code == 0, result.output
    payload = remote.call_args.kwargs["json"]
    assert (payload["start"], payload["end"]) == ("2025-01-01", "2025-02-01")
    assert payload["modes"] == ["EW", "IW", "SM"]
    assert payload["dry_run"] is dry_run


@pytest.mark.parametrize("option", ["--date-strat", "--unknown"])
def test_default_eodag_rejects_unknown_options_before_submission(
    local_execution, option
):
    result = CliRunner().invoke(client.cli, ["--local", option, "2025-01-01"])
    assert result.exit_code == 2
    assert "No such option" in result.output
    assert option in result.output
    client.save_historical_run.assert_not_awaited()
    local_execution[0].assert_not_awaited()
    local_execution[2].assert_not_called()


@pytest.mark.parametrize(
    "modes,expected",
    [
        (["IW"], ["IW"]),
        (["EW", "IW", "SM"], ["EW", "IW", "SM"]),
        (["iw", "ew", "iw"], ["IW", "EW"]),
    ],
)
def test_eodag_mode_flags_are_normalized_and_sent_as_a_list(
    remote, tmp_path, modes, expected
):
    path = tmp_path / "area.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    flags = [flag for mode in modes for flag in ("--mode", mode)]
    result = invoke("eodag", "--geometry", str(path), *flags)
    assert result.exit_code == 0, result.output
    assert remote.call_args.kwargs["json"]["modes"] == expected


def test_invalid_mode_stops_before_local_side_effects(local_execution, tmp_path):
    path = tmp_path / "area.geojson"
    path.write_text("{}")
    result = CliRunner().invoke(
        client.cli, ["--local", "eodag", "--geometry", str(path), "--mode", "WV"]
    )
    assert result.exit_code == 2
    assert "Invalid value for '--mode'" in result.output
    client.save_historical_run.assert_not_awaited()
    local_execution[0].assert_not_awaited()
    local_execution[2].assert_not_called()


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
        ["eodag", "--date-end", "invalid-date"],
        ["--date-start", "invalid-date"],
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
    assert (source / "geodes_catalog.py").is_file()
    assert (source / "queues.py").is_file()
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
