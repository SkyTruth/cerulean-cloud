"""Historical-run client request contracts and local archive inventory."""

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from click.testing import CliRunner

from scripts import historical_run as client


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
        "timeout": 600,
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


def test_s3_queue_passes_filters_and_explicit_dry_run(remote):
    result = invoke(
        "s3",
        "--date-start",
        "2025-01-01",
        "--date-end",
        "2025-01-02",
        "--mode",
        "IW",
        "--mode",
        "EW",
        "--polarization",
        "DV",
        "--dry-run",
    )
    assert result.exit_code == 0, result.output
    assert remote.call_args.kwargs["json"] == {
        "source": "s3",
        "start": "2025-01-01",
        "end": "2025-01-02",
        "modes": ["IW", "EW"],
        "polarizations": ["DV"],
        "dry_run": True,
        "action": "enqueue",
        "manual_trigger": False,
    }


def test_large_s3_inventory_can_run_locally_without_api_key_or_pulumi(
    remote, monkeypatch
):
    listing = MagicMock(return_value=["first", "second"])
    monkeypatch.setattr(client, "list_sentinel1_scene_ids", listing)
    pulumi = MagicMock(side_effect=AssertionError("must not invoke Pulumi"))
    monkeypatch.setattr(client.subprocess, "check_output", pulumi)
    monkeypatch.delenv("CERULEAN_API_KEY", raising=False)
    result = CliRunner().invoke(
        client.cli,
        ["s3", "--date-start", "2025-01-01", "--date-end", "2026-01-01", "--list-only"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "scene_count": 2,
        "scene_ids": ["first", "second"],
    }
    listing.assert_called_once_with(
        date(2025, 1, 1), date(2026, 1, 1), modes=None, polarizations=None
    )
    remote.assert_not_called()
    pulumi.assert_not_called()


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


def test_cloud_function_source_imports_from_deployment_root():
    source = (
        Path(__file__).resolve().parents[2]
        / "cerulean_cloud"
        / "cloud_function_historical_run"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import main; assert main.list_sentinel1_scene_ids.__module__ == 'inventory'",
        ],
        cwd=source,
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
