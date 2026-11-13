"""Execute the notebook's selection and submission cells without cloud access."""

import ast
import datetime
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from cerulean_cloud.cloud_function_historical_run import main as historical


def test_notebook_date_range_uses_eodag(notebook_cells, tmp_path):
    source = next(
        source
        for source in notebook_cells
        if source.startswith("# FROM EODAG DATE RANGE")
    )
    geometry = {"type": "FeatureCollection", "features": []}
    path = tmp_path / "test/test_cerulean_cloud/fixtures/whole_world.geojson"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(geometry))
    search = MagicMock(return_value=(3, ["ocean"]))
    frame = MagicMock()
    namespace = {
        "datetime": datetime,
        "json": json,
        "repo_root": tmp_path,
        "handle_search": search,
        "pd": frame,
    }
    exec(source, namespace)
    search.assert_called_once_with(
        {
            "source": "eodag",
            "start": "2025-01-01",
            "end": "2025-01-02",
            "geometry": geometry,
        }
    )
    assert namespace["scenes"] == ["ocean"]
    frame.DataFrame.assert_called_once_with({"scene_id": ["ocean"]})


@pytest.mark.parametrize("column", ["scene_id", "s1_scene_id"])
def test_notebook_common_options_exclude_csv_and_deduplicate(
    notebook_cells, read_csv, tmp_path, column
):
    path = tmp_path / "processed.csv"
    path.write_text(f"{column}\nalready-done\n")
    source = next(
        source
        for source in notebook_cells
        if source.startswith("# COMMON SELECTION OPTIONS")
    )
    source = source.replace("PROCESSED_CSV = None", f"PROCESSED_CSV = {str(path)!r}")
    namespace = {
        "scenes": ["first", "already-done", "first", "second"],
        "read_scene_ids_csv": read_csv,
    }
    exec(source, namespace)
    assert namespace["scenes"] == ["first", "second"]


@pytest.fixture
def notebook_cells():
    path = Path(__file__).resolve().parents[2] / "notebooks/TriggerOnDemand.ipynb"
    return ["".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]]


@pytest.fixture
def notebook_target():
    return {
        "os": SimpleNamespace(environ={}),
        "GCP_PROJECT": "project",
        "GCP_REGION": "region",
        "pulumi_stack": "test",
        "resolve_orchestrator_queue": MagicMock(return_value="deployed-queue"),
    }


@pytest.fixture
def read_csv(notebook_cells):
    setup = next(
        source for source in notebook_cells if "def read_scene_ids_csv" in source
    )
    function = next(
        node
        for node in ast.parse(setup).body
        if isinstance(node, ast.FunctionDef) and node.name == "read_scene_ids_csv"
    )
    namespace = {"pd": pd}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "notebook", "exec"),
        namespace,
    )
    return namespace["read_scene_ids_csv"]


@pytest.mark.parametrize("column", ["scene_id", "s1_scene_id"])
def test_notebook_accepts_current_and_legacy_csvs(
    notebook_cells, read_csv, tmp_path, column
):
    path = tmp_path / "scenes.csv"
    path.write_text(f"{column}\n first \nsecond\n")
    source = next(
        source for source in notebook_cells if source.startswith("# FROM CSV")
    )
    source = source.replace(
        'scenes_csv = "scene_ids.csv"', f"scenes_csv = {str(path)!r}"
    )
    namespace = {"read_scene_ids_csv": read_csv}
    exec(source, namespace)
    assert namespace["scenes"] == ["first", "second"]


def test_notebook_reports_missing_csv_id_column(read_csv, tmp_path):
    path = tmp_path / "scenes.csv"
    path.write_text("unrecognized\nscene\n")
    with pytest.raises(ValueError, match="scene_id or s1_scene_id"):
        read_csv(path)


@pytest.mark.parametrize(
    "stack,pulumi_stack",
    [("TEST", "test"), ("STAGING", "staging"), ("PRODUCTION", "prod20240903")],
)
@pytest.mark.parametrize("override_db", [False, True])
def test_notebook_configures_local_runner_for_selected_stack(
    notebook_cells, monkeypatch, stack, pulumi_stack, override_db
):
    import dotenv

    config = dict.fromkeys(
        [
            "NAME",
            "TITILER_URL",
            "TITILER_API_KEY",
            "API_KEY",
            "INFERENCE_URL",
            "DB_URL",
            "ASA_QUEUE",
            "ASA_URL",
            "SR_URL",
            "ORCHESTRATOR_URL",
        ],
        "unused",
    )
    config.update(
        DB_URL="postgresql+asyncpg://localhost/selected-stack",
        ORCHESTRATOR_URL="https://orchestrator.example",
        API_KEY="test-key",
    )
    monkeypatch.setenv("STACK_SECRETS", json.dumps({stack: config}))
    for key in (
        "DB_URL",
        "GCPPROJECT",
        "GCPREGION",
        "QUEUE",
        "ORCHESTRATOR_URL",
        "API_KEY",
    ):
        monkeypatch.setenv(key, "old-value")
    if override_db:
        monkeypatch.setenv(
            "HISTORICAL_DB_URL", "postgresql+asyncpg://localhost/proxy-db"
        )
    else:
        monkeypatch.delenv("HISTORICAL_DB_URL", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", MagicMock())
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    namespace = {"stack": stack, "GCP_PROJECT": "project", "GCP_REGION": "region"}
    source = next(
        source for source in notebook_cells if "def read_scene_ids_csv" in source
    )
    exec(source, namespace)
    assert namespace["historical_run"] is historical.run_historical
    assert os.environ["DB_URL"] == (
        "postgresql://localhost/proxy-db"
        if override_db
        else "postgresql://localhost/selected-stack"
    )
    assert os.environ["GCPPROJECT"] == "project"
    assert os.environ["GCPREGION"] == "region"
    assert (
        os.environ["QUEUE"] == "old-value"
    )  # Resolved at submission time, not selection.
    assert os.environ["ORCHESTRATOR_URL"] == "https://orchestrator.example"
    assert os.environ["API_KEY"] == "test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [False, True])
async def test_notebook_batches_selected_scenes_through_local_historical_run(
    notebook_cells, notebook_target, dry_run
):
    source = next(source for source in notebook_cells if source.startswith("DRY_RUN ="))
    source = source.replace("DRY_RUN = False", f"DRY_RUN = {dry_run!r}")
    scenes = [f"scene-{i}" for i in range(205)]
    submission = AsyncMock(side_effect=[{"trigger_id": i} for i in range(3)])
    namespace = {**notebook_target, "scenes": scenes, "historical_run": submission}
    await eval(
        compile(source, "notebook", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),
        namespace,
    )
    payloads = [call.args[0] for call in submission.await_args_list]
    assert [len(payload["scene_ids"]) for payload in payloads] == [100, 100, 5]
    assert [scene for payload in payloads for scene in payload["scene_ids"]] == scenes
    assert all(payload["manual_trigger"] is True for payload in payloads)
    for payload in payloads:
        assert payload["dry_run"] is dry_run
    assert namespace["batch_results"] == [{"trigger_id": i} for i in range(3)]
    notebook_target["resolve_orchestrator_queue"].assert_called_once_with(
        "project", "region", "test"
    )
    assert notebook_target["os"].environ["QUEUE"] == "deployed-queue"


@pytest.mark.asyncio
async def test_notebook_stops_on_submission_failure_and_retains_completed_results(
    notebook_cells,
    notebook_target,
):
    source = next(source for source in notebook_cells if source.startswith("DRY_RUN ="))
    submission = AsyncMock(
        side_effect=[{"trigger_id": 1}, RuntimeError("queue failed")]
    )
    namespace = {
        **notebook_target,
        "scenes": [f"scene-{i}" for i in range(205)],
        "historical_run": submission,
    }
    with pytest.raises(RuntimeError, match="queue failed"):
        await eval(
            compile(source, "notebook", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),
            namespace,
        )
    assert submission.await_count == 2
    assert namespace["batch_results"] == [{"trigger_id": 1}]


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [None, "false", 0])
async def test_notebook_requires_explicit_boolean_dry_run(notebook_cells, dry_run):
    source = next(source for source in notebook_cells if source.startswith("DRY_RUN ="))
    source = source.replace("DRY_RUN = False", f"DRY_RUN = {dry_run!r}")
    submission = AsyncMock()
    with pytest.raises(ValueError, match="DRY_RUN must be True or False"):
        await eval(
            compile(source, "notebook", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),
            {"scenes": ["scene"], "historical_run": submission},
        )
    submission.assert_not_awaited()


def test_notebook_no_longer_posts_to_scene_relevancy(notebook_cells):
    # Both old orchestration routes now use the historical-run submission cell.
    assert not any(
        '"Records":' in source or '"Sns":' in source for source in notebook_cells
    )
