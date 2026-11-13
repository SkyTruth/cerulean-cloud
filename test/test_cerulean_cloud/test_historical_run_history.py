"""Historical submission provenance and PostgreSQL history records."""

import subprocess
import sys
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from shapely.geometry import box, mapping

from scripts import historical_run as client
from cerulean_cloud import database_schema as db


def region(*boxes):
    features = [{"type": "Feature", "geometry": mapping(b)} for b in boxes]
    return {"type": "FeatureCollection", "features": features}


@pytest.mark.parametrize("key", [["--api-key=secret"], ["--api-key", "secret"]])
def test_metadata_normalizes_geometry_and_redacts_api_key(monkeypatch, key):
    monkeypatch.setattr(sys, "argv", ["historical_run.py", *key, "eodag"])
    git = MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout=b"v1\n"))
    monkeypatch.setattr(client.subprocess, "run", git)
    metadata = client.history_metadata(
        {
            "start": "2025-01-01",
            "end": "2025-01-02",
            "geometry": region(box(1, 1, 2, 2)),
        }
    )
    assert "secret" not in metadata["cli_command"]
    assert "[REDACTED]" in metadata["cli_command"]
    assert (metadata["date_start"], metadata["date_end"]) == (
        date(2025, 1, 1),
        date(2025, 1, 2),
    )
    assert metadata["geometry"].startswith("SRID=4326;MULTIPOLYGON")
    assert metadata["git_hash"] == metadata["git_tag"] == "v1"


def test_scene_ids_supply_acquisition_dates():
    scenes = [
        "S1A_IW_GRDH_1SDV_20250103T235900_20250104T000100_0_0_0",
        "S1A_IW_GRDH_1SDV_20250101T000000_20250101T000100_0_0_0",
    ]
    metadata = client.history_metadata({"scene_ids": scenes})
    assert (metadata["date_start"], metadata["date_end"]) == (
        date(2025, 1, 1),
        date(2025, 1, 4),
    )
    assert metadata["geometry"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"scene_ids": ["invalid"]},
        {"start": "2025-01-01", "end": "2025-01-02", "geometry": region()},
    ],
)
def test_unreadable_selection_is_rejected(payload):
    with pytest.raises(ValueError):
        client.history_metadata(payload)


@pytest.mark.asyncio
async def test_history_row_is_inserted_then_updated_with_its_trigger(
    postgresql, monkeypatch
):
    url = (
        f"postgresql://{postgresql.info.user}@{postgresql.info.host}:"
        f"{postgresql.info.port}/{postgresql.info.dbname}"
    )
    monkeypatch.setenv("DB_URL", url)
    engine = sa.create_engine(url)
    record = db.HistoricalRun(
        cli_command="historical_run --local",
        date_start=date(2025, 1, 1),
        date_end=date(2025, 1, 2),
        geometry="SRID=4326;MULTIPOLYGON(((1 1,2 1,2 2,1 1)))",
        dry_run=True,
        submission_start_time=datetime(2025, 1, 3, tzinfo=timezone.utc),
        trigger_=db.Trigger(trigger_logs="", trigger_type="HISTORICAL_RUN"),
    )
    query = sa.text(
        "SELECT h.submission_end_time, h.queued_task_count, t.scene_count, "
        "ST_SRID(h.geometry::geometry) FROM historical_run h "
        "JOIN trigger t ON t.id = h.trigger"
    )
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS postgis"))
            db.Trigger.__table__.create(conn)
        # Without the table, the insert fails and leaves no trigger behind.
        with pytest.raises(sa.exc.ProgrammingError, match="historical_run"):
            await client.save_historical_run(record)
        db.HistoricalRun.__table__.create(engine)
        record = await client.save_historical_run(record)
        with engine.connect() as conn:
            assert conn.execute(query).one() == (None, None, None, 4326)

        record.submission_end_time = datetime(2025, 1, 4, tzinfo=timezone.utc)
        record.queued_task_count = record.trigger_.scene_count = 3
        await client.save_historical_run(record)
        with engine.connect() as conn:
            assert conn.execute(query).one() == (
                record.submission_end_time,
                3,
                3,
                4326,
            )
            assert conn.execute(sa.text("SELECT count(*) FROM trigger")).scalar() == 1
    finally:
        engine.dispose()
