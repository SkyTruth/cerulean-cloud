import asyncio
import importlib
import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from werkzeug.exceptions import Forbidden

from cerulean_cloud.cloud_function_asa import queuer

PG_CTL = Path(__file__).resolve().parents[2] / ".conda" / "ceru-ci" / "bin" / "pg_ctl"
# The 05:00 UTC batch on 20 Nov 2026 takes captures before 17 Nov 18:00.
NOW = datetime(2026, 11, 20, 5, 0)


@pytest.fixture
def asa_main(monkeypatch):
    # Cloud Functions loads entry-point modules from the service directory.
    service_root = Path(queuer.__file__).parent
    monkeypatch.syspath_prepend(str(service_root))
    # These tests never decode the DARK analyzer's vector tiles.
    if importlib.util.find_spec("mapbox_vector_tile") is None:
        monkeypatch.setitem(sys.modules, "mapbox_vector_tile", Mock())
    return importlib.import_module("cerulean_cloud.cloud_function_asa.main")


def test_ais_eligibility_advances_at_release(asa_main):
    release = datetime(2026, 1, 2, 3, 30)
    before = release - timedelta(seconds=1)
    assert asa_main.published_through(before) == datetime(2025, 12, 30)
    assert asa_main.published_through(release) == datetime(2025, 12, 31)


@pytest.mark.parametrize(
    "email",
    ["scheduler@example.com", "someone-else@example.com"],
    ids=["scheduler", "other-account"],
)
def test_only_scheduler_account_can_start_a_batch(asa_main, monkeypatch, email):
    monkeypatch.setenv("ASA_SCHEDULER_SA", "scheduler@example.com")
    monkeypatch.setenv("FUNCTION_URL", "https://asa.example")
    verify = Mock(return_value={"email": email})
    monkeypatch.setattr(asa_main.id_token, "verify_oauth2_token", verify)
    batch = AsyncMock(return_value="Success!")
    monkeypatch.setattr(asa_main, "handle_asa_batch", batch)
    request = SimpleNamespace(
        headers={"Authorization": "Bearer token"},
        get_json=lambda **_: {"batch": True},
    )
    if email == "scheduler@example.com":
        assert asa_main.main(request) == "Success!"
        batch.assert_awaited_once_with(32, None, False)
    else:
        with pytest.raises(Forbidden):
            asa_main.main(request)
        batch.assert_not_awaited()
    verify.assert_called_once_with("token", ANY, audience="https://asa.example")


def test_write_ais_slices(asa_main):
    # Scene "b" spans two batches; scene "c" has only its coverage marker row.
    schema = pa.schema(
        [
            ("scene_id", pa.string()),
            ("timestamp", pa.timestamp("us", tz="UTC")),
            ("lon", pa.float64()),
        ]
    )
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    batches = [
        pa.record_batch([["a", "a", "b"], [t, t, t], [1.0, 2.0, 3.0]], schema=schema),
        pa.record_batch([["b", "c"], [t, None], [4.0, None]], schema=schema),
    ]
    uploads = {}

    class Blob:
        def __init__(self, name):
            self.name = name

        def upload_from_string(self, data, *, if_generation_match):
            assert if_generation_match == 0
            uploads[self.name] = pq.read_table(io.BytesIO(data))

    bucket = SimpleNamespace(name="bucket", blob=Blob)
    names = {
        scene: f"ais/2026-01-0{i}/{scene}.parquet" for i, scene in enumerate("abc", 1)
    }
    written = list(asa_main.write_ais_slices(iter(batches), bucket, names))
    assert written == [(scene, f"gs://bucket/{names[scene]}") for scene in "abc"]
    tables = [uploads[names[scene]] for scene in "abc"]
    assert [table["lon"].to_pylist() for table in tables] == [
        [1.0, 2.0],
        [3.0, 4.0],
        [],
    ]
    assert all(table.column_names == ["timestamp", "lon"] for table in tables)


def test_force_retry_recreates_a_purged_task_on_the_same_day(monkeypatch):
    # Cloud Tasks remembers a deleted name. Simulate the name reservation after purge.
    reserved_names = set()

    def create_task(request):
        name = request["task"]["name"]
        if name in reserved_names:
            raise queuer.AlreadyExists("reserved task name")
        reserved_names.add(name)
        return SimpleNamespace(name=name)

    client = Mock()
    client.create_task.side_effect = create_task
    client.task_path.side_effect = lambda project, location, queue, task_id: task_id
    monkeypatch.setattr(queuer, "_tasks_client", lambda: client)
    monkeypatch.setattr(
        queuer, "_schedule_time", lambda **kwargs: NOW.replace(tzinfo=timezone.utc)
    )
    for name in ("PROJECT_ID", "GCPREGION", "ASA_QUEUE", "FUNCTION_URL"):
        monkeypatch.setenv(name, name)

    def enqueue(force_retry=False):
        return queuer.add_to_asa_queue(
            "scene", ["VESSEL"], task_suffix=[7], force_retry=force_retry
        )

    assert enqueue() is not None
    assert enqueue() is None
    assert enqueue(True) is not None
    assert enqueue(True) is not None
    assert len(reserved_names) == 3


def test_overlapping_asa_batches(asa_main, postgresql, monkeypatch):
    from sqlalchemy import select
    from werkzeug.exceptions import ServiceUnavailable

    info = postgresql.info
    monkeypatch.setenv(
        "DB_URL",
        f"postgresql+asyncpg://{info.user}@{info.host}:{info.port}/{info.dbname}",
    )
    monkeypatch.delenv("ASA_IS_DRY_RUN", raising=False)
    backfill_days = 2557

    async def hold_batch(session, query, **kwargs):
        # Finish a transaction as queue_batch does between selection and enqueues.
        async with session.begin():
            await session.execute(select(1))
        # A second producer must exit before reaching queue_batch.
        monkeypatch.setattr(asa_main, "queue_batch", AsyncMock(return_value=[]))
        assert "Skipped" in await asa_main.handle_asa_batch(backfill_days)
        with pytest.raises(ServiceUnavailable):
            await asa_main.handle_asa_batch()
        asa_main.queue_batch.assert_not_awaited()
        raise RuntimeError("interrupted batch")

    async def run():
        monkeypatch.setattr(asa_main, "queue_batch", hold_batch)
        with pytest.raises(RuntimeError, match="interrupted batch"):
            await asa_main.handle_asa_batch(backfill_days)
        # Disposing the failed invocation's connection releases the lock.
        assert await asa_main.handle_asa_batch(backfill_days) == "Success!"
        asa_main.queue_batch.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.skipif(
    not PG_CTL.exists(), reason="canonical ceru-ci PostgreSQL pg_ctl is unavailable"
)
def test_batch_query(asa_main, postgresql, monkeypatch):
    import sqlalchemy as sa
    from geoalchemy2.shape import from_shape
    from shapely.geometry import MultiPolygon, box

    import cerulean_cloud.database_schema as db

    engine = sa.create_engine(
        f"postgresql://{postgresql.info.user}:@"
        f"{postgresql.info.host}:{postgresql.info.port}/{postgresql.info.dbname}"
    )
    done = datetime(2026, 11, 19)
    # scene_id: (capture time, asa_vessel_time, asa_dark_time, queued at, active slick)
    scenes = {
        "never_queued": (datetime(2026, 11, 17, 17, 59, 59), None, None, None, True),
        "retry": (datetime(2026, 11, 1), None, None, NOW - timedelta(days=1), True),
        "in_flight": (
            datetime(2026, 11, 2),
            None,
            None,
            NOW - timedelta(minutes=10),
            True,
        ),
        "dark_only": (datetime(2026, 11, 5), done, None, None, True),
        "complete": (datetime(2026, 11, 6), done, done, None, True),
        "unpublished": (datetime(2026, 11, 17, 18), None, None, None, True),
        "no_slicks": (datetime(2026, 11, 7), None, None, None, False),
        "archive": (datetime(2023, 3, 1), None, None, None, True),
    }
    square = box(0, 0, 1, 1)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE EXTENSION postgis"))
        db.Base.metadata.create_all(conn)
        # Skip the foreign keys to triggers, models and classes these rows don't need.
        conn.execute(sa.text("SET LOCAL session_replication_role = replica"))
        for i, (scene_id, (start, vessel, dark, queued, active)) in enumerate(
            scenes.items(), 1
        ):
            conn.execute(
                sa.insert(db.Sentinel1Grd).values(
                    id=i,
                    scene_id=scene_id,
                    start_time=start,
                    end_time=start + timedelta(seconds=25),
                    url="",
                    geometry=from_shape(square, srid=4326),
                )
            )
            conn.execute(
                sa.insert(db.OrchestratorRun).values(
                    id=i,
                    inference_start_time=start,
                    inference_end_time=start,
                    inference_run_logs="",
                    geometry=from_shape(square, srid=4326),
                    trigger=1,
                    model=1,
                    sentinel1_grd=i,
                    asa_vessel_time=vessel,
                    asa_infra_time=done,
                    asa_dark_time=dark,
                    asa_batch_queued_at=queued,
                )
            )
            conn.execute(
                sa.insert(db.Slick).values(
                    id=i,
                    slick_timestamp=start,
                    geometry=from_shape(MultiPolygon([square]), srid=4326),
                    active=active,
                    orchestrator_run=i,
                    inference_idx=1,
                    cls=1,
                    machine_confidence=0.9,
                )
            )

    def selected(**options):
        with sa.orm.Session(engine) as session:
            query = asa_main.batch_query(NOW, **options)
            return [scene.scene_id for scene, _ in session.execute(query)]

    try:
        # VESSEL first, never-queued before retries; a run queued 10 min ago waits.
        assert selected(lookback_days=32) == ["never_queued", "retry", "dark_only"]
        assert selected(lookback_days=32, retry_incomplete=True) == [
            "never_queued",
            "retry",
            "in_flight",
            "dark_only",
        ]
        # One-off runs go on from the pivot and wrap around, skipping queued runs
        # unless retry_incomplete is set.
        pivot = datetime(2026, 11, 3)
        assert selected(lookback_days=1500, pivot=pivot) == [
            "dark_only",
            "never_queued",
            "archive",
        ]
        assert selected(lookback_days=1500, pivot=pivot, retry_incomplete=True) == [
            "dark_only",
            "never_queued",
            "archive",
            "retry",
            "in_flight",
        ]

        # Queue one run through the normal batch path. The next tick advances
        # even though the worker has not written its completion timestamps yet.
        monkeypatch.setattr(asa_main, "MAX_BATCH_RUNS", 1)
        enqueue = Mock()
        monkeypatch.setattr(asa_main, "add_to_asa_queue", enqueue)

        async def queue_first():
            url = engine.url.set(drivername="postgresql+asyncpg")
            async_engine = asa_main.get_engine(url)
            try:
                async with asa_main.DatabaseClient(async_engine) as client:
                    query = asa_main.batch_query(NOW, 1500, pivot)
                    assert await asa_main.queue_batch(client.session, query) == []
            finally:
                await async_engine.dispose()

        asyncio.run(queue_first())
        assert enqueue.call_args.args == ("dark_only",)
        assert selected(lookback_days=1500, pivot=pivot) == ["never_queued"]
    finally:
        engine.dispose()
