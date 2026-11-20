import asyncio
import contextlib
import importlib
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cerulean_cloud.cloud_function_asa import queuer

PG_CTL = Path(__file__).resolve().parents[2] / ".conda" / "ceru-ci" / "bin" / "pg_ctl"
# A batch at 15:00 UTC on 20 Nov 2026 takes captures before 17 Nov 18:00.
NOW = datetime(2026, 11, 20, 15, 0)


@pytest.fixture
def asa_main(monkeypatch):
    # Cloud Functions loads entry-point modules from the service directory.
    service_root = Path(queuer.__file__).parent
    monkeypatch.syspath_prepend(str(service_root))
    # These tests never decode the DARK analyzer's vector tiles.
    if importlib.util.find_spec("mapbox_vector_tile") is None:
        monkeypatch.setitem(sys.modules, "mapbox_vector_tile", Mock())
    return importlib.import_module("cerulean_cloud.cloud_function_asa.main")


@pytest.mark.parametrize(
    "now, through",
    [
        # Before 12:00 UTC the newest release is yesterday's, through 29 Dec.
        (datetime(2026, 1, 2, 1, 25), datetime(2025, 12, 30)),
        (datetime(2026, 1, 2, 11, 59), datetime(2025, 12, 30)),
        # From 12:00 UTC today's release adds 30 Dec.
        (datetime(2026, 1, 2, 12, 0), datetime(2025, 12, 31)),
        (datetime(2026, 1, 2, 15, 0), datetime(2025, 12, 31)),
    ],
)
def test_published_through(asa_main, now, through):
    assert asa_main.published_through(now) == through


def test_pending_run_flags(asa_main):
    run = SimpleNamespace(
        asa_vessel_time=None,
        asa_infra_time=datetime(2026, 1, 2),
        asa_dark_time=None,
        asa_natural_time=None,  # NATURAL is not tracked yet
    )
    assert asa_main.pending_run_flags(run) == ["VESSEL", "DARK"]


def test_completed_run_flags(asa_main):
    analyzers = {
        "VESSEL": SimpleNamespace(data_is_available=False),
        "INFRA": SimpleNamespace(),
        "DARK": SimpleNamespace(data_is_available=True),
    }
    assert asa_main.completed_run_flags(
        ["VESSEL", "INFRA", "DARK", "NATURAL"], analyzers
    ) == ["INFRA", "DARK", "NATURAL"]


def test_dark_availability_is_a_python_bool(asa_main, monkeypatch):
    # pandas hands back numpy.bool_, which `is False` checks would miss.
    result = pd.DataFrame({"data_available": pd.Series([False], dtype="boolean")})
    monkeypatch.setattr("pandas_gbq.read_gbq", lambda *args, **kwargs: result)
    scene = SimpleNamespace(
        scene_id="s", start_time=datetime(2026, 1, 1), end_time=datetime(2026, 1, 1)
    )
    analyzer = SimpleNamespace(s1_scene=scene, gfw_project_id="p", credentials=None)
    assert asa_main.ASA_MAPPING["DARK"].check_data_availability(analyzer) is False


@pytest.mark.parametrize(
    "uri", ["gs://asa/ais/2026-01-02/other.parquet", "gs://elsewhere/ais/s.parquet"]
)
def test_ais_slice_must_be_this_scenes(asa_main, monkeypatch, uri):
    monkeypatch.setenv("ASA_BUCKET", "asa")
    analyzer = SimpleNamespace(ais_uri=uri, s1_scene=SimpleNamespace(scene_id="s"))
    with pytest.raises(ValueError):
        asa_main.ASA_MAPPING["VESSEL"].retrieve_ais_data(analyzer)


@pytest.mark.parametrize(
    "token, body",
    [
        ("api-key", {"batch": True}),  # the API key can't start a batch
        ("api-key", [1]),  # a body that isn't an object
    ],
)
def test_main_rejects_mismatched_requests(asa_main, monkeypatch, token, body):
    monkeypatch.setenv("API_KEY", "api-key")
    request = SimpleNamespace(
        headers={"Authorization": f"Bearer {token}"},
        get_json=lambda silent=False: body,
    )
    with pytest.raises(Exception) as raised:
        asa_main.main(request)
    assert raised.value.code == 400


def test_main_checks_the_scheduler_token_before_reading_the_body(asa_main, monkeypatch):
    monkeypatch.setenv("API_KEY", "api-key")
    get_json = Mock()
    request = SimpleNamespace(headers={}, get_json=get_json)
    with pytest.raises(Exception) as raised:
        asa_main.main(request)
    assert raised.value.code == 401
    assert not get_json.called


def test_ais_time_filter_merges_overlaps_and_keeps_gaps(asa_main):
    windows = [
        (datetime(2025, 5, 1, 10), datetime(2025, 5, 2, 0)),
        (datetime(2023, 5, 1, 0), datetime(2023, 5, 1, 14)),
        (datetime(2025, 5, 1, 0), datetime(2025, 5, 1, 14)),
    ]
    sql, params = asa_main.ais_time_filter(windows)
    assert sql == (
        "(seg.timestamp BETWEEN @t0_start AND @t0_end"
        " OR seg.timestamp BETWEEN @t1_start AND @t1_end)"
    )
    assert [(p.name, p.value) for p in params] == [
        ("t0_start", datetime(2023, 5, 1, 0)),
        ("t0_end", datetime(2023, 5, 1, 14)),
        ("t1_start", datetime(2025, 5, 1, 0)),
        ("t1_end", datetime(2025, 5, 2, 0)),
    ]


@pytest.mark.parametrize(
    "claims, status",
    [
        (None, 401),
        (ValueError("bad token"), 401),
        ({"email": "someone-else@example.com"}, 403),
        ({"email": "scheduler@example.com"}, None),
    ],
)
def test_verify_scheduler_token(asa_main, monkeypatch, claims, status):
    monkeypatch.setenv("ASA_SCHEDULER_SA", "scheduler@example.com")
    if isinstance(claims, Exception):
        verify = Mock(side_effect=claims)
    else:
        verify = Mock(return_value=claims)
    monkeypatch.setattr(asa_main.id_token, "verify_oauth2_token", verify)
    headers = {} if claims is None else {"Authorization": "Bearer token"}
    request = SimpleNamespace(headers=headers)
    if status is None:
        asa_main.verify_scheduler_token(request)
    else:
        with pytest.raises(Exception) as raised:
            asa_main.verify_scheduler_token(request)
        assert raised.value.code == status
    # A missing token is rejected without fetching Google's certificates.
    assert verify.called == (claims is not None)


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


@pytest.mark.parametrize("ais_uri", [None, "gs://asa/ais/2026-01-02/scene.parquet"])
def test_queue_ais_uri(monkeypatch, ais_uri):
    client = Mock()
    monkeypatch.setattr(queuer.tasks_v2, "CloudTasksClient", Mock(return_value=client))
    queuer._tasks_client.cache_clear()
    for name in ("PROJECT_ID", "GCPREGION", "ASA_QUEUE", "FUNCTION_URL"):
        monkeypatch.setenv(name, name)
    queuer.add_to_asa_queue("scene", run_flags=["VESSEL"], ais_uri=ais_uri)
    request = client.create_task.call_args.kwargs["request"]
    payload = json.loads(request["task"]["http_request"]["body"])
    if ais_uri is None:
        assert "ais_uri" not in payload
    else:
        assert payload["ais_uri"] == ais_uri


def test_deduped_task_returns_none(monkeypatch):
    client = Mock()
    client.create_task.side_effect = queuer.AlreadyExists("exists")
    monkeypatch.setattr(queuer.tasks_v2, "CloudTasksClient", Mock(return_value=client))
    queuer._tasks_client.cache_clear()
    for name in ("PROJECT_ID", "GCPREGION", "ASA_QUEUE", "FUNCTION_URL"):
        monkeypatch.setenv(name, name)
    assert queuer.add_to_asa_queue("scene", run_flags=["INFRA"]) is None


@pytest.mark.parametrize("created", [True, False])
def test_queue_scene_stamps_only_created_tasks(asa_main, monkeypatch, created):
    class Session:
        executed = []

        @contextlib.asynccontextmanager
        async def begin(self):
            yield

        async def execute(self, statement):
            self.executed.append(statement)

    session = Session()
    queue = Mock(return_value=Mock() if created else None)
    monkeypatch.setattr(asa_main, "add_to_asa_queue", queue)
    asyncio.run(asa_main.queue_scene(session, "s", {"VESSEL"}, {7, 3}, "gs://a/s"))
    assert queue.call_args.kwargs["task_suffix"] == [3, 7]
    assert len(session.executed) == int(created)
    if created:
        assert "asa_batch_queued_at" in str(session.executed[0])


def test_write_ais_slices_skips_a_failed_upload(asa_main):
    schema = pa.schema([("scene_id", pa.string()), ("timestamp", pa.timestamp("us"))])
    t = datetime(2026, 1, 1)
    batches = [pa.record_batch([["a", "b", "c"], [t, t, t]], schema=schema)]

    class Blob:
        def __init__(self, name):
            self.name = name

        def upload_from_string(self, data, *, if_generation_match):
            if self.name.endswith("/b.parquet"):
                raise OSError("upload failed")
            if self.name.endswith("/c.parquet"):
                raise asa_main.PreconditionFailed("another batch wrote this scene")

    bucket = SimpleNamespace(name="bucket", blob=Blob)
    names = {s: f"ais/{s}.parquet" for s in "abc"}
    assert list(asa_main.write_ais_slices(iter(batches), bucket, names)) == [
        ("a", "gs://bucket/ais/a.parquet"),
        ("b", None),
        ("c", "gs://bucket/ais/c.parquet"),
    ]


def test_cache_listing_failure_still_queues_dark_only_work(asa_main, monkeypatch):
    scene = SimpleNamespace(scene_id="s", start_time=NOW)
    vessel = SimpleNamespace(
        id=1, asa_vessel_time=None, asa_infra_time=NOW, asa_dark_time=NOW
    )
    dark = SimpleNamespace(
        id=2, asa_vessel_time=NOW, asa_infra_time=NOW, asa_dark_time=None
    )
    session = Mock()
    session.begin = contextlib.nullcontext
    session.execute = AsyncMock(
        return_value=SimpleNamespace(all=lambda: [(scene, vessel), (scene, dark)])
    )
    storage = Mock()
    storage.return_value.bucket.return_value.list_blobs.side_effect = (
        asa_main.GoogleAPIError("list failed")
    )
    monkeypatch.setattr(asa_main.storage, "Client", storage)
    monkeypatch.setenv("ASA_BUCKET", "bucket")
    monkeypatch.setenv("PROJECT_ID", "test-project")
    queue, bq = AsyncMock(), Mock()
    monkeypatch.setattr(asa_main, "queue_scene", queue)
    monkeypatch.setattr(asa_main.bigquery, "Client", bq)
    with pytest.raises(asa_main.GoogleAPIError, match="list failed"):
        asyncio.run(asa_main.queue_batch(session, Mock()))
    queue.assert_awaited_once_with(session, "s", {"DARK"}, {2}, ais_uri=None)
    bq.assert_not_called()


@pytest.mark.parametrize("all_cached", [True, False])
def test_queue_batch_reuses_parquets(asa_main, monkeypatch, all_cached):
    # Both runs are eligible again. Scene a has only a legacy run-date slice.
    a = SimpleNamespace(scene_id="a", start_time=datetime(2026, 10, 1, 12))
    b = SimpleNamespace(scene_id="b", start_time=datetime(2026, 10, 9, 12))
    rows = [
        (
            scene,
            SimpleNamespace(
                id=i, asa_vessel_time=None, asa_infra_time=NOW, asa_dark_time=NOW
            ),
        )
        for i, scene in enumerate([a, b], 1)
    ]
    session = Mock()
    session.begin = contextlib.nullcontext
    session.execute = AsyncMock(return_value=SimpleNamespace(all=lambda: rows))
    queue = AsyncMock()
    monkeypatch.setattr(asa_main, "queue_scene", queue)
    bucket = Mock()
    bucket.name = "test-bucket"
    names = ["ais/2026-11-19/a.parquet", "ais/2026-11-20/a.parquet"]
    if all_cached:
        names += ["ais/2026-10-09/b.parquet", "ais/2026-11-20/b.parquet"]
    bucket.list_blobs.return_value = [
        SimpleNamespace(name=n, time_created=datetime.now(timezone.utc))
        for n in sorted(names)
    ]
    monkeypatch.setattr(
        asa_main.storage, "Client", Mock(return_value=Mock(bucket=lambda _: bucket))
    )
    monkeypatch.setenv("ASA_BUCKET", bucket.name)
    monkeypatch.setenv("PROJECT_ID", "test-project")
    monkeypatch.setattr(asa_main, "MAX_BATCH_AIS_DATES", 1)
    bq = Mock()
    monkeypatch.setattr(asa_main.bigquery, "Client", bq)
    analyzer = Mock(
        return_value=SimpleNamespace(
            ais_envelope=pd.Series([SimpleNamespace(wkt="polygon")]),
            ais_start_time=b.start_time - timedelta(hours=8),
            ais_end_time=b.start_time + timedelta(hours=6),
        )
    )
    monkeypatch.setitem(asa_main.ASA_MAPPING, "VESSEL", analyzer)
    if not all_cached:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "{}")
        monkeypatch.setattr(asa_main.Credentials, "from_service_account_info", Mock())
        monkeypatch.setattr(asa_main.bigquery_storage, "BigQueryReadClient", Mock())
        schema = pa.schema(
            [("scene_id", pa.string()), ("timestamp", pa.timestamp("us"))]
        )
        bq.return_value.list_rows.return_value.to_arrow_iterable.return_value = [
            pa.record_batch([["b"], [None]], schema=schema)
        ]
        bucket.blob.side_effect = lambda name: SimpleNamespace(
            name=name, upload_from_string=Mock()
        )

    assert asyncio.run(asa_main.queue_batch(session, Mock())) == []
    assert [(args[1], kwargs["ais_uri"]) for args, kwargs in queue.await_args_list] == [
        ("a", "gs://test-bucket/ais/2026-11-20/a.parquet"),
        ("b", "gs://test-bucket/ais/2026-10-09/b.parquet"),
    ]
    if all_cached:
        bq.assert_not_called()
        analyzer.assert_not_called()
        bucket.blob.assert_not_called()
    else:
        analyzer.assert_called_once_with(b)
        params = bq.return_value.query.call_args.kwargs["job_config"].query_parameters
        assert [p.struct_values["scene_id"] for p in params[0].values] == ["b"]
        capture = b.start_time.replace(tzinfo=timezone.utc)
        assert [p.value for p in params[1:]] == [
            capture - timedelta(hours=8),
            capture + timedelta(hours=6),
        ]
        bucket.blob.assert_called_once_with("ais/2026-10-09/b.parquet")


@pytest.mark.parametrize(
    "source_day, outcome",
    [
        ("10-01", "copied"),
        ("11-20", "copied"),
        ("10-01", "concurrent"),
        ("11-20", "expired"),
    ],
)
def test_refresh_expiring_cached_slice(asa_main, source_day, outcome):
    now = datetime.now(timezone.utc)
    canonical = "ais/2026-10-01/s.parquet"
    old = SimpleNamespace(
        name=f"ais/2026-{source_day}/s.parquet",
        time_created=now - timedelta(days=6, hours=12),
        generation=123,
    )
    fresh = SimpleNamespace(name=canonical, time_created=now, generation=456)
    bucket = Mock()
    bucket.name = "bucket"
    bucket.list_blobs.return_value = [old]
    if outcome == "copied":
        bucket.copy_blob.return_value = fresh
    elif outcome == "concurrent":
        bucket.copy_blob.side_effect = asa_main.PreconditionFailed("another writer")
        bucket.get_blob.return_value = fresh
    else:
        bucket.copy_blob.side_effect = asa_main.NotFound("source expired")
        bucket.get_blob.return_value = None

    uris = asa_main.cached_ais_slices(bucket, {"s": canonical})
    assert uris == ({} if outcome == "expired" else {"s": f"gs://bucket/{canonical}"})
    bucket.copy_blob.assert_called_once_with(
        old,
        bucket,
        canonical,
        if_source_generation_match=123,
        if_generation_match=123 if old.name == canonical else 0,
    )


@pytest.mark.parametrize(
    "body, options",
    [
        ({}, (32, None, False)),
        ({"lookback_days": 1500}, (1500, None, False)),
        (
            {"lookback_days": 1500, "pivot": "2024-06-01", "retry_incomplete": True},
            (1500, datetime(2024, 6, 1), True),
        ),
        # The last day with captures before the cutoff.
        (
            {"lookback_days": 1500, "pivot": "2026-11-17"},
            (1500, datetime(2026, 11, 17), False),
        ),
    ],
)
def test_batch_options(asa_main, body, options):
    assert asa_main.batch_options(body, NOW) == options


@pytest.mark.parametrize(
    "body",
    [
        {"lookback_days": "1500"},
        {"lookback_days": 0},
        {"lookback_days": True},
        {"lookback_days": 1500, "retry_incomplete": "yes"},
        {"pivot": "2024-06-01"},  # daily runs have no pivot
        {"retry_incomplete": True},  # daily runs already retry
        {"lookback_days": 1500, "pivot": "2026-11-18"},  # after the cutoff
        {"lookback_days": 40, "pivot": "2024-06-01"},  # before the lookback
        {"lookback_days": 1500, "pivot": "June 1"},
        {"lookback_days": 1500, "pivot": 20240601},
    ],
)
def test_batch_options_rejects(asa_main, body):
    with pytest.raises(Exception) as raised:
        asa_main.batch_options(body, NOW)
    assert raised.value.code == 400


@pytest.mark.skipif(
    not PG_CTL.exists(), reason="canonical ceru-ci PostgreSQL pg_ctl is unavailable"
)
def test_batch_query(asa_main, postgresql):
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
        "never_queued": (datetime(2026, 11, 10), None, None, None, True),
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
        "unpublished": (datetime(2026, 11, 18), None, None, None, True),
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
        ]
    finally:
        engine.dispose()
