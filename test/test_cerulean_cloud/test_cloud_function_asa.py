import importlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cerulean_cloud.cloud_function_asa import queuer


@pytest.fixture
def asa_main(monkeypatch):
    # Cloud Functions loads entry-point modules from the service directory.
    service_root = Path(queuer.__file__).parent
    monkeypatch.syspath_prepend(str(service_root))
    # These tests never decode the DARK analyzer's vector tiles.
    if importlib.util.find_spec("mapbox_vector_tile") is None:
        monkeypatch.setitem(sys.modules, "mapbox_vector_tile", Mock())
    return importlib.import_module("cerulean_cloud.cloud_function_asa.main")


def test_published_through(asa_main):
    assert asa_main.published_through(datetime(2026, 1, 2, 1, 25)) == datetime(
        2025, 12, 30
    )


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

        def upload_from_string(self, data):
            uploads[self.name] = pq.read_table(io.BytesIO(data))

    bucket = SimpleNamespace(name="bucket", blob=Blob)
    written = list(asa_main.write_ais_slices(iter(batches), bucket, "ais/2026-01-02"))
    assert written == [
        (scene, f"gs://bucket/ais/2026-01-02/{scene}.parquet") for scene in "abc"
    ]
    tables = [uploads[f"ais/2026-01-02/{scene}.parquet"] for scene in "abc"]
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
