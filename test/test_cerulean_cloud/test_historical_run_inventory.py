"""Archive listing and endpoint behavior without contacting cloud services."""

import json
import sys
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from xml.sax.saxutils import escape

import httpx
import pytest
from flask import Flask, request
from shapely.geometry import box, mapping

from cerulean_cloud.cloud_function_historical_run import main as historical
from cerulean_cloud.cloud_function_historical_run import inventory


@pytest.fixture
def listings(monkeypatch):
    responses = {}

    def handle(request):
        assert request.url.host == "sentinel-s1-l1c.s3.eu-central-1.amazonaws.com"
        key = (
            request.url.params["prefix"],
            request.url.params.get("continuation-token"),
        )
        status, text = responses.pop(key)
        return httpx.Response(status, text=text)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(inventory.httpx, "Client", lambda **kwargs: client)
    yield responses
    assert not responses


def add_listing(listings, prefix, folders, *, next_token=None, token=None):
    entries = "".join(
        f"<CommonPrefixes><Prefix>{escape(folder)}</Prefix></CommonPrefixes>"
        for folder in folders
    )
    continuation = (
        f"<NextContinuationToken>{escape(next_token)}</NextContinuationToken>"
        if next_token
        else ""
    )
    listings[prefix, token] = (
        200,
        (
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<IsTruncated>{str(bool(next_token)).lower()}</IsTruncated>"
            f"{entries}{continuation}</ListBucketResult>"
        ),
    )


def test_archive_all_modes_and_polarizations(listings):
    day = "GRD/2025/1/17/"
    add_listing(listings, day, [day + "IW/", day + "EW/"])
    add_listing(listings, day + "IW/", [day + "IW/DV/", day + "IW/SV/"])
    add_listing(listings, day + "EW/", [day + "EW/DH/"])
    for folder, scene in [("IW/DV/", "dv"), ("IW/SV/", "sv"), ("EW/DH/", "dh")]:
        add_listing(listings, day + folder, [day + folder + scene + "/"])
    assert historical.list_sentinel1_scene_ids(
        date(2025, 1, 17), date(2025, 1, 17)
    ) == ["dh", "dv", "sv"]


def test_archive_filters_and_inclusive_year_boundary(listings):
    for day in ["GRD/2025/12/31/", "GRD/2026/1/1/"]:
        add_listing(listings, day, [day + "EW/", day + "IW/"])
        add_listing(listings, day + "IW/", [day + "IW/DH/", day + "IW/DV/"])
        add_listing(listings, day + "IW/DV/", [day + "IW/DV/scene/"])
    assert historical.list_sentinel1_scene_ids(
        date(2025, 12, 31), date(2026, 1, 1), modes=("IW",), polarizations=("DV",)
    ) == ["scene", "scene"]


def test_archive_pagination_at_every_level(listings):
    day = "GRD/2025/1/17/"
    token = "page+2&more"
    for prefix, child in [(day, day + "IW/"), (day + "IW/", day + "IW/DV/")]:
        add_listing(listings, prefix, [], next_token=token)
        add_listing(listings, prefix, [child], token=token)
    prefix = day + "IW/DV/"
    add_listing(listings, prefix, [prefix + "first/"], next_token=token)
    add_listing(listings, prefix, [prefix + "second/"], token=token)
    assert historical.list_sentinel1_scene_ids(
        date(2025, 1, 17), date(2025, 1, 17)
    ) == ["first", "second"]


def test_failed_archive_listing_raises_instead_of_returning_partial_ids(listings):
    add_listing(listings, "GRD/2025/1/17/", [])
    listings["GRD/2025/1/18/", None] = (503, "Unavailable")
    with pytest.raises(httpx.HTTPStatusError):
        historical.list_sentinel1_scene_ids(date(2025, 1, 17), date(2025, 1, 18))


def test_reversed_archive_dates():
    with pytest.raises(ValueError, match="end_date"):
        historical.list_sentinel1_scene_ids(date(2025, 1, 18), date(2025, 1, 17))


def test_eodag_uses_geodes_and_keeps_spatial_dates_and_ocean_filter(
    monkeypatch,
):
    dag = MagicMock()
    ocean_result = SimpleNamespace(geometry=box(0, 0, 2, 2), properties={"id": "ocean"})
    land_result = SimpleNamespace(
        geometry=box(10, 10, 11, 11), properties={"id": "land"}
    )
    dag.search_iter_page.return_value = [[land_result], [ocean_result]]
    gateway = MagicMock(return_value=dag)
    eodag = SimpleNamespace(EODataAccessGateway=gateway, setup_logging=MagicMock())
    monkeypatch.setitem(sys.modules, "eodag", eodag)
    geometry = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(box(-20, -20, 20, 20)),
                "properties": {},
            }
        ],
    }
    request = {
        "start": "2022-01-01T00:00:00Z",
        "end": "2022-01-02",
        "geometry": geometry,
    }
    historical.validate_request(request)
    assert historical.handle_search(request, ocean_poly=box(-1, -1, 3, 3)) == (
        2,
        ["ocean"],
    )
    dag.set_preferred_provider.assert_called_once_with("geodes")
    assert gateway.call_args.kwargs["user_conf_file_path"].endswith("geodes.yml")
    criteria = dag.search_iter_page.call_args.kwargs
    assert criteria["start"] == request["start"]
    assert criteria["end"] == request["end"]
    assert criteria["productType"] == "S1_SAR_GRD"
    assert criteria["polarizationMode"] == "VV VH"
    assert criteria["items_per_page"] == 25
    assert criteria["geom"].startswith("MULTIPOLYGON")


@pytest.fixture
def endpoint(monkeypatch):
    for key, value in {
        "API_KEY": "test-key",
        "GCPPROJECT": "project",
        "GCPREGION": "region",
        "QUEUE": "orchestrator",
        "FUNCTIONNAME": "historical",
        "ORCHESTRATOR_URL": "https://orchestrator.example/base/",
        "IS_DRY_RUN": "true",
    }.items():
        monkeypatch.setenv(key, value)
    batch = AsyncMock(return_value={"id": 42})
    monkeypatch.setattr(historical, "add_trigger_row", batch)
    tasks = MagicMock()
    monkeypatch.setattr(historical.tasks_v2, "CloudTasksClient", lambda: tasks)
    app = Flask(__name__)
    app.add_url_rule("/", view_func=lambda: historical.main(request), methods=["POST"])
    return app.test_client(), batch, tasks


def post(endpoint, payload):
    return endpoint[0].post(
        "/", json=payload, headers={"Authorization": "Bearer test-key"}
    )


def test_explicit_inventory_deduplicates_without_batch_or_queue(endpoint):
    response = post(
        endpoint, {"action": "list", "scene_ids": ["first", " first ", "second"]}
    )
    assert response.status_code == 200
    assert response.json == {
        "scene_count": 3,
        "filtered_scene_count": 2,
        "scene_ids": ["first", "second"],
    }
    endpoint[1].assert_not_called()
    endpoint[2].create_task.assert_not_called()


def test_s3_inventory_routes_filters_without_batch_or_queue(endpoint, monkeypatch):
    listing = MagicMock(return_value=["scene"])
    monkeypatch.setattr(historical, "list_sentinel1_scene_ids", listing)
    filtering = MagicMock(return_value=["scene"])
    monkeypatch.setattr(historical, "filter_sentinel1_scene_ids", filtering)
    response = post(
        endpoint,
        {
            "action": "list",
            "source": "s3",
            "start": "2025-01-01",
            "end": "2025-01-02",
            "modes": ["IW", "EW"],
            "polarizations": ["DV", "SV"],
        },
    )
    assert response.status_code == 200
    listing.assert_called_once_with(
        date(2025, 1, 1),
        date(2025, 1, 2),
        modes=["IW", "EW"],
        polarizations=["DV", "SV"],
    )
    filtering.assert_called_once_with(
        ["scene"], geometry=None, ocean_poly=None, apply_relevancy=True
    )
    endpoint[1].assert_not_called()
    endpoint[2].create_task.assert_not_called()


@pytest.mark.parametrize(
    "setting,override,expected",
    [("true", None, True), ("", None, False), ("true", False, False), ("", True, True)],
)
def test_batch_tracking_queue_and_dry_run_defaults_or_overrides(
    endpoint, monkeypatch, setting, override, expected
):
    monkeypatch.setenv("IS_DRY_RUN", setting)
    payload = {"scene_ids": ["first", "first", "second"], "manual_trigger": True}
    if override is not None:
        payload["dry_run"] = override
    response = post(endpoint, payload)
    assert response.status_code == 200
    assert response.json == {
        "scene_count": 3,
        "filtered_scene_count": 2,
        "queued_count": 2,
        "trigger_id": 42,
        "dry_run": expected,
    }
    batch = endpoint[1]
    batch.assert_awaited_once()
    assert batch.call_args.args == (3, 2)
    assert "cloud_function" in batch.call_args.kwargs["logs_url"]
    tasks = endpoint[2]
    assert tasks.create_task.call_count == 2
    for call, scene in zip(tasks.create_task.call_args_list, ["first", "second"]):
        request = call.kwargs["request"]
        assert request["parent"] == tasks.queue_path.return_value
        task = request["task"]
        assert task["dispatch_deadline"] == {"seconds": 1800}
        assert (
            task["http_request"]["url"]
            == "https://orchestrator.example/base/orchestrate"
        )
        assert task["http_request"]["headers"]["Authorization"] == "Bearer test-key"
        assert json.loads(task["http_request"]["body"]) == {
            "scene_id": scene,
            "trigger": 42,
            "dry_run": expected,
            "manual_trigger": True,
        }


def test_default_submission_does_not_force_manual_source_association_names(endpoint):
    assert post(endpoint, {"scene_ids": ["scene"]}).status_code == 200
    task = endpoint[2].create_task.call_args.kwargs["request"]["task"]
    assert "manual_trigger" not in json.loads(task["http_request"]["body"])


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"scene_ids": "scene"},
        {"scene_ids": [""]},
        {"scene_ids": [None]},
        {"scene_ids": [], "start": "2025-01-01"},
        {"scene_ids": [], "source": "s3"},
        {"scene_ids": [], "dry_run": "false"},
        {"scene_ids": [], "manual_trigger": 1},
        {"scene_ids": [], "action": []},
        {"source": []},
        {"source": "s3", "start": "2025-01-02", "end": "2025-01-01"},
        {"source": "s3", "start": "2025-01-01", "end": "2025-01-02", "geometry": {}},
        {"source": "s3", "start": "2025-01-01", "end": "2025-01-02", "modes": "IW"},
        {"start": "2025-01-01", "end": "2025-01-02"},
    ],
)
def test_invalid_requests_fail_before_discovery_or_side_effects(
    endpoint, monkeypatch, payload
):
    search = MagicMock(side_effect=AssertionError("must not search"))
    monkeypatch.setattr(historical, "handle_search", search)
    assert post(endpoint, payload).status_code == 400
    search.assert_not_called()
    endpoint[1].assert_not_called()
    endpoint[2].create_task.assert_not_called()


@pytest.mark.parametrize(
    "authorization", ["", " ", "Bearer", "Bearer wrong", "Basic test-key"]
)
def test_bad_authentication_returns_401(endpoint, authorization):
    response = endpoint[0].post(
        "/", json={"scene_ids": []}, headers={"Authorization": authorization}
    )
    assert response.status_code == 401
    endpoint[1].assert_not_called()


def test_discovery_failure_does_not_create_batch_or_tasks(endpoint, monkeypatch):
    monkeypatch.setattr(
        historical,
        "handle_search",
        MagicMock(side_effect=RuntimeError("listing failed")),
    )
    assert post(endpoint, {"scene_ids": []}).status_code == 500
    endpoint[1].assert_not_called()
    endpoint[2].create_task.assert_not_called()


def test_queue_failure_is_not_reported_as_success(endpoint):
    endpoint[2].create_task.side_effect = RuntimeError("queue failed")
    assert post(endpoint, {"scene_ids": ["scene"]}).status_code == 500


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_trigger_connection_is_closed_on_success_and_failure(monkeypatch, fail):
    connection = AsyncMock()
    connection.fetchrow.return_value = {"id": 42}
    if fail:
        connection.fetchrow.side_effect = RuntimeError("insert failed")
    monkeypatch.setattr(
        historical.asyncpg, "connect", AsyncMock(return_value=connection)
    )
    if fail:
        with pytest.raises(RuntimeError, match="insert failed"):
            await historical.add_trigger_row(3, 2, "logs")
    else:
        assert await historical.add_trigger_row(3, 2, "logs") == {"id": 42}
        assert connection.fetchrow.call_args.args[1:] == (
            3,
            2,
            "logs",
            "HISTORICAL_RUN",
        )
    connection.close.assert_awaited_once()


@pytest.mark.parametrize("action", ["list", "enqueue"])
@pytest.mark.parametrize("use_region", [False, True])
def test_s3_filters_footprints_before_batch_and_queue(
    endpoint, monkeypatch, action, use_region
):
    def scene(n, pol="DV", res="H"):
        return (
            f"S1A_IW_GRD{res}_1S{pol}_20250101T000000_20250101T000025_000001_000001_{n}"
        )

    footprints = {
        scene("ocean"): box(0, 0, 1, 1),
        scene("outside"): box(4, 4, 5, 5),
        scene("land"): box(2, 2, 3, 3),
    }
    ids = [*footprints, scene("hh", "DH"), scene("low", res="F")]
    monkeypatch.setattr(historical, "list_sentinel1_scene_ids", lambda *a, **kw: ids)
    monkeypatch.setattr(
        inventory,
        "load_ocean_poly",
        lambda: box(-1, -1, 1.5, 1.5).union(box(4, 4, 6, 6)),
    )
    requested = []

    def metadata(request):
        scene_id = request.url.path.split("/")[-2]
        requested.append(scene_id)
        assert request.url.path.startswith("/GRD/2025/1/1/IW/DV/")
        return httpx.Response(200, json={"footprint": mapping(footprints[scene_id])})

    client = httpx.Client(transport=httpx.MockTransport(metadata))
    monkeypatch.setattr(inventory.httpx, "Client", lambda **kw: client)
    payload = {
        "source": "s3",
        "start": "2025-01-01",
        "end": "2025-01-01",
        "action": action,
    }
    if use_region:
        payload["geometry"] = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": mapping(box(-1, -1, 3.5, 3.5))}
            ],
        }
    response = post(endpoint, payload)
    expected = [scene("ocean")] if use_region else [scene("ocean"), scene("outside")]
    assert response.status_code == 200
    assert response.json["scene_count"] == 5
    assert response.json["filtered_scene_count"] == len(expected)
    assert requested == list(footprints)
    if action == "list":
        assert response.json["scene_ids"] == expected
        endpoint[1].assert_not_called()
        endpoint[2].create_task.assert_not_called()
    else:
        assert endpoint[1].call_args.args == (5, len(expected))
        tasks = endpoint[2].create_task.call_args_list
        assert [
            json.loads(call.kwargs["request"]["task"]["http_request"]["body"])[
                "scene_id"
            ]
            for call in tasks
        ] == expected


@pytest.mark.parametrize("failure", ["http", "missing_footprint"])
def test_s3_metadata_failure_does_not_queue_partial_batch(
    endpoint, monkeypatch, failure
):
    ids = [
        f"S1A_IW_GRDH_1SDV_20250101T000000_20250101T000025_000001_000001_{n}"
        for n in range(2)
    ]
    monkeypatch.setattr(historical, "list_sentinel1_scene_ids", lambda *a, **kw: ids)
    monkeypatch.setattr(inventory, "load_ocean_poly", lambda: box(-1, -1, 2, 2))

    def metadata(request):
        if ids[0] in request.url.path:
            return httpx.Response(200, json={"footprint": mapping(box(0, 0, 1, 1))})
        return httpx.Response(503 if failure == "http" else 200, json={})

    client = httpx.Client(transport=httpx.MockTransport(metadata))
    monkeypatch.setattr(inventory.httpx, "Client", lambda **kw: client)
    response = post(
        endpoint, {"source": "s3", "start": "2025-01-01", "end": "2025-01-01"}
    )
    assert response.status_code == 500
    endpoint[1].assert_not_called()
    endpoint[2].create_task.assert_not_called()


@pytest.mark.parametrize(
    "geometry",
    [
        None,
        {},
        [],
        {"type": "FeatureCollection", "features": []},
        {
            "type": "FeatureCollection",
            "features": [{"geometry": {"type": "Point", "coordinates": [0, 0]}}],
        },
    ],
)
def test_invalid_s3_region_fails_before_discovery(endpoint, monkeypatch, geometry):
    listing = MagicMock(side_effect=AssertionError("must not list"))
    monkeypatch.setattr(historical, "list_sentinel1_scene_ids", listing)
    response = post(
        endpoint,
        {
            "source": "s3",
            "start": "2025-01-01",
            "end": "2025-01-01",
            "geometry": geometry,
        },
    )
    assert response.status_code == 400
    listing.assert_not_called()
    endpoint[1].assert_not_called()
    endpoint[2].create_task.assert_not_called()


@pytest.mark.parametrize("action", ["list", "enqueue"])
def test_s3_relevancy_bypass_keeps_all_ids_without_metadata(
    endpoint, monkeypatch, action
):
    monkeypatch.setattr(
        historical, "list_sentinel1_scene_ids", lambda *a, **kw: ["land", "hh", "land"]
    )
    monkeypatch.setattr(
        inventory.httpx,
        "Client",
        MagicMock(side_effect=AssertionError("no metadata requests")),
    )
    response = post(
        endpoint,
        {
            "source": "s3",
            "start": "2025-01-01",
            "end": "2025-01-02",
            "apply_relevancy": False,
            "action": action,
        },
    )
    assert response.status_code == 200
    assert response.json["scene_count"] == 3
    assert response.json["filtered_scene_count"] == 2
    if action == "list":
        assert response.json["scene_ids"] == ["land", "hh"]
        endpoint[1].assert_not_called()
        endpoint[2].create_task.assert_not_called()
    else:
        assert endpoint[1].call_args.args == (3, 2)
        assert endpoint[2].create_task.call_count == 2


def test_s3_region_still_applies_when_relevancy_is_disabled(monkeypatch):
    ids = [
        f"S1A_EW_GRDM_1SDH_20250101T000000_20250101T000025_000001_000001_{n}"
        for n in range(2)
    ]

    def metadata(request):
        footprint = box(0, 0, 1, 1) if ids[0] in request.url.path else box(5, 5, 6, 6)
        return httpx.Response(200, json={"footprint": mapping(footprint)})

    client = httpx.Client(transport=httpx.MockTransport(metadata))
    monkeypatch.setattr(inventory.httpx, "Client", lambda **kw: client)
    monkeypatch.setattr(
        inventory,
        "load_ocean_poly",
        MagicMock(side_effect=AssertionError("no ocean filtering")),
    )
    region = {
        "type": "FeatureCollection",
        "features": [{"geometry": mapping(box(-1, -1, 2, 2))}],
    }
    assert inventory.filter_sentinel1_scene_ids(
        ids, geometry=region, apply_relevancy=False
    ) == [ids[0]]


@pytest.mark.parametrize("value", ["false", 0, None])
def test_s3_relevancy_option_requires_boolean(endpoint, value):
    response = post(
        endpoint,
        {
            "source": "s3",
            "start": "2025-01-01",
            "end": "2025-01-02",
            "apply_relevancy": value,
        },
    )
    assert response.status_code == 400
    endpoint[1].assert_not_called()
    endpoint[2].create_task.assert_not_called()
