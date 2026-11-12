"""Historical selection and queueing with mocked cloud APIs and a local test DB."""

import base64
import json
import sys
import uuid
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from xml.sax.saxutils import escape

import httpx
import pytest
import requests
from flask import Flask, request
from shapely.geometry import box, mapping

from cerulean_cloud.cloud_function_historical_run import main as historical
from cerulean_cloud.cloud_function_historical_run import inventory


QUEUE_PATH = "projects/project/locations/region/queues/orchestrator"
OPERATION_NAME = "projects/project/locations/region/operations/batch"


def operation_response(body, status=200):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode()
    return response


def queued_requests(endpoint):
    return [
        request
        for call in endpoint[2].post.call_args_list
        for request in call.kwargs["json"]["requests"]
    ]


def task_payload(task):
    return json.loads(base64.b64decode(task["httpRequest"]["body"]))


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
    monkeypatch.setattr(
        historical, "get_processed_scene_ids", AsyncMock(return_value=set())
    )
    session = MagicMock(spec=historical.AuthorizedSession)
    session.__enter__.return_value = session

    def create_batch(url, **kwargs):
        return operation_response(
            {
                "name": OPERATION_NAME,
                "done": True,
                "response": {
                    "tasks": [
                        {"name": f"{QUEUE_PATH}/tasks/{index}"}
                        for index, _ in enumerate(kwargs["json"]["requests"])
                    ]
                },
            }
        )

    session.post.side_effect = create_batch
    session.get.side_effect = AssertionError("unexpected operation polling")
    monkeypatch.setattr(
        historical.google.auth, "default", MagicMock(return_value=(object(), "project"))
    )
    monkeypatch.setattr(
        historical, "AuthorizedSession", MagicMock(return_value=session)
    )
    app = Flask(__name__)
    app.add_url_rule("/", view_func=lambda: historical.main(request), methods=["POST"])
    return app.test_client(), batch, session


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
        "already_processed_scene_count": 0,
        "scene_ids": ["first", "second"],
    }
    endpoint[1].assert_not_called()
    endpoint[2].post.assert_not_called()


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
    endpoint[2].post.assert_not_called()


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
        "already_processed_scene_count": 0,
        "queued_count": 2,
        "trigger_id": 42,
        "dry_run": expected,
    }
    batch = endpoint[1]
    batch.assert_awaited_once()
    assert batch.call_args.args == (3, 2)
    assert "cloud_function" in batch.call_args.kwargs["logs_url"]
    session = endpoint[2]
    session.post.assert_called_once()
    call = session.post.call_args
    assert call.args == (
        f"{historical.CLOUD_TASKS_API}/{QUEUE_PATH}/tasks:batchCreate",
    )
    assert (
        str(uuid.UUID(call.kwargs["json"]["requestId"]))
        == call.kwargs["json"]["requestId"]
    )
    historical.google.auth.default.assert_called_once_with(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    for queued_request, scene in zip(queued_requests(endpoint), ["first", "second"]):
        assert queued_request["parent"] == QUEUE_PATH
        task = queued_request["task"]
        assert task["dispatchDeadline"] == "1800s"
        assert task["httpRequest"]["httpMethod"] == "POST"
        assert (
            task["httpRequest"]["url"]
            == "https://orchestrator.example/base/orchestrate"
        )
        assert task["httpRequest"]["headers"]["Authorization"] == "Bearer test-key"
        assert task_payload(task) == {
            "scene_id": scene,
            "trigger": 42,
            "dry_run": expected,
            "manual_trigger": True,
        }


def test_default_submission_does_not_force_manual_source_association_names(endpoint):
    assert post(endpoint, {"scene_ids": ["scene"]}).status_code == 200
    task = queued_requests(endpoint)[0]["task"]
    assert "manual_trigger" not in task_payload(task)


@pytest.mark.parametrize("scene_count", [0, 1, 100, 101, 250])
def test_submission_chunks_scenes_without_losing_or_repeating_them(
    endpoint, scene_count
):
    scenes = [f"scene-{index}" for index in range(scene_count)]
    response = post(endpoint, {"scene_ids": scenes})
    assert response.status_code == 200
    assert response.json["queued_count"] == scene_count
    batches = [call.kwargs["json"] for call in endpoint[2].post.call_args_list]
    assert [len(batch["requests"]) for batch in batches] == [
        min(100, scene_count - start) for start in range(0, scene_count, 100)
    ]
    assert len({batch["requestId"] for batch in batches}) == len(batches)
    assert [
        task_payload(request["task"])["scene_id"]
        for request in queued_requests(endpoint)
    ] == scenes
    if not scenes:
        historical.google.auth.default.assert_not_called()
        historical.AuthorizedSession.assert_not_called()


def test_submission_waits_for_operation_without_resubmitting(endpoint, monkeypatch):
    session = endpoint[2]
    running = {"name": OPERATION_NAME}
    session.post.side_effect = [operation_response(running)]
    session.get.side_effect = [
        operation_response(running),
        operation_response(
            {
                "name": OPERATION_NAME,
                "done": True,
                "response": {"tasks": [{"name": "task"}]},
            }
        ),
    ]
    monkeypatch.setattr(historical.time, "sleep", MagicMock())
    response = post(endpoint, {"scene_ids": ["scene"]})
    assert response.status_code == 200
    assert response.json["queued_count"] == 1
    session.post.assert_called_once()
    assert session.get.call_count == 2
    for call in session.get.call_args_list:
        assert call.args == (f"{historical.CLOUD_TASKS_API}/{OPERATION_NAME}",)
        assert 0 < call.kwargs["timeout"] <= 30


@pytest.mark.parametrize("failure", ["http", "operation", "unconfirmed"])
def test_unsuccessful_batch_is_not_reported_as_queued(endpoint, failure):
    body = {"name": OPERATION_NAME, "done": True}
    if failure == "operation":
        body["error"] = {"code": 13, "message": "Internal error"}
    endpoint[2].post.side_effect = [
        operation_response(body, 403 if failure == "http" else 200)
    ]
    response = post(endpoint, {"scene_ids": ["scene"]})
    assert response.status_code == 500
    endpoint[2].post.assert_called_once()


def test_partial_batch_stops_later_batches_and_logs_confirmed_count(endpoint, caplog):
    session = endpoint[2]
    create_batch = session.post.side_effect

    def partial_second_batch(url, **kwargs):
        if session.post.call_count == 1:
            return create_batch(url, **kwargs)
        return operation_response(
            {
                "name": OPERATION_NAME,
                "done": True,
                "metadata": {
                    "state": "PARTIALLY_SUCCEEDED",
                    "failedRequests": {"0": {"code": 8, "message": "quota exhausted"}},
                },
                "response": {
                    "tasks": [{"name": f"task-{index}"} for index in range(99)]
                },
            }
        )

    session.post.side_effect = partial_second_batch
    response = post(endpoint, {"scene_ids": [f"scene-{index}" for index in range(201)]})
    assert response.status_code == 500
    assert session.post.call_count == 2
    assert OPERATION_NAME in caplog.text
    assert "199/201 tasks confirmed" in caplog.text
    assert "batch_start=100" in caplog.text
    assert "quota exhausted" in caplog.text
    assert "test-key" not in caplog.text


def test_operation_polling_http_failure_is_not_reported_as_success(
    endpoint, monkeypatch
):
    endpoint[2].post.side_effect = [operation_response({"name": OPERATION_NAME})]
    endpoint[2].get.side_effect = [operation_response({}, 503)]
    monkeypatch.setattr(historical.time, "sleep", MagicMock())
    assert post(endpoint, {"scene_ids": ["scene"]}).status_code == 500
    endpoint[2].post.assert_called_once()
    endpoint[2].get.assert_called_once()


def test_operation_polling_stops_at_enqueue_deadline(endpoint, monkeypatch, caplog):
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(historical, "ENQUEUE_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(historical.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(
        historical.time,
        "sleep",
        lambda seconds: setattr(clock, "now", clock.now + seconds),
    )
    endpoint[2].post.side_effect = [operation_response({"name": OPERATION_NAME})]
    endpoint[2].get.side_effect = lambda *args, **kwargs: operation_response(
        {"name": OPERATION_NAME}
    )
    assert post(endpoint, {"scene_ids": ["scene"]}).status_code == 500
    assert clock.now == 2
    endpoint[2].post.assert_called_once()
    endpoint[2].get.assert_called_once()
    assert "batchCreate timed out" in caplog.text


def test_discovery_time_counts_toward_enqueue_deadline(endpoint, monkeypatch):
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(historical.time, "monotonic", lambda: clock.now)

    def slow_search(payload):
        clock.now = historical.ENQUEUE_TIMEOUT_SECONDS + 1
        return 1, ["scene"]

    monkeypatch.setattr(historical, "handle_search", slow_search)
    assert post(endpoint, {"scene_ids": ["scene"]}).status_code == 500
    endpoint[2].post.assert_not_called()


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
    endpoint[2].post.assert_not_called()


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
    endpoint[2].post.assert_not_called()


def test_queue_failure_is_not_reported_as_success(endpoint):
    endpoint[2].post.side_effect = RuntimeError("queue failed")
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


@pytest.mark.parametrize("source", ["explicit", "s3", "eodag"])
@pytest.mark.parametrize("action", ["list", "enqueue"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_processed_scenes_are_excluded_from_every_request_path(
    endpoint, monkeypatch, source, action, dry_run
):
    scenes = ["new", "completed", "failed", "completed", "unfinished"]
    if source == "explicit":
        payload = {"scene_ids": scenes}
    else:
        monkeypatch.setattr(historical, "handle_search", lambda _: (5, scenes))
        payload = {"source": source, "start": "2025-01-01", "end": "2025-01-02"}
        if source == "s3":
            payload["apply_relevancy"] = False
        else:
            payload["geometry"] = {"type": "FeatureCollection", "features": []}
    historical.get_processed_scene_ids.return_value = {"completed"}
    response = post(endpoint, {**payload, "action": action, "dry_run": dry_run})
    assert response.status_code == 200
    assert response.json["scene_count"] == 5
    assert response.json["already_processed_scene_count"] == 1
    assert response.json["filtered_scene_count"] == 3
    historical.get_processed_scene_ids.assert_awaited_once_with(
        ["new", "completed", "failed", "unfinished"]
    )
    expected = ["new", "failed", "unfinished"]
    if action == "list":
        assert response.json["scene_ids"] == expected
        endpoint[1].assert_not_called()
        endpoint[2].post.assert_not_called()
    else:
        assert response.json["queued_count"] == 3
        assert endpoint[1].call_args.args == (5, 3)
        assert [
            task_payload(request["task"])["scene_id"]
            for request in queued_requests(endpoint)
        ] == expected


def test_all_scenes_already_processed_creates_no_tasks(endpoint):
    historical.get_processed_scene_ids.return_value = {"completed"}
    response = post(endpoint, {"scene_ids": ["completed", "completed"]})
    assert response.status_code == 200
    assert response.json["already_processed_scene_count"] == 1
    assert response.json["filtered_scene_count"] == response.json["queued_count"] == 0
    assert endpoint[1].call_args.args == (2, 0)
    endpoint[2].post.assert_not_called()
    historical.google.auth.default.assert_not_called()


@pytest.mark.asyncio
async def test_local_runner_uses_db_exclusion_tracking_and_queue_without_http(
    endpoint, monkeypatch
):
    monkeypatch.setattr(
        historical,
        "verify_api_key",
        MagicMock(side_effect=AssertionError("no HTTP request")),
    )
    monkeypatch.setattr(
        historical,
        "make_cloud_function_logs_url",
        MagicMock(side_effect=AssertionError("no cloud function logs for local runs")),
    )
    historical.get_processed_scene_ids.return_value = {"completed"}
    result = await historical.run_historical(
        {
            "scene_ids": ["new", "completed", "new"],
            "dry_run": False,
            "manual_trigger": True,
        }
    )
    assert result == {
        "scene_count": 3,
        "filtered_scene_count": 1,
        "already_processed_scene_count": 1,
        "trigger_id": 42,
        "queued_count": 1,
        "dry_run": False,
    }
    endpoint[1].assert_awaited_once_with(3, 1, logs_url="")
    assert task_payload(queued_requests(endpoint)[0]["task"]) == {
        "scene_id": "new",
        "trigger": 42,
        "dry_run": False,
        "manual_trigger": True,
    }


@pytest.mark.asyncio
async def test_local_discovery_does_not_consume_queue_submission_budget(
    endpoint, monkeypatch
):
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(historical.time, "monotonic", lambda: clock.now)

    def slow_search(payload):
        clock.now = historical.ENQUEUE_TIMEOUT_SECONDS + 1
        return 1, ["scene"]

    monkeypatch.setattr(historical, "handle_search", slow_search)
    result = await historical.run_historical({"scene_ids": ["scene"]})
    assert result["queued_count"] == 1
    endpoint[2].post.assert_called_once()


@pytest.mark.asyncio
async def test_local_preview_checks_db_without_trigger_or_tasks(endpoint):
    historical.get_processed_scene_ids.return_value = {"completed"}
    result = await historical.run_historical(
        {"action": "list", "scene_ids": ["new", "completed"]}
    )
    assert result["scene_ids"] == ["new"]
    assert result["already_processed_scene_count"] == 1
    endpoint[1].assert_not_awaited()
    endpoint[2].post.assert_not_called()


@pytest.mark.asyncio
async def test_local_invalid_request_stops_before_db_or_queue(endpoint):
    with pytest.raises(ValueError, match="dry_run must be a boolean"):
        await historical.run_historical({"scene_ids": ["scene"], "dry_run": "false"})
    historical.get_processed_scene_ids.assert_not_awaited()
    endpoint[1].assert_not_awaited()
    endpoint[2].post.assert_not_called()


@pytest.mark.parametrize("action", ["list", "enqueue"])
def test_processed_lookup_failure_stops_before_trigger_and_queue(endpoint, action):
    historical.get_processed_scene_ids.side_effect = RuntimeError(
        "database unavailable"
    )
    assert post(endpoint, {"scene_ids": ["scene"], "action": action}).status_code == 500
    endpoint[1].assert_not_called()
    endpoint[2].post.assert_not_called()


@pytest.mark.asyncio
async def test_empty_processed_lookup_never_connects(monkeypatch):
    connect = AsyncMock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr(historical.asyncpg, "connect", connect)
    assert await historical.get_processed_scene_ids([]) == set()
    connect.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_processed_lookup_closes_connection_on_success_and_failure(
    monkeypatch, fail
):
    connection = AsyncMock()
    connection.fetch.return_value = [{"scene_id": "completed"}]
    if fail:
        connection.fetch.side_effect = RuntimeError("lookup failed")
    monkeypatch.setenv("DB_URL", "postgresql://test-db")
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(historical.asyncpg, "connect", connect)
    if fail:
        with pytest.raises(RuntimeError, match="lookup failed"):
            await historical.get_processed_scene_ids(["completed", "new"])
    else:
        assert await historical.get_processed_scene_ids(["completed", "new"]) == {
            "completed"
        }
    connect.assert_awaited_once_with(
        "postgresql://test-db", timeout=30, command_timeout=30
    )
    assert connection.fetch.call_args.args[1:] == (["completed", "new"],)
    connection.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_processed_lookup_requires_any_successful_run(postgresql, monkeypatch):
    # Exercise the real query on local PostgreSQL; inference output is not needed.
    url = (
        f"postgresql://{postgresql.info.user}@{postgresql.info.host}:"
        f"{postgresql.info.port}/{postgresql.info.dbname}"
    )
    monkeypatch.setenv("DB_URL", url)
    connection = await historical.asyncpg.connect(url)
    try:
        await connection.execute(
            """
            CREATE TABLE sentinel1_grd (id bigint PRIMARY KEY, scene_id text UNIQUE);
            CREATE TABLE orchestrator_run (sentinel1_grd bigint, success boolean, model integer);
            INSERT INTO sentinel1_grd VALUES
                (1, 'completed'), (2, 'failed'), (3, 'unfinished'),
                (4, 'metadata-only'), (5, 'unrequested');
            INSERT INTO orchestrator_run VALUES
                (1, TRUE, 1), (1, TRUE, 2), (1, FALSE, 2), (1, NULL, 2),
                (2, FALSE, 1), (3, NULL, 1), (5, TRUE, 1);
            """
        )
        assert await historical.get_processed_scene_ids(
            ["new", "completed", "failed", "unfinished", "metadata-only", "completed"]
        ) == {"completed"}
    finally:
        await connection.close()


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
        endpoint[2].post.assert_not_called()
    else:
        assert endpoint[1].call_args.args == (5, len(expected))
        assert [
            task_payload(request["task"])["scene_id"]
            for request in queued_requests(endpoint)
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
    endpoint[2].post.assert_not_called()


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
    endpoint[2].post.assert_not_called()


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
        endpoint[2].post.assert_not_called()
    else:
        assert endpoint[1].call_args.args == (3, 2)
        assert len(queued_requests(endpoint)) == 2


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
    endpoint[2].post.assert_not_called()
