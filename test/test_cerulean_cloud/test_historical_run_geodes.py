"""Exercise the real pinned EODAG adapter with mocked GEODES HTTP responses."""

import json as jsonlib
from datetime import date, timedelta
from threading import Barrier
from unittest.mock import Mock

import pytest
import requests
from shapely.geometry import box, mapping

from cerulean_cloud.cloud_function_historical_run import main as historical


@pytest.fixture
def geodes(monkeypatch, tmp_path):
    import eodag.api.core

    # EODAG writes its config/index under expanduser('~') even when given a
    # provider file. Keep tests independent of any personal EODAG configuration.
    expanduser = eodag.api.core.os.path.expanduser
    monkeypatch.setattr(
        eodag.api.core.os.path,
        "expanduser",
        lambda path: str(tmp_path) if path == "~" else expanduser(path),
    )
    calls = []
    pages = {}

    def post(session, url, json, **kwargs):
        assert url == "https://geodes-portal.cnes.fr/api/stac/items"
        assert "auth" not in kwargs
        calls.append(jsonlib.loads(jsonlib.dumps(json)))
        start = json["query"]["start_datetime"]["gte"][:10]
        key = (start, json["page"])
        if key not in pages:
            key = json["page"]
        queued = pages[key]
        if isinstance(queued, list):
            status, data = queued.pop(0)
            if not queued:
                del pages[key]
        else:
            status, data = pages.pop(key)
        if callable(data):
            data = data()
        response = requests.Response()
        response.status_code = status
        if status == 429:
            response.headers["Retry-After"] = "7"
        response.url = url
        response._content = jsonlib.dumps(data).encode()
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    return pages, calls


def feature(number, footprint=None, polarization="VV VH", resolution="H"):
    polarization_code = "SV" if polarization == "VV" else "DV"
    beam = "S3" if resolution == "F" else "IW"
    scene_id = (
        f"S1A_{beam}_GRD{resolution}_1S{polarization_code}_20250101T114435_20250101T114504_"
        f"057250_070AE5_{number:04X}"
    )
    return {
        "type": "Feature",
        "id": f"URN:FEATURE:DATA:gdh:catalog-{number}:V1",
        "geometry": mapping(box(0, 0, 1, 1) if footprint is None else footprint),
        "properties": {
            "identifier": scene_id,
            "dataset": "PEPS_S1_L1",
            "product:type": "GRD",
            "sar:polarizations": polarization,
            "sar:instrument_mode": "SM" if resolution == "F" else "IW",
            "start_datetime": "2025-01-01T11:44:35.035Z",
            "end_datetime": "2025-01-01T11:45:04.052Z",
        },
        "assets": {},
        "links": [],
    }


def search(modes=None, with_geometry=True, end="2025-01-02"):
    payload = {
        "source": "eodag",
        "start": "2025-01-01T00:00:00Z",
        "end": end,
        **({"modes": modes} if modes is not None else {}),
        "geometry": {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": mapping(box(-1, -1, 20, 20))}],
        },
    }
    if not with_geometry:
        del payload["geometry"]
    return historical.handle_search(payload, ocean_poly=box(-1, -1, 2, 2))


@pytest.mark.parametrize("modes", [None, ["IW"], ["EW", "IW", "SM"]])
@pytest.mark.parametrize("polarization", ["VV", "VV VH"])
def test_geodes_pagination_filters_and_sentinel_id_mapping(geodes, modes, polarization):
    pages, calls = geodes
    pages[1] = (
        200,
        {
            "features": [feature(i, polarization=polarization) for i in range(80)],
            "context": {"matched": 81},
            "links": [{"rel": "next", "href": "https://unused.invalid/next"}],
        },
    )
    pages[2] = (
        200,
        {
            "features": [feature(80, box(10, 10, 11, 11), polarization)],
            "context": {"matched": 81},
        },
    )
    total, selected = search(modes)
    assert total == 81
    assert selected == [
        feature(i, polarization=polarization)["properties"]["identifier"]
        for i in range(80)
    ]
    assert not pages
    assert [body["page"] for body in calls] == [1, 2]
    for body in calls:
        assert body["limit"] == 80
        assert body["query"]["dataset"] == {"in": ["PEPS_S1_L1"]}
        assert body["query"]["product:type"] == {"in": ["GRD"]}
        assert body["query"]["sar:polarizations"] == {"in": ["VV", "VV VH"]}
        if modes is None:
            assert "sar:instrument_mode" not in body["query"]
        else:
            assert body["query"]["sar:instrument_mode"] == {"in": modes}
        assert body["query"]["start_datetime"]["gte"].startswith("2025-01-01T00:00:00")
        assert body["query"]["start_datetime"]["lte"].startswith("2025-01-02T00:00:00")
        assert body["intersects"]["type"] == "MultiPolygon"
        assert body["sortBy"] == [
            {"field": "start_datetime", "direction": "asc"},
            {"field": "identifier", "direction": "asc"},
        ]


def test_geodes_keeps_only_high_and_medium_resolution(geodes):
    pages, _ = geodes
    products = [
        feature(i, resolution=resolution)
        for i, resolution in enumerate(["H", "M", "F", "_"])
    ]
    pages[1] = (200, {"features": products, "context": {"matched": 4}})

    assert search() == (
        4,
        [product["properties"]["identifier"] for product in products[:2]],
    )


def test_geodes_bundled_ocean_mask_keeps_ocean_and_coastal_scenes(geodes):
    pages, _ = geodes
    products = [
        feature(0, box(2, 55, 3, 56)),  # North Sea
        feature(1, box(-1, 54.3, 0, 54.6)),  # English coast
        feature(2, box(-100, 38, -99, 39)),  # Kansas
        feature(3, box(51.2, 42.2, 51.3, 42.3)),  # Caspian Sea
    ]
    pages[1] = (200, {"features": products, "context": {"matched": 4}})

    assert historical.handle_search({"start": "2025-01-01", "end": "2025-01-02"}) == (
        4,
        [product["properties"]["identifier"] for product in products[:2]],
    )


def test_geodes_empty_search_is_valid(geodes):
    pages, calls = geodes
    pages[1] = (200, {"features": [], "context": {"matched": 0}})
    assert search() == (0, [])
    assert len(calls) == 1


def test_geodes_global_search_omits_spatial_filter_on_every_page(geodes):
    pages, calls = geodes
    pages[1] = (
        200,
        {"features": [feature(i) for i in range(80)], "context": {"matched": 81}},
    )
    pages[2] = (
        200,
        {"features": [feature(80, box(10, 10, 11, 11))], "context": {"matched": 81}},
    )
    assert search(["IW"], with_geometry=False) == (
        81,
        [feature(i)["properties"]["identifier"] for i in range(80)],
    )
    assert [body["page"] for body in calls] == [1, 2]
    assert all("intersects" not in body for body in calls)
    assert all(body["query"]["sar:instrument_mode"] == {"in": ["IW"]} for body in calls)


def test_geodes_failure_after_first_page_does_not_return_partial_results(
    geodes, monkeypatch
):
    pages, calls = geodes
    monkeypatch.setattr(historical.time, "sleep", lambda _: None)
    pages[1] = (
        200,
        {"features": [feature(i) for i in range(80)], "context": {"matched": 81}},
    )
    pages[2] = [(503, {"error": "unavailable"})] * 3
    with pytest.raises(RuntimeError, match="page 2 .* failed after 3 attempts"):
        search()
    assert [body["page"] for body in calls] == [1, 2, 2, 2]


@pytest.mark.parametrize("status", [429, 503])
def test_geodes_transient_page_failure_retries_without_duplicates(
    geodes, monkeypatch, status
):
    pages, calls = geodes
    sleep = Mock()
    monkeypatch.setattr(historical.time, "sleep", sleep)
    pages[1] = (
        200,
        {"features": [feature(i) for i in range(80)], "context": {"matched": 81}},
    )
    pages[2] = [
        (status, {"error": "unavailable"}),
        (200, {"features": [feature(80)], "context": {"matched": 81}}),
    ]
    total, selected = search(["IW"])
    assert total == 81
    assert selected == [feature(i)["properties"]["identifier"] for i in range(81)]
    assert [body["page"] for body in calls] == [1, 2, 2]
    assert calls[1] == calls[2]
    if status == 429:
        sleep.assert_called_once_with(7)


def test_geodes_repeated_page_is_an_error(geodes):
    pages, _ = geodes
    full_page = {
        "features": [feature(i) for i in range(80)],
        "context": {"matched": 160},
    }
    pages[1] = (200, full_page)
    pages[2] = (200, full_page)
    with pytest.raises(RuntimeError, match="page 2 repeated catalog record"):
        search()


def test_geodes_weekly_windows_deduplicate_boundary_records(geodes):
    pages, calls = geodes
    boundary = feature(1)
    boundary["properties"]["start_datetime"] = "2025-01-08T00:00:00.000Z"
    pages["2025-01-01", 1] = (
        200,
        {"features": [feature(0), boundary], "context": {"matched": 2}},
    )
    pages["2025-01-08", 1] = (
        200,
        {"features": [boundary, feature(2)], "context": {"matched": 2}},
    )
    total, selected = search(end="2025-01-10")
    assert total == 3
    assert selected == [feature(i)["properties"]["identifier"] for i in range(3)]
    boundaries = sorted(call["query"]["start_datetime"].items() for call in calls)
    assert {dict(pair)["gte"][:10] for pair in boundaries} == {
        "2025-01-01",
        "2025-01-08",
    }
    assert {dict(pair)["lte"][:10] for pair in boundaries} == {
        "2025-01-08",
        "2025-01-10",
    }


def test_geodes_starts_32_weekly_windows_concurrently(geodes, monkeypatch):
    monkeypatch.delenv("GEODES_PAGE_WORKERS", raising=False)
    pages, calls = geodes
    barrier = Barrier(32, timeout=10)

    def first_page():
        barrier.wait()
        return {"features": [], "context": {"matched": 0}}

    empty = {"features": [], "context": {"matched": 0}}
    year = date(2025, 1, 1)
    for week in range(53):
        start = year + timedelta(weeks=week)
        pages[start.isoformat(), 1] = (200, first_page if week < 32 else empty)
    assert search(end="2026-01-01") == (0, [])
    assert not pages
    assert len(calls) == 53


def test_geodes_requests_pages_in_the_same_window_concurrently(geodes, monkeypatch):
    monkeypatch.setenv("GEODES_PAGE_WORKERS", "2")
    pages, calls = geodes
    barrier = Barrier(2, timeout=5)

    def later_page(start, end):
        def response():
            barrier.wait()
            return {
                "features": [feature(i) for i in range(start, end)],
                "context": {"matched": 161},
            }

        return response

    pages[1] = (
        200,
        {"features": [feature(i) for i in range(80)], "context": {"matched": 161}},
    )
    pages[2] = (200, later_page(80, 160))
    pages[3] = (200, later_page(160, 161))
    total, selected = search()
    assert total == 161
    assert selected == [feature(i)["properties"]["identifier"] for i in range(161)]
    assert [body["page"] for body in calls[:1]] == [1]
    assert {body["page"] for body in calls[1:]} == {2, 3}


@pytest.mark.parametrize(
    "matched,match",
    [(81, "returned 0 of 1 expected records"), (82, "changed the match count")],
)
def test_geodes_rejects_incomplete_inventory(geodes, matched, match):
    pages, _ = geodes
    pages[1] = (
        200,
        {"features": [feature(i) for i in range(80)], "context": {"matched": 81}},
    )
    pages[2] = (200, {"features": [], "context": {"matched": matched}})
    with pytest.raises(RuntimeError, match=match):
        search()
