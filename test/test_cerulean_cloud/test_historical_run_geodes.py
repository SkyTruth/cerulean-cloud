"""Exercise the real pinned EODAG adapter with mocked GEODES HTTP responses."""

import json as jsonlib

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

    def post(url, json, **kwargs):
        assert url == "https://geodes-portal.cnes.fr/api/stac/items"
        assert "auth" not in kwargs
        calls.append(json)
        status, data = pages.pop(json["page"])
        response = requests.Response()
        response.status_code = status
        response.url = url
        response._content = jsonlib.dumps(data).encode()
        return response

    monkeypatch.setattr(requests, "post", post)
    return pages, calls


def feature(number, footprint=None):
    scene_id = (
        f"S1A_IW_GRDH_1SDV_20250101T114435_20250101T114504_057250_070AE5_{number:04X}"
    )
    return {
        "type": "Feature",
        "id": f"URN:FEATURE:DATA:gdh:catalog-{number}:V1",
        "geometry": mapping(box(0, 0, 1, 1) if footprint is None else footprint),
        "properties": {
            "identifier": scene_id,
            "dataset": "PEPS_S1_L1",
            "product:type": "GRD",
            "sar:polarizations": "VV VH",
            "start_datetime": "2025-01-01T11:44:35.035Z",
            "end_datetime": "2025-01-01T11:45:04.052Z",
        },
        "assets": {},
        "links": [],
    }


def search():
    return historical.handle_search(
        {
            "source": "eodag",
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-02",
            "geometry": {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": mapping(box(-1, -1, 20, 20))}
                ],
            },
        },
        ocean_poly=box(-1, -1, 2, 2),
    )


def test_geodes_pagination_filters_and_sentinel_id_mapping(geodes):
    pages, calls = geodes
    pages[1] = (
        200,
        {
            "features": [feature(i) for i in range(25)],
            "context": {"matched": 26},
            "links": [{"rel": "next", "href": "https://unused.invalid/next"}],
        },
    )
    pages[2] = (
        200,
        {"features": [feature(25, box(10, 10, 11, 11))], "context": {"matched": 26}},
    )
    total, selected = search()
    assert total == 26
    assert selected == [feature(i)["properties"]["identifier"] for i in range(25)]
    assert not pages
    assert [body["page"] for body in calls] == [1, 2]
    for body in calls:
        assert body["limit"] == 25
        assert body["query"]["dataset"] == {"in": ["PEPS_S1_L1"]}
        assert body["query"]["product:type"] == {"in": ["GRD"]}
        assert body["query"]["sar:polarizations"] == {"in": ["VV VH"]}
        assert body["query"]["start_datetime"]["gte"].startswith("2025-01-01T00:00:00")
        assert body["query"]["start_datetime"]["lte"].startswith("2025-01-02T00:00:00")
        assert body["intersects"]["type"] == "MultiPolygon"
        assert body["sortBy"] == [{"field": "start_datetime", "direction": "asc"}]


def test_geodes_empty_search_is_valid(geodes):
    pages, calls = geodes
    pages[1] = (200, {"features": [], "context": {"matched": 0}})
    assert search() == (0, [])
    assert len(calls) == 1


def test_geodes_failure_after_first_page_does_not_return_partial_results(geodes):
    pages, _ = geodes
    pages[1] = (
        200,
        {"features": [feature(i) for i in range(25)], "context": {"matched": 26}},
    )
    pages[2] = (503, {"error": "unavailable"})
    # EODAG 2.5 surfaces a failed HTTP page as a None-results TypeError.
    # The batch must fail rather than accepting the already-read first page.
    with pytest.raises(TypeError, match="NoneType"):
        search()
