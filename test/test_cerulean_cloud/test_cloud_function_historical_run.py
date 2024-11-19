import json

import pytest
from dateutil.parser import parse

from cerulean_cloud.cloud_function_historical_run.main import (
    handle_search,
    load_ocean_poly,
    make_cloud_function_logs_url,
)


@pytest.mark.skip
def test_handle_search():
    ocean_poly = load_ocean_poly(
        "cerulean_cloud/cloud_function_scene_relevancy/OceanGeoJSON_lowres.geojson"
    )
    with open("test/test_cerulean_cloud/fixtures/whole_world.geojson") as src:
        geom_whole_world = json.load(src)

    request = {"start": "2022-01-01", "end": "2022-01-02", "geometry": geom_whole_world}

    total_scenes, filtered_scenes = handle_search(request, ocean_poly=ocean_poly)
    assert total_scenes == 479  # total scenes
    assert len(filtered_scenes) == 271  # scenes in water

    with open(
        "test/test_cerulean_cloud/fixtures/whole_world_search_geom.geojson"
    ) as src:
        geom = json.load(src)

    request = {"start": "2022-08-07", "end": "2022-08-08", "geometry": geom}

    total_scenes, filtered_scenes = handle_search(request, ocean_poly=ocean_poly)
    assert total_scenes == 305  # total scenes
    assert len(filtered_scenes) == 163  # scenes in water

    # Before S1B is gone
    request = {"start": "2018-01-01", "end": "2018-01-02", "geometry": geom_whole_world}

    total_scenes, filtered_scenes = handle_search(request, ocean_poly=ocean_poly)
    assert total_scenes == 800  # total scenes
    assert len(filtered_scenes) == 431  # scenes in water


def test_make_logs():
    url = make_cloud_function_logs_url(
        "cerulean-cloud-test-cf-scene-relevancy-4f8e4cf",
        parse("2022-07-15T11:13:12.183229778Z"),
        "cerulean-338116",
    )
    assert url == (
        "https://console.cloud.google.com/logs/query;"
        "query=resource.type%20%3D%20%22cloud_function%22%20resource.labels.function_name%20%3D%20%22cerulean-cloud-test-cf-scene-relevancy-4f8e4cf%22;"
        "timeRange=2022-07-15T11:13:12.183229Z%2F2022-07-15T11:15:12.183229Z;"
        "cursorTimestamp=2022-07-15T11:13:12.183229Z?project=cerulean-338116"
    )
