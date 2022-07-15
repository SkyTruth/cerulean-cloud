import json

from cerulean_cloud.cloud_function_scene_relevancy.main import (
    handle_notification,
    load_ocean_poly,
)


def test_handle_notification():
    ocean_poly = load_ocean_poly(
        "cerulean_cloud/cloud_function_scene_relevancy/OceanGeoJSON_lowres.geojson"
    )
    with open("test/test_cerulean_cloud/fixtures/event.json") as src:
        event = json.load(src)

    res = handle_notification(event, ocean_poly=ocean_poly)
    assert len(res) == 0
