import json

from dateutil.parser import parse

from cerulean_cloud.cloud_function_scene_relevancy.main import (
    handle_notification,
    load_ocean_poly,
    make_cloud_function_logs_url,
)


def test_handle_notification():
    ocean_poly = load_ocean_poly(
        "cerulean_cloud/cloud_function_scene_relevancy/OceanGeoJSON_lowres.geojson"
    )
    with open("test/test_cerulean_cloud/fixtures/event.json") as src:
        event = json.load(src)

    res = handle_notification(event, ocean_poly=ocean_poly)
    assert len(res) == 0


def test_make_logs():
    url = make_cloud_function_logs_url(
        "cerulean-cloud-test-cloud-function-scene-relevancy-4f8e4cf",
        parse("2022-07-15T11:13:12.183229778Z"),
        "cerulean-338116",
    )
    assert url == (
        "https://console.cloud.google.com/logs/query;"
        "query=resource.type%20%3D%20%22cloud_function%22%20resource.labels.function_name%20%3D%20%22cerulean-cloud-test-cloud-function-scene-relevancy-4f8e4cf%22;"
        "timeRange=2022-07-15T11:13:12.183229Z%2F2022-07-15T11:15:12.183229Z;"
        "cursorTimestamp=2022-07-15T11:13:12.183229Z?project=cerulean-338116"
    )
