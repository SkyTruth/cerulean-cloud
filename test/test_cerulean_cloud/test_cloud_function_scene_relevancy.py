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
    with open("test/test_cerulean_cloud/fixtures/valid_sns_event.json") as src:
        event = json.load(src)

    assert json.loads(event["Records"][0]["Sns"]["Message"])["version"] == 3

    res = handle_notification(event, ocean_poly=ocean_poly)
    assert len(res) == 1


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


def test_handle_notification_filters():
    ocean_poly = load_ocean_poly(
        "cerulean_cloud/cloud_function_scene_relevancy/OceanGeoJSON_lowres.geojson"
    )

    invalids = [
        "invalid_sns_event_pol.json",
        "invalid_sns_event_res.json",
        "invalid_sns_event_mode.json",
        "invalid_sns_event_loc.json",
    ]
    for invalid in invalids:
        with open(f"test/test_cerulean_cloud/fixtures/{invalid}") as src:
            invalid_event = json.load(src)
        invalid_res = handle_notification(invalid_event, ocean_poly=ocean_poly)
        assert len(invalid_res) == 0, f"Expected no results for invalid scene {invalid}"
