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

    # Valid scene/polygon (should produce a non-empty result if it normally does)
    valid_scene_id = (
        "S1A_IW_GRDH_1SDV_20240730T001806_20240730T001831_054983_06B2B8_3CE1"
    )
    valid_event = {
        "Records": [
            {
                "Sns": {
                    "Message": json.dumps(
                        {
                            "id": valid_scene_id,
                            "footprint": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [
                                            [0.0, 0.0],
                                            [0.0, 1.0],
                                            [1.0, 1.0],
                                            [1.0, 0.0],
                                            [0.0, 0.0],
                                        ]
                                    ]
                                ],
                            },
                        }
                    )
                }
            }
        ]
    }
    valid_res = handle_notification(valid_event, ocean_poly=ocean_poly)
    # Depending on logic, this should be non-empty if it meets the criteria for relevancy
    # If the function currently filters on ocean/intersection, adjust the polygon or expectations accordingly.
    # For now we assume it should return something. If it doesn't, adjust your logic or polygon.
    assert len(valid_res) > 0, "Expected at least one result for a valid scene/polygon"

    # Invalid scene IDs (should all be rejected)
    invalid_scene_ids = [
        "S1A_S3_GRDH_1SDV_20",
        "S1A_IW_GRDM_1SDV_20",
        "S1A_IW_GRDH_1SDH_20",
    ]

    for scene_id in invalid_scene_ids:
        invalid_event = {
            "Records": [
                {
                    "Sns": {
                        "Message": json.dumps(
                            {
                                "id": scene_id,
                                "footprint": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [
                                            [
                                                [0.0, 0.0],
                                                [0.0, 1.0],
                                                [1.0, 1.0],
                                                [1.0, 0.0],
                                                [0.0, 0.0],
                                            ]
                                        ]
                                    ],
                                },
                            }
                        )
                    }
                }
            ]
        }
        invalid_res = handle_notification(invalid_event, ocean_poly=ocean_poly)
        assert (
            len(invalid_res) == 0
        ), f"Expected no results for invalid scene_id {scene_id}"

    # Invalid polygon (should be rejected)
    invalid_polygon_event = {
        "Records": [
            {
                "Sns": {
                    "Message": json.dumps(
                        {
                            "id": valid_scene_id,
                            "footprint": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [
                                            [10.0, 10.0],
                                            [10.0, 11.0],
                                            [11.0, 11.0],
                                            [11.0, 10.0],
                                            [10.0, 10.0],
                                        ]
                                    ]
                                ],
                            },
                        }
                    )
                }
            }
        ]
    }
    invalid_poly_res = handle_notification(invalid_polygon_event, ocean_poly=ocean_poly)
    assert len(invalid_poly_res) == 0, "Expected no results for invalid polygon"
