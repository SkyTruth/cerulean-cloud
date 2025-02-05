import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest
import rasterio
from rasterio.plot import reshape_as_image

import cerulean_cloud.titiler_client
from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceResult,
    InferenceResultStack,
)
from cerulean_cloud.cloud_run_orchestrator.clients import (
    CloudRunInferenceClient,
    get_dist_array,
    get_scene_date_month,
    get_ship_density,
    handle_aux_datasets,
)
from cerulean_cloud.tiling import TMS
from cerulean_cloud.titiler_client import TitilerClient


def get_mock_layer(short_name, source_url=""):
    mock_layer = Mock()
    mock_layer.short_name = short_name
    mock_layer.source_url = source_url
    return mock_layer


async def mock_get_base_tile(self, scene_id, tile, scale, rescale):
    with rasterio.open("test/test_cerulean_cloud/fixtures/example_tile.png") as src:
        img_array = reshape_as_image(src.read())

    return img_array


async def mock_get_offset_tile(
    self, scene_id, minx, miny, maxx, maxy, width, height, rescale
):
    with rasterio.open("test/test_cerulean_cloud/fixtures/example_tile.png") as src:
        img_array = reshape_as_image(src.read())

    return img_array


@pytest.fixture
def fixture_cloud_inference_tile(httpx_mock):
    titiler_client = TitilerClient(url="some_url")
    with open(
        "test/test_cerulean_cloud/fixtures/MLXF_ais__sq_07a7fea65ceb3429c1ac249f4187f414_9c69e5b4361b6bc412a41f85cdec01ee.zip",
        "rb",
    ) as src:
        httpx_mock.add_response(content=src.read())

    return CloudRunInferenceClient(
        url="http://inferenceurl.com",
        titiler_client=titiler_client,
        scene_id="S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5",
        tileset_envelope_bounds=[
            55.69982872351191,
            24.566447533809654,
            58.53597315567021,
            26.496758065384803,
        ],
        image_hw_pixels=(4181, 6458),
        layers=[
            get_mock_layer("VV"),
            get_mock_layer("VESSEL"),
            get_mock_layer(
                "INFRA", "test/test_cerulean_cloud/fixtures/test_cogeo.tiff"
            ),
        ],
        scale=1,
    )


@patch.object(
    cerulean_cloud.titiler_client.TitilerClient, "get_base_tile", mock_get_base_tile
)
@pytest.mark.skip
@pytest.mark.asyncio
async def test_get_tile_inference(fixture_cloud_inference_tile, httpx_mock):
    payload = {
        "inference_input": InferenceResultStack(
            stack=[InferenceResult(tile_probs_b64="", bounds=[1, 2, 3, 4])]
        ).dict(),
        "model_dict": {
            "model_type": "MASKRCNN",
            "thresholds": {
                "poly_nms_thresh": 0.4,
                "pixel_nms_thresh": 0.4,
                "bbox_score_thresh": 0.2,
                "poly_score_thresh": 0.2,
                "pixel_score_thresh": 0.2,
                "groundtruth_dice_thresh": 0.0,
            },
        },
    }
    httpx_mock.add_response(
        method="POST",
        url=fixture_cloud_inference_tile.url + "/predict",
        json=payload,
    )
    tasks = [
        fixture_cloud_inference_tile.get_tile_inference(
            httpx_mock, tile=TMS._tile(0, 0, 0)
        ),
        fixture_cloud_inference_tile.get_tile_inference(
            httpx_mock, tile=TMS._tile(0, 0, 0)
        ),
        fixture_cloud_inference_tile.get_tile_inference(
            httpx_mock,
            bounds=list(TMS.bounds(TMS._tile(0, 0, 0))),
        ),
        fixture_cloud_inference_tile.get_tile_inference(
            httpx_mock,
            bounds=list(TMS.bounds(TMS._tile(0, 0, 0))),
        ),
    ]
    res = await asyncio.gather(*tasks, return_exceptions=True)
    print(res)
    assert res[0].stack[0].classes == ""
    assert res[1].stack[0].classes == ""
    assert res[2].stack[0].classes == ""
    assert res[3].stack[0].classes == ""


def test_get_ship_density(httpx_mock):
    with open(
        "test/test_cerulean_cloud/fixtures/MLXF_ais__sq_07a7fea65ceb3429c1ac249f4187f414_9c69e5b4361b6bc412a41f85cdec01ee.zip",
        "rb",
    ) as src:
        httpx_mock.add_response(content=src.read())
    arr = get_ship_density(
        bounds=(55.698181, 24.565813, 58.540211, 26.494711), img_shape=(4181, 6458)
    )
    assert arr.shape == (4181, 6458)
    assert arr.dtype == np.dtype(np.uint8)
    assert np.max(arr) == 255
    assert np.min(arr) == 0


def test_get_dist_array():
    arr = get_dist_array(
        bounds=(55.698181, 24.565813, 58.540211, 26.494711),
        img_shape=(4181, 6458),
        raster_ds="test/test_cerulean_cloud/fixtures/test_cogeo.tiff",
    )
    assert arr.shape == (4181, 6458)
    assert arr.dtype == np.dtype(np.uint8)
    assert np.max(arr) == 255
    assert np.min(arr) == 0

    # test all 0 input


def test_handle_aux_datasets(httpx_mock):
    ar_mem_file = handle_aux_datasets(
        [
            get_mock_layer("VV"),
            get_mock_layer(
                "INFRA", "test/test_cerulean_cloud/fixtures/test_cogeo.tiff"
            ),
            get_mock_layer(
                "INFRA", "test/test_cerulean_cloud/fixtures/test_cogeo.tiff"
            ),
        ],
        scene_id="S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5",
        tileset_envelope_bounds=[
            55.69982872351191,
            24.566447533809654,
            58.53597315567021,
            26.496758065384803,
        ],
        image_hw_pixels=(4181, 6458),
    )
    with ar_mem_file.open() as src:
        ar = src.read()
        assert ar.shape == (2, 4181, 6458)

    with open(
        "test/test_cerulean_cloud/fixtures/MLXF_ais__sq_07a7fea65ceb3429c1ac249f4187f414_9c69e5b4361b6bc412a41f85cdec01ee.zip",
        "rb",
    ) as src:
        httpx_mock.add_response(content=src.read())

    ar_mem_file = handle_aux_datasets(
        [
            get_mock_layer("VV"),
            get_mock_layer("VESSEL"),
            get_mock_layer(
                "INFRA", "test/test_cerulean_cloud/fixtures/test_cogeo.tiff"
            ),
        ],
        scene_id="S1A_IW_GRDH_1SDV_20200802T141646_20200802T141711_033729_03E8C7_E4F5",
        tileset_envelope_bounds=[
            55.69982872351191,
            24.566447533809654,
            58.53597315567021,
            26.496758065384803,
        ],
        image_hw_pixels=(4181, 6458),
    )
    with ar_mem_file.open() as src:
        ar = src.read()
        assert ar.shape == (2, 4181, 6458)


def test_get_ship_density_recent_live():
    print(
        get_scene_date_month(
            "S1A_IW_GRDH_1SDV_20220808T083805_20220808T083834_044458_054E25_B895"
        )
    )
    # If the month is not long enough, the API returns an empty geotiff
    date_time_obj = datetime.now().replace(day=1, hour=0, minute=0, second=0)
    current_month_time = date_time_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(current_month_time)
    ar = get_ship_density(
        [
            145.37495713519354,
            -10.611489825106801,
            147.96902460466112,
            -8.47969827741699,
        ],
        (4096, 4608),
        current_month_time,
    )
    assert ar.shape == (4096, 4608)
