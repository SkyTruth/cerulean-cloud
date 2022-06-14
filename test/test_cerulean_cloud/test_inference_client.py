from unittest.mock import patch

import pytest
import rasterio
from rasterio.plot import reshape_as_image

import cerulean_cloud.titiler_client
from cerulean_cloud.cloud_run_orchestrator.clients import CloudRunInferenceClient
from cerulean_cloud.tiling import TMS
from cerulean_cloud.titiler_client import TitilerClient


def mock_get_base_tile(self, sceneid, tile, scale, rescale):

    with rasterio.open("test/test_cerulean_cloud/fixtures/example_tile.png") as src:
        img_array = reshape_as_image(src.read())

    return img_array


def mock_get_offset_tile(self, sceneid, minx, miny, maxx, maxy, rescale):
    with rasterio.open("test/test_cerulean_cloud/fixtures/example_tile.png") as src:
        img_array = reshape_as_image(src.read())

    return img_array


@pytest.fixture
def fixture_cloud_inference_tile():
    titiler_client = TitilerClient(url="some_url")
    return CloudRunInferenceClient(
        url="http://inferenceurl.com/", titiler_client=titiler_client
    )


@patch.object(
    cerulean_cloud.titiler_client.TitilerClient, "get_base_tile", mock_get_base_tile
)
def test_get_base_tile_inference(fixture_cloud_inference_tile, httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url=fixture_cloud_inference_tile.url + "predict/",
        content="",
    )

    res = fixture_cloud_inference_tile.get_base_tile_inference(
        sceneid="ABC", tile=TMS._tile(0, 0, 0), rescale=(0, 100)
    )
    assert res.content == b""


@patch.object(
    cerulean_cloud.titiler_client.TitilerClient, "get_offset_tile", mock_get_offset_tile
)
def test_get_offset_tile_inference(fixture_cloud_inference_tile, httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url=fixture_cloud_inference_tile.url + "predict/",
        content="",
    )

    res = fixture_cloud_inference_tile.get_offset_tile_inference(
        sceneid="ABC", bounds=[1, 2, 3, 4], rescale=(0, 100)
    )
    assert res.content == b""
