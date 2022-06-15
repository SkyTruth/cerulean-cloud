from base64 import b64decode
from test.test_cerulean_cloud.test_inference_client import (
    mock_get_base_tile,
    mock_get_offset_tile,
)
from unittest.mock import patch

import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_image

import cerulean_cloud
from cerulean_cloud.cloud_run_offset_tiles.schema import InferenceResult
from cerulean_cloud.cloud_run_orchestrator.clients import CloudRunInferenceClient
from cerulean_cloud.cloud_run_orchestrator.handler import (
    _orchestrate,
    b64_image_to_array,
)
from cerulean_cloud.cloud_run_orchestrator.schema import OrchestratorInput
from cerulean_cloud.tiling import TMS
from cerulean_cloud.titiler_client import TitilerClient

S1_ID = "S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF"


def get_base_tile_3band(*args, **kwargs):
    with rasterio.open(
        "test/test_cerulean_cloud/fixtures/tile_512_512_3band.png"
    ) as src:
        ar = reshape_as_image(src.read())
    return ar


@pytest.mark.skip
@patch.object(
    cerulean_cloud.titiler_client.TitilerClient, "get_base_tile", get_base_tile_3band
)
def test_create_fixture_inference(
    url="https://0xshe4bmk8.execute-api.eu-central-1.amazonaws.com/",
    inference_url="https://cerulean-cloud-staging-cloud-run-offset-tiles-49b-5qkjkyomta-ew.a.run.app",
):
    titiler_client = TitilerClient(url=url)
    S1_ID = "S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF"

    tiles = list(TMS.tiles(*titiler_client.get_bounds(S1_ID), [10], truncate=False))
    tile = tiles[20]

    inference_client = CloudRunInferenceClient(
        url=inference_url, titiler_client=titiler_client
    )

    res = inference_client.get_base_tile_inference(S1_ID, tile)
    print(res)

    array = b64_image_to_array(res.classes)

    with rasterio.open(
        "test/test_cerulean_cloud/fixtures/classes_512_512.png",
        "w",
        driver="PNG",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=array.dtype,
        compress="deflate",
    ) as dst:
        dst.write(array)

    array = b64_image_to_array(res.confidence)

    with rasterio.open(
        "test/test_cerulean_cloud/fixtures/confidence_512_512.png",
        "w",
        driver="PNG",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=array.dtype,
        compress="deflate",
    ) as dst:
        dst.write(array)

    with open("test/test_cerulean_cloud/fixtures/enc_classes_512_512.txt", "w") as dst:
        dst.write(res.classes)

    with open(
        "test/test_cerulean_cloud/fixtures/enc_confidence_512_512.txt", "w"
    ) as dst:
        dst.write(res.confidence)


@pytest.fixture
def fixture_titiler_client():
    return TitilerClient("some_url")


@pytest.fixture
def fixture_cloud_inference(fixture_titiler_client):
    return CloudRunInferenceClient(
        url="some_url", titiler_client=fixture_titiler_client
    )


def mock_get_base_tile_inference(self, sceneid, tile, rescale):
    with open("test/test_cerulean_cloud/fixtures/enc_classes_512_512.txt") as src:
        enc_classes = src.read()

    with open("test/test_cerulean_cloud/fixtures/enc_confidence_512_512.txt") as src:
        enc_confidence = src.read()
    return InferenceResult(
        classes=enc_classes, confidence=enc_confidence, bounds=list(TMS.bounds(tile))
    )


def mock_get_offset_tile_inference(self, sceneid, bounds, rescale):
    with open("test/test_cerulean_cloud/fixtures/enc_classes_512_512.txt") as src:
        enc_classes = src.read()

    with open("test/test_cerulean_cloud/fixtures/enc_confidence_512_512.txt") as src:
        enc_confidence = src.read()
    return InferenceResult(
        classes=enc_classes, confidence=enc_confidence, bounds=bounds
    )


@patch.object(
    cerulean_cloud.titiler_client.TitilerClient, "get_base_tile", mock_get_base_tile
)
@patch.object(
    cerulean_cloud.titiler_client.TitilerClient, "get_offset_tile", mock_get_offset_tile
)
@patch.object(
    cerulean_cloud.titiler_client.TitilerClient,
    "get_bounds",
    lambda *args: [32.989094, 43.338009, 36.540836, 45.235191],
)
@patch.object(
    cerulean_cloud.titiler_client.TitilerClient,
    "get_statistics",
    lambda *args: {"vv": {"min": 1, "max": 10}},
)
@patch.object(
    cerulean_cloud.cloud_run_orchestrator.clients.CloudRunInferenceClient,
    "get_base_tile_inference",
    mock_get_base_tile_inference,
)
@patch.object(
    cerulean_cloud.cloud_run_orchestrator.clients.CloudRunInferenceClient,
    "get_offset_tile_inference",
    mock_get_offset_tile_inference,
)
def test_orchestrator(httpx_mock, fixture_titiler_client, fixture_cloud_inference):
    payload = OrchestratorInput(sceneid=S1_ID)
    res = _orchestrate(payload, TMS, fixture_titiler_client, fixture_cloud_inference)
    assert res.ntiles == 252
    assert res.noffsettiles == 286
    assert res.base_inference
    assert res.offset_inference

    with MemoryFile(b64decode(res.base_inference)) as memfile:
        with memfile.open() as dataset:
            np_img = dataset.read()
            assert np_img.shape == (2, 6144, 10752)

    with MemoryFile(b64decode(res.offset_inference)) as memfile:
        with memfile.open() as dataset:
            np_img = dataset.read()
            assert np_img.shape == (2, 6656, 11264)
