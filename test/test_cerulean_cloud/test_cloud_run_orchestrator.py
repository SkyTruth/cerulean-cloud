import json
import os
import sys
from base64 import b64decode
from unittest.mock import patch

import httpx
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
    from_bounds_get_offset_bounds,
    from_tiles_get_offset_shape,
)
from cerulean_cloud.cloud_run_orchestrator.schema import OrchestratorInput
from cerulean_cloud.tiling import TMS, from_base_tiles_create_offset_tiles
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


async def mock_get_base_tile_inference(self, tile, rescale):
    with open("test/test_cerulean_cloud/fixtures/enc_classes_512_512.txt") as src:
        enc_classes = src.read()

    with open("test/test_cerulean_cloud/fixtures/enc_confidence_512_512.txt") as src:
        enc_confidence = src.read()
    return InferenceResult(
        classes=enc_classes, confidence=enc_confidence, bounds=list(TMS.bounds(tile))
    )


async def mock_get_offset_tile_inference(self, bounds, rescale):
    with open("test/test_cerulean_cloud/fixtures/enc_classes_512_512.txt") as src:
        enc_classes = src.read()

    with open("test/test_cerulean_cloud/fixtures/enc_confidence_512_512.txt") as src:
        enc_confidence = src.read()
    return InferenceResult(
        classes=enc_classes, confidence=enc_confidence, bounds=bounds
    )


async def mock_get_bounds(*args):
    return [32.989094, 43.338009, 36.540836, 45.235191]


async def mock_get_statistics(*args, **kwargs):
    return {"min": 1, "max": 10}


@patch.object(
    cerulean_cloud.titiler_client.TitilerClient, "get_statistics", mock_get_statistics
)
@patch.object(
    cerulean_cloud.titiler_client.TitilerClient, "get_bounds", mock_get_bounds
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
@pytest.mark.asyncio
async def test_orchestrator(httpx_mock, fixture_titiler_client, monkeypatch):
    monkeypatch.setenv(
        "AUX_INFRA_DISTANCE", "test/test_cerulean_cloud/fixtures/test_cogeo.tiff"
    )
    payload = OrchestratorInput(sceneid=S1_ID)
    with open(
        "test/test_cerulean_cloud/fixtures/MLXF_ais__sq_07a7fea65ceb3429c1ac249f4187f414_9c69e5b4361b6bc412a41f85cdec01ee.zip",
        "rb",
    ) as src:
        httpx_mock.add_response(content=src.read())

    res = await _orchestrate(payload, TMS, fixture_titiler_client)
    # max payload is 32 MB
    assert sys.getsizeof(res.json()) / 1000000 < 32
    assert res.ntiles == 66
    assert res.noffsettiles == 84
    assert res.base_inference
    assert res.offset_inference
    with MemoryFile(b64decode(res.base_inference)) as memfile:
        with memfile.open() as dataset:
            np_img = dataset.read()
            assert np_img.shape == (2, 3072, 5632)
    with MemoryFile(b64decode(res.offset_inference)) as memfile:
        with memfile.open() as dataset:
            np_img = dataset.read()
            assert np_img.shape == (2, 3584, 6144)


def test_from_tiles_get_offset_shape():
    bounds = [32.989094, 43.338009, 36.540836, 45.235191]
    base_tiles = list(TMS.tiles(*bounds, [9], truncate=False))
    offset_image_shape = from_tiles_get_offset_shape(base_tiles, scale=2)
    assert offset_image_shape == (3584, 6144)


def test_from_bounds_get_offset_bounds():
    bounds = [32.989094, 43.338009, 36.540836, 45.235191]
    base_tiles = list(TMS.tiles(*bounds, [9], truncate=False))
    offset_tiles_bounds = from_base_tiles_create_offset_tiles(base_tiles)
    offset_bounds = from_bounds_get_offset_bounds(offset_tiles_bounds)
    assert offset_bounds == pytest.approx(
        [
            32.5195312499997442,
            43.0664062500000568,
            36.7382812499997442,
            45.5273437500000568,
        ]
    )


def custom_response(url, data, timeout):
    data = json.loads(data)
    r = InferenceResult(
        classes=data["image"], confidence=data["image"], bounds=data["bounds"]
    )
    return httpx.Response(status_code=200, json=r.dict())


@pytest.mark.skip
@patch.object(httpx, "post", custom_response)
@patch.dict(
    os.environ,
    {
        "AUX_INFRA_DISTANCE": "https://storage.googleapis.com/ceruleanml/aux_datasets/infra_locations_01_cogeo.tiff",
        "INFERENCE_URL": "http://someurl.test",
    },
)
def test_orchestrator_live():
    payload = OrchestratorInput(sceneid=S1_ID)
    titiler_client = TitilerClient(
        "https://0xshe4bmk8.execute-api.eu-central-1.amazonaws.com/"
    )

    res = _orchestrate(payload, TMS, titiler_client)
    assert res.ntiles == 66
    assert res.noffsettiles == 84
    assert res.base_inference
    assert res.offset_inference

    with MemoryFile(b64decode(res.base_inference)) as memfile:
        with memfile.open() as dataset:
            np_img = dataset.read()
            with rasterio.open(
                "scratch/test_out_base.tiff", **dataset.profile, mode="w"
            ) as dst:
                dst.write(np_img)
                # 6 since class is 3 and conf is 3
            assert np_img.shape == (6, 3072, 5632)

    with MemoryFile(b64decode(res.offset_inference)) as memfile:
        with memfile.open() as dataset:
            np_img = dataset.read()
            with rasterio.open(
                "scratch/test_out_offset.tiff", **dataset.profile, mode="w"
            ) as dst:
                dst.write(np_img)
            assert np_img.shape == (6, 3584, 6144)
