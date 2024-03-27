import json
import os
import sys
from base64 import b64decode
from datetime import datetime
from unittest.mock import patch

import geojson
import git
import httpx
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.plot import reshape_as_image
from shapely.geometry import box

import cerulean_cloud
from cerulean_cloud.cloud_run_offset_tiles.schema import (
    InferenceResult,
    InferenceResultStack,
)
from cerulean_cloud.cloud_run_orchestrator.clients import CloudRunInferenceClient
from cerulean_cloud.cloud_run_orchestrator.handler import (
    _orchestrate,
    b64_image_to_array,
    flatten_feature_list,
    group_bounds_from_list_of_bounds,
    is_tile_over_water,
    make_cloud_log_url,
    offset_group_shape_from_base_tiles,
)
from cerulean_cloud.cloud_run_orchestrator.merging import merge_inferences
from cerulean_cloud.cloud_run_orchestrator.schema import OrchestratorInput
from cerulean_cloud.roda_sentinelhub_client import RodaSentinelHubClient
from cerulean_cloud.tiling import TMS, offset_bounds_from_base_tiles
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

    res = inference_client.get_tile_inference(None, tile=tile)
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
def fixture_roda_sentinelhub_client():
    return RodaSentinelHubClient(url="some_url")


async def mock_get_tile_inference(
    http_client, tile=None, bounds=None, rescale=(0, 255)
):
    with open("test/test_cerulean_cloud/fixtures/enc_classes_512_512.txt") as src:
        enc_classes = src.read()

    with open("test/test_cerulean_cloud/fixtures/enc_confidence_512_512.txt") as src:
        enc_confidence = src.read()
    return InferenceResultStack(
        stack=[
            InferenceResult(
                classes=enc_classes,
                confidence=enc_confidence,
                bounds=list(TMS.bounds(tile)) if tile else bounds,
            )
        ]
    )


async def mock_get_bounds(*args):
    return [32.989094, 43.338009, 36.540836, 45.235191]


async def mock_get_statistics(*args, **kwargs):
    return {"min": 1, "max": 10}


async def mock_get_product_info(*args, **kwargs):
    return json.load(open("test/test_cerulean_cloud/fixtures/productInfo.json"))


@pytest.mark.skip
@patch.object(
    cerulean_cloud.titiler_client.TitilerClient, "get_statistics", mock_get_statistics
)
@patch.object(
    cerulean_cloud.titiler_client.TitilerClient, "get_bounds", mock_get_bounds
)
@patch.object(
    cerulean_cloud.roda_sentinelhub_client.RodaSentinelHubClient,
    "get_product_info",
    mock_get_product_info,
)
@patch.object(
    cerulean_cloud.cloud_run_orchestrator.clients.CloudRunInferenceClient,
    "get_tile_inference",
    mock_get_tile_inference,
)
@pytest.mark.asyncio
async def test_orchestrator(
    httpx_mock, fixture_titiler_client, fixture_roda_sentinelhub_client, monkeypatch
):
    payload = OrchestratorInput(sceneid=S1_ID)
    with open(
        "test/test_cerulean_cloud/fixtures/MLXF_ais__sq_07a7fea65ceb3429c1ac249f4187f414_9c69e5b4361b6bc412a41f85cdec01ee.zip",
        "rb",
    ) as src:
        httpx_mock.add_response(content=src.read())

    res = await _orchestrate(
        payload, TMS, fixture_titiler_client, fixture_roda_sentinelhub_client, None
    )
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

    assert res.classification
    assert len(res.classification["features"]) > 1


def test_from_tiles_get_offset_shape():
    bounds = [32.989094, 43.338009, 36.540836, 45.235191]
    base_tiles = list(TMS.tiles(*bounds, [9], truncate=False))
    offset_image_shape = offset_group_shape_from_base_tiles(base_tiles, scale=2)
    assert offset_image_shape == (3584, 6144)


def test_from_bounds_get_offset_bounds():
    bounds = [32.989094, 43.338009, 36.540836, 45.235191]
    base_tiles = list(TMS.tiles(*bounds, [9], truncate=False))
    offset_tiles_bounds = offset_bounds_from_base_tiles(base_tiles)
    offset_bounds = group_bounds_from_list_of_bounds(offset_tiles_bounds)
    assert offset_bounds == pytest.approx(
        [
            32.5195312499997442,
            43.0664062500000568,
            36.7382812499997442,
            45.5273437500000568,
        ]
    )


def test_is_tile_over_water():
    # land and water
    bounds = [32.989094, 43.338009, 36.540836, 45.235191]
    base_tiles = list(TMS.tiles(*bounds, [9], truncate=False))
    offset_tiles_bounds = offset_bounds_from_base_tiles(base_tiles)
    assert len(base_tiles) == 66
    assert len(offset_tiles_bounds) == 84

    base_tiles_over_water = [t for t in base_tiles if is_tile_over_water(TMS.bounds(t))]
    assert len(base_tiles_over_water) == 59

    offset_bounds_over_water = [b for b in offset_tiles_bounds if is_tile_over_water(b)]
    assert len(offset_bounds_over_water) == 75

    # Fully over water
    bounds = [-13.461797, 37.952782, -10.128826, 39.863739]
    base_tiles = list(TMS.tiles(*bounds, [9], truncate=False))
    offset_tiles_bounds = offset_bounds_from_base_tiles(base_tiles)
    assert len(base_tiles) == 77
    assert len(offset_tiles_bounds) == 96

    base_tiles_over_water = [t for t in base_tiles if is_tile_over_water(TMS.bounds(t))]
    assert len(base_tiles_over_water) == 77

    offset_bounds_over_water = [b for b in offset_tiles_bounds if is_tile_over_water(b)]
    assert len(offset_bounds_over_water) == 96

    # Fully over land
    bounds = [
        -74.82934700262595,
        -2.2906399527608623,
        -72.28342220242573,
        -0.18418333168440187,
    ]
    base_tiles = list(TMS.tiles(*bounds, [9], truncate=False))
    offset_tiles_bounds = offset_bounds_from_base_tiles(base_tiles)
    assert len(base_tiles) == 56
    assert len(offset_tiles_bounds) == 72

    base_tiles_over_water = [t for t in base_tiles if is_tile_over_water(TMS.bounds(t))]
    assert len(base_tiles_over_water) == 0

    offset_bounds_over_water = [b for b in offset_tiles_bounds if is_tile_over_water(b)]
    assert len(offset_bounds_over_water) == 0


def custom_response(url, data, timeout):
    data = json.loads(data)
    r = InferenceResult(
        classes=data["image"], confidence=data["image"], bounds=data["bounds"]
    )
    return httpx.Response(status_code=200, json=r.dict())


@pytest.mark.skip
@pytest.mark.asyncio
@patch.dict(
    os.environ,
    {
        "AUX_INFRA_DISTANCE": "https://storage.googleapis.com/ceruleanml/aux_datasets/infra_locations_01_cogeo.tiff",
        "INFERENCE_URL": "http://someurl.test",
    },
)
@patch.object(
    cerulean_cloud.cloud_run_orchestrator.clients.CloudRunInferenceClient,
    "get_tile_inference",
    mock_get_tile_inference,
)
async def test_orchestrator_live():
    payload = OrchestratorInput(
        sceneid=S1_ID
    )  # "S1A_IW_GRDH_1SDV_20201121T225759_20201121T225828_035353_04216C_62EA")
    titiler_client = TitilerClient(
        "https://0xshe4bmk8.execute-api.eu-central-1.amazonaws.com/"
    )
    roda_sentinelhub_client = RodaSentinelHubClient()
    engine = cerulean_cloud.database_client.get_engine()

    res = await _orchestrate(
        payload, TMS, titiler_client, roda_sentinelhub_client, engine
    )
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
            assert np_img.shape == (2, 3072, 5632)

    with MemoryFile(b64decode(res.offset_inference)) as memfile:
        with memfile.open() as dataset:
            np_img = dataset.read()
            with rasterio.open(
                "scratch/test_out_offset.tiff", **dataset.profile, mode="w"
            ) as dst:
                dst.write(np_img)
            assert np_img.shape == (2, 3584, 6144)


def test_make_cloud_log_url():
    start_time = datetime.strptime("2022-07-06 13:39:30.56396", "%Y-%m-%d %H:%M:%S.%f")
    res = make_cloud_log_url(
        "cerulean-cloud-test-cloud-run-orch", start_time, "cerulean-338116"
    )
    assert res == (
        "https://console.cloud.google.com/logs/query;query="
        "resource.type%20%3D%20%22cloud_run_revision%22%20resource.labels.service_name%20%3D%20%22cerulean-cloud-test-cloud-run-orch%22;"
        "timeRange=2022-07-06T13:39:30.563960Z%2F2022-07-06T13:41:30.563960Z;"
        "cursorTimestamp=2022-07-06T13:39:30.563960Z?"
        "project=cerulean-338116"
    )


def test_flatten_result():
    result_stack = InferenceResultStack(
        stack=[
            InferenceResult(
                features=[geojson.Feature(geometry=box(1, 2, 3, 4))],
                bounds=[1, 2, 3, 4],
            ),
            InferenceResult(
                features=[geojson.Feature(geometry=box(1, 2, 3, 4))],
                bounds=[1, 2, 3, 4],
            ),
        ]
    )
    res = [result_stack, result_stack]

    flat_list = flatten_feature_list(res)

    assert len(flat_list) == 4
    assert isinstance(flat_list[0], geojson.Feature)

    result_stack = InferenceResultStack(
        stack=[
            InferenceResult(
                features=[geojson.Feature(geometry=box(1, 2, 3, 4))],
                bounds=[1, 2, 3, 4],
            ),
            InferenceResult(
                features=[geojson.Feature(geometry=box(1, 2, 3, 4))],
                bounds=[1, 2, 3, 4],
            ),
        ]
    )
    res = [
        result_stack,
        InferenceResultStack(
            stack=[
                InferenceResult(
                    features=[
                        geojson.Feature(geometry=box(1, 2, 3, 4)),
                        geojson.Feature(geometry=box(1, 2, 3, 4)),
                        geojson.Feature(geometry=box(1, 2, 3, 4)),
                    ],
                    bounds=[1, 2, 3, 4],
                ),
                InferenceResult(
                    features=[],
                    bounds=[1, 2, 3, 4],
                ),
            ]
        ),
    ]

    flat_list = flatten_feature_list(res)

    assert len(flat_list) == 5
    assert isinstance(flat_list[0], geojson.Feature)

    res = [result_stack, InferenceResultStack(stack=[])]

    flat_list = flatten_feature_list(res)

    assert len(flat_list) == 2
    assert isinstance(flat_list[0], geojson.Feature)


def test_func_merge_inferences():
    with open("test/test_cerulean_cloud/fixtures/base.geojson") as src:
        base_tile_fc = dict(geojson.load(src))

    with open("test/test_cerulean_cloud/fixtures/offset.geojson") as src:
        offset_tile_fc = dict(geojson.load(src))

    merged = merge_inferences(
        [base_tile_fc, offset_tile_fc],
        proximity_meters=500,
        closing_meters=100,
        opening_meters=100,
    )
    with open("test/test_cerulean_cloud/fixtures/merge.geojson", "w") as outfile:
        json.dump(merged, outfile)
    assert merged["type"] == "FeatureCollection"
    assert len(merged["features"]) == 14

    for f in merged["features"]:
        print(f)
        assert f["geometry"]
        assert f["geometry"]["type"] in ["Polygon", "MultiPolygon"]
        assert f["properties"]
        assert f["properties"]["machine_confidence"]
        assert f["properties"]["inf_idx"]


def test_func_merge_inferences_empty():
    with open("test/test_cerulean_cloud/fixtures/offset.geojson") as src:
        offset_tile_fc = dict(geojson.load(src))

    merged = merge_inferences([geojson.FeatureCollection(features=[]), offset_tile_fc])
    assert merged["type"] == "FeatureCollection"
    assert len(merged["features"]) == 5

    merged = merge_inferences([offset_tile_fc, geojson.FeatureCollection(features=[])])
    assert merged["type"] == "FeatureCollection"
    assert len(merged["features"]) == 5

    merged = merge_inferences(
        [
            geojson.FeatureCollection(features=[]),
            geojson.FeatureCollection(features=[]),
        ],
    )
    assert merged["type"] == "FeatureCollection"
    assert len(merged["features"]) == 0


def test_get_tag():
    repo = git.Repo(search_parent_directories=True)
    git_tag = next(
        (tag.name for tag in repo.tags if tag.commit == repo.head.commit), None
    )
    if git_tag:
        assert isinstance(git_tag, str)
