from test.test_cerulean_cloud.test_inference_client import (
    mock_get_base_tile,
    mock_get_offset_tile,
)
from unittest.mock import patch

import pytest

import cerulean_cloud
import cerulean_cloud.cloud_run_orchestrator.handler as handler
from cerulean_cloud.cloud_run_offset_tiles.schema import InferenceResult
from cerulean_cloud.cloud_run_orchestrator.clients import CloudRunInferenceClient
from cerulean_cloud.cloud_run_orchestrator.schema import OrchestratorInput
from cerulean_cloud.tiling import TMS
from cerulean_cloud.titiler_client import TitilerClient

S1_ID = "S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF"


@pytest.fixture
def fixture_titiler_client():
    return TitilerClient("some_url")


@pytest.fixture
def fixture_cloud_inference(fixture_titiler_client):
    return CloudRunInferenceClient(
        url="some_url", titiler_client=fixture_titiler_client
    )


def mock_get_base_tile_inference(self, sceneid, tile, rescale):
    return InferenceResult(classes="", confidence="", bounds=list(TMS.bounds(tile)))


def mock_get_offset_tile_inference(self, sceneid, bounds, rescale):
    return InferenceResult(classes="", confidence="", bounds=bounds)


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
    res = handler._orchestrate(
        payload, TMS, fixture_titiler_client, fixture_cloud_inference
    )
    assert res.ntiles == 252
    assert res.noffsettiles == 286
