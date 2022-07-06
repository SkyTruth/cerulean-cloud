import json

import pytest

from cerulean_cloud.roda_sentinelhub_client import RodaSentinelHubClient

S1_IDS = [
    "S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF",
    "S1A_IW_GRDH_1SDV_20201121T225759_20201121T225828_035353_04216C_62EA",
]


@pytest.mark.asyncio
async def test_get_product_info(httpx_mock):
    client = RodaSentinelHubClient()

    with open("test/test_cerulean_cloud/fixtures/productInfo.json") as src:
        info = json.load(src)
        httpx_mock.add_response(
            "https://roda.sentinel-hub.com/sentinel-s1-l1c/GRD/2020/7/29/IW/DV/S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF/productInfo.json",
            json=info,
        )

    res = await client.get_product_info(S1_IDS[0])
    assert "footprint" in res
