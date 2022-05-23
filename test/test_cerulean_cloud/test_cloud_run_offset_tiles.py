import pytest

from cerulean_cloud.tiling import TMS
from cerulean_cloud.titiler_client import TitilerClient


@pytest.mark.skip
def test_create_fixture_tile(
    url="https://0xshe4bmk8.execute-api.eu-central-1.amazonaws.com/",
):
    titiler_client = TitilerClient(url=url)
    S1_ID = "S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF"

    tiles = list(TMS.tiles(*titiler_client.get_bounds(S1_ID), [10], truncate=False))
    tile = tiles[20]
    # import pdb; pdb.set_trace()
    array = titiler_client.get_base_tile(S1_ID, tile=tile, scale=2)
    print(array)
