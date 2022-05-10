import mercantile
import pytest

from cerulean_cloud.tiling import adjacent_tile, pixel_to_location
from cerulean_cloud.titiler_client import TitilerClient


@pytest.fixture
def titiler_client():
    return TitilerClient("https://0xshe4bmk8.execute-api.eu-central-1.amazonaws.com/")


S1_IDS = [
    "S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF",
    "S1A_IW_GRDH_1SDV_20201121T225759_20201121T225828_035353_04216C_62EA",
]


def test_get_bounds(titiler_client):
    b = titiler_client.get_bounds(S1_IDS[0])
    assert len(b) == 4
    assert b == [32.989094, 43.338009, 36.540836, 45.235191]


@pytest.fixture
def tiles_s1_scene():
    tiles = mercantile.tiles(
        *[32.989094, 43.338009, 36.540836, 45.235191], [11], truncate=False
    )

    return list(tiles)


def test_base_tile(titiler_client, tiles_s1_scene):
    array = titiler_client.get_base_tile(S1_IDS[0], tile=tiles_s1_scene[0])
    assert array.shape == (256, 256, 4)


def test_offset_tile(titiler_client, tiles_s1_scene):
    maxx, miny = pixel_to_location(adjacent_tile(tiles_s1_scene[0], 1, 1), 0.5, 0.5)
    minx, maxy = pixel_to_location(tiles_s1_scene[0], 0.5, 0.5)
    array = titiler_client.get_offset_tile(S1_IDS[0], minx, miny, maxx, maxy)
    print(minx, miny, maxx, maxy)
    assert array.shape == (256, 256, 4)
