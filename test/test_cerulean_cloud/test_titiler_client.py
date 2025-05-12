import mercantile
import pytest

from cerulean_cloud.tiling import adjacent_tile, pixel_to_location
from cerulean_cloud.titiler_client import TMS_TITLE, TitilerClient


@pytest.fixture
def titiler_client():
    return TitilerClient("https://titiler.url/")


S1_IDS = [
    "S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF",
    "S1A_IW_GRDH_1SDV_20201121T225759_20201121T225828_035353_04216C_62EA",
]


@pytest.mark.asyncio
async def test_get_bounds(titiler_client, httpx_mock):
    scene_id = S1_IDS[0]
    httpx_mock.add_response(
        method="GET",
        url=titiler_client.url + f"bounds?sceneid={scene_id}",
        json={"bounds": [32.989094, 43.338009, 36.540836, 45.235191]},
    )
    b = await titiler_client.get_bounds(scene_id)
    assert len(b) == 4
    assert b == [32.989094, 43.338009, 36.540836, 45.235191]


@pytest.mark.asyncio
async def test_get_statistics(titiler_client, httpx_mock):
    scene_id = S1_IDS[0]
    httpx_mock.add_response(
        method="GET",
        url=titiler_client.url + f"statistics?sceneid={scene_id}&bands=vv",
        json={
            "vv": {
                "min": 19,
                "max": 1075,
                "mean": 86.85049361077405,
                "count": 400518,
                "sum": 34785186,
                "std": 48.50235864218863,
                "median": 78,
                "majority": 46,
                "minority": 350,
                "unique": 394,
                "histogram": [
                    [330487, 63767, 6023, 230, 9, 1, 0, 0, 0, 1],
                    [
                        19,
                        124.6,
                        230.2,
                        335.79999999999995,
                        441.4,
                        547,
                        652.5999999999999,
                        758.1999999999999,
                        863.8,
                        969.4,
                        1075,
                    ],
                ],
                "valid_percent": 71.37,
                "masked_pixels": 160634,
                "valid_pixels": 400518,
                "percentile_98": 223,
                "percentile_2": 31,
            }
        },
    )
    s = await titiler_client.get_statistics(scene_id)
    assert len(s) == 16
    assert s.get("max") == 1075


@pytest.fixture
def tiles_s1_scene():
    tiles = mercantile.tiles(
        *[32.989094, 43.338009, 36.540836, 45.235191], [11], truncate=False
    )

    return list(tiles)


@pytest.mark.asyncio
async def test_base_tile(titiler_client, tiles_s1_scene, httpx_mock):
    scene_id = S1_IDS[0]
    tile = tiles_s1_scene[0]

    with open("test/test_cerulean_cloud/fixtures/example_tile.png", "rb") as src:
        img_bytes = src.read()
    httpx_mock.add_response(
        method="GET",
        url=titiler_client.url
        + f"tiles/{TMS_TITLE}/{tile.z}/{tile.x}/{tile.y}?sceneid={scene_id}&bands=vv&format=png&scale=1&rescale=0,255",
        content=img_bytes,
    )
    array = await titiler_client.get_base_tile(S1_IDS[0], tile=tile)
    assert array.shape == (256, 256, 2)


@pytest.mark.asyncio
async def test_offset_tile(titiler_client, tiles_s1_scene, httpx_mock):
    scene_id = S1_IDS[0]
    tile = tiles_s1_scene[0]

    maxx, miny = pixel_to_location(adjacent_tile(tile, 1, 1), 0.5, 0.5)
    minx, maxy = pixel_to_location(tile, 0.5, 0.5)

    with open("test/test_cerulean_cloud/fixtures/example_tile.png", "rb") as src:
        img_bytes = src.read()

    httpx_mock.add_response(
        method="GET",
        url=titiler_client.url
        + f"bbox/{minx},{miny},{maxx},{maxy}/256x256.png?sceneid={scene_id}&bands=vv&rescale=0,255",
        content=img_bytes,
    )

    array = await titiler_client.get_offset_tile(S1_IDS[0], minx, miny, maxx, maxy)
    print(minx, miny, maxx, maxy)
    assert array.shape == (256, 256, 2)


def test_get_base_tile_url(titiler_client):
    res = titiler_client.get_base_tile_url("ABC")
    assert (
        res
        == "https://titiler.url/tiles/{z}/{x}/{y}?sceneid=ABC&bands=vv&scale=1&rescale=0,255"
    )
