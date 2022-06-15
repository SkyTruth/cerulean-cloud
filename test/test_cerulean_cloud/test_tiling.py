import geojson
import numpy as np
import pytest
import shapely.geometry
import supermercado

from cerulean_cloud.tiling import TMS, from_base_tiles_create_offset_tiles


@pytest.fixture
def tiles_s1_scene():
    tiles = TMS.tiles(
        *[32.989094, 43.338009, 36.540836, 45.235191], [11], truncate=False
    )

    return list(tiles)


def test_from_base_tiles_create_offset_tiles(tiles_s1_scene):  # noqa: F811
    out = from_base_tiles_create_offset_tiles(tiles_s1_scene)

    tiles_np = np.array([(tile.x, tile.y, tile.z) for tile in tiles_s1_scene])
    tilexmin, tilexmax, tileymin, tileymax = supermercado.super_utils.get_range(
        tiles_np
    )

    expected_result = (
        len(tiles_s1_scene) + tilexmax - tilexmin + 1 + tileymax - tileymin + 1 + 1
    )
    assert len(out) == expected_result
    print(out[-1])
    assert out[-1] == pytest.approx(
        (36.518554687499716, 43.28613281250006, 36.606445312499716, 43.37402343750006)
    )

    # Make sure bounds are in minx, miny, maxx, maxy

    for i, o in enumerate(out):
        # print(i)
        assert o[1] < o[3]
        assert o[0] < o[2]


@pytest.mark.skip
def test_save_tiles_to_file():
    # zoom level 10
    # tile size 512x512
    # returns resolution around 80m, 0.004 degrees
    tiles_s1_scene = list(
        TMS.tiles(*[32.989094, 43.338009, 36.540836, 45.235191], [10], truncate=False)
    )

    feat_base = [TMS.feature(tile) for tile in tiles_s1_scene]
    with open("base_tiles.json", "w") as dst:
        geojson.dump(geojson.FeatureCollection(features=feat_base), dst)

    feat_offset = [
        geojson.Feature(
            geometry=shapely.geometry.mapping(shapely.geometry.box(*bbox)),
            bbox=list(bbox),
        )
        for bbox in from_base_tiles_create_offset_tiles(tiles_s1_scene)
    ]
    with open("offset_tiles.json", "w") as dst:
        geojson.dump(geojson.FeatureCollection(features=feat_offset), dst)
