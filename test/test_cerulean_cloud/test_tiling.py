import geojson
import mercantile
import numpy as np
import pytest
import shapely.geometry
import supermercado

from cerulean_cloud.tiling import from_base_tiles_create_offset_tiles


@pytest.fixture
def tiles_s1_scene():
    tiles = mercantile.tiles(
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

    assert out[-1] == pytest.approx(
        (36.474609375, 43.3890482867738, 36.650390625, 43.261172481247115)
    )


@pytest.mark.skip
def test_save_tiles_to_file(tiles_s1_scene):
    feat_base = [mercantile.feature(tile) for tile in tiles_s1_scene]
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
