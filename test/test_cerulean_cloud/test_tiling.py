import mercantile
import numpy as np
import pytest
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
        len(tiles_s1_scene) + tilexmax - tilexmin + tileymax - tileymin + 1
    )
    assert len(out) == expected_result
