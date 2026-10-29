import geojson
import numpy as np
import pytest
import shapely.geometry
import supermercado
from shapely.ops import unary_union

from cerulean_cloud.tiling import TMS, group_tiles_by_x, offset_bounds_from_base_tiles


@pytest.fixture
def tiles_s1_scene():
    tiles = TMS.tiles(
        *[32.989094, 43.338009, 36.540836, 45.235191], [11], truncate=False
    )

    return list(tiles)


def test_from_base_tiles_create_offset_tiles(tiles_s1_scene):  # noqa: F811
    out = offset_bounds_from_base_tiles(tiles_s1_scene)

    tiles_np = np.array([(tile.x, tile.y, tile.z) for tile in tiles_s1_scene])
    tilexmin, tilexmax, tileymin, tileymax = supermercado.super_utils.get_range(
        tiles_np
    )
    assert len(out) == (tilexmax - tilexmin + 2) * (tileymax - tileymin + 2)
    print(out[-1])
    assert out[-1] == pytest.approx(
        (36.518554687499716, 43.28613281250006, 36.606445312499716, 43.37402343750006)
    )

    # Make sure bounds are in minx, miny, maxx, maxy

    for i, o in enumerate(out):
        # print(i)
        assert o[1] < o[3]
        assert o[0] < o[2]


@pytest.mark.parametrize("offset_amount", [0.0, 0.33, 0.66])
@pytest.mark.parametrize(
    "bounds",
    [
        # Original upstream issue #77 scene, now returned as wrapped bounds.
        (175.38292995222957, 61.06949078480844, -179.34123633603335, 62.88226850489882),
        (179.5, 61.0, 179.9, 62.0),
        (-179.9, 61.0, -179.5, 62.0),
    ],
    ids=["crossing", "near-east", "near-west"],
)
def test_offset_tiles_at_antimeridian(bounds, offset_amount):
    base_tiles = list(TMS.tiles(*bounds, [9], truncate=False))
    groups = group_tiles_by_x(base_tiles)
    offsets = offset_bounds_from_base_tiles(base_tiles, offset_amount)
    tile_width = 360 / TMS.matrix(9).matrixWidth

    assert len(groups) == (2 if bounds[0] > bounds[2] else 1)
    assert len(offsets) == len(set(offsets))
    assert len(offsets) <= sum(
        (len({t.x for t in group}) + 1) * (len({t.y for t in group}) + 1)
        for group in groups
    )
    for west, south, east, north in offsets:
        assert -180 <= west < east <= 180
        assert east - west == pytest.approx(tile_width)
        assert north - south == pytest.approx(tile_width)

    # Boundary windows retain the full scene coverage at every offset.
    coverage = unary_union([shapely.geometry.box(*b) for b in offsets])
    base_coverage = unary_union(
        [shapely.geometry.box(*TMS.bounds(tile)) for tile in base_tiles]
    )
    assert coverage.covers(base_coverage)


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
        for bbox in offset_bounds_from_base_tiles(tiles_s1_scene)
    ]
    with open("offset_tiles.json", "w") as dst:
        geojson.dump(geojson.FeatureCollection(features=feat_offset), dst)
