"""tiling helpers
Inspired from
https://github.com/mapbox/robosat/blob/master/robosat/tiles.py
"""

from typing import List, Tuple

import morecantile
import numpy as np
import supermercado

TMS = morecantile.tms.get("WorldCRS84Quad")


def pixel_to_location(
    tile: morecantile.Tile, dx: float, dy: float
) -> Tuple[float, float]:
    """Converts a pixel in a tile to a coordinate.
    TMS.tiles(west, south, east, north, zooms, truncate=False)

    Args:
      tile: the morecantile tile to calculate the location in.
      dx: the relative x offset in range [0, 1] (fraction of tile width).
      dy: the relative y offset in range [0, 1] (fraction of tile height).

    Returns:
      The coordinate for the pixel in the tile.
    """

    assert 0 <= dx <= 1, "x offset is in [0, 1]"
    assert 0 <= dy <= 1, "y offset is in [0, 1]"

    west, south, east, north = TMS.bounds(tile)

    def lerp(a, b, c):
        return a + c * (b - a)

    lon = lerp(west, east, dx)
    lat = lerp(south, north, dy)

    return lon, lat


def adjacent_tile(tile: morecantile.Tile, dx: int, dy: int) -> morecantile.Tile:
    """Retrieves an adjacent tile from a tile store.
    Args:
      tile: the original tile to get an adjacent tile for.
      dx: the offset in tile x direction.
      dy: the offset in tile y direction.
    Returns:
      The adjacent tile's image or `None` if it does not exist.
    """

    x, y, z = map(int, [tile.x, tile.y, tile.z])
    other = morecantile.Tile(x=x + dx, y=y + dy, z=z)
    return other


def offset_bounds_from_base_tiles(
    tiles: List[morecantile.Tile],
) -> List[Tuple[float, float, float, float]]:
    """from a set of base tiles, generate offset tiles"""
    out_offset_tile_bounds = []

    tiles_np = np.array([(tile.x, tile.y, tile.z) for tile in tiles])
    zoom = tiles_np[:, 2].max()
    tilexmin, tilexmax, tileymin, tileymax = supermercado.super_utils.get_range(
        tiles_np
    )

    for new_x in range(tilexmin, tilexmax + 2):
        # +2 because tilexmax needs to be included (+1) and the new grid has one extra row/column (+1)
        for new_y in range(tileymin, tileymax + 2):
            # +2 because tileymax needs to be included (+1) and the new grid has one extra row/column (+1)
            tile = morecantile.Tile(new_x, new_y, zoom)
            adj_tile = adjacent_tile(tile, -1, -1)
            minx, miny = pixel_to_location(adj_tile, 0.5, 0.5)
            maxx, maxy = pixel_to_location(tile, 0.5, 0.5)
            out_offset_tile_bounds += [(minx, miny, maxx, maxy)]

    return out_offset_tile_bounds
