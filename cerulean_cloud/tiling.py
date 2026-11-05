"""tiling helpers
Inspired from
https://github.com/mapbox/robosat/blob/master/robosat/tiles.py
"""

from itertools import groupby
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

    # WorldCRS84Quad coordinates are already longitude/latitude. xy_bounds also
    # allows extrapolating neighboring tiles at the edge of the grid.
    west, south, east, north = TMS.xy_bounds(tile)

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


def group_tiles_by_x(tiles: List[morecantile.Tile]) -> List[List[morecantile.Tile]]:
    """Keep contiguous tile columns together, separating the dateline sides."""
    columns = sorted({tile.x for tile in tiles})
    groups = []
    for _, group in groupby(enumerate(columns), key=lambda item: item[1] - item[0]):
        group_columns = {x for _, x in group}
        groups.append([tile for tile in tiles if tile.x in group_columns])
    return groups


def offset_bounds_from_base_tiles(
    tiles: List[morecantile.Tile],
    offset_amount: float = 0.5,
) -> List[Tuple[float, float, float, float]]:
    """from a set of base tiles, generate offset tiles"""
    out_offset_tile_bounds = []

    for group in group_tiles_by_x(tiles):
        tiles_np = np.array([(tile.x, tile.y, tile.z) for tile in group])
        zoom = tiles_np[:, 2].max()
        tilexmin, tilexmax, tileymin, tileymax = supermercado.super_utils.get_range(
            tiles_np
        )

        for new_x in range(tilexmin, tilexmax + 2):
            # Include the last column and one extra offset column.
            for new_y in range(tileymin, tileymax + 2):
                tile = morecantile.Tile(new_x, new_y, zoom)
                adj_tile = adjacent_tile(tile, -1, -1)  # Negative dY is upwards!!!
                minx, maxy = pixel_to_location(adj_tile, offset_amount, offset_amount)
                maxx, miny = pixel_to_location(tile, offset_amount, offset_amount)
                # Move boundary windows inside the world without changing their
                # width or the resolution seen by the model.
                shift = max(-180.0 - minx, 0.0) - max(maxx - 180.0, 0.0)
                out_offset_tile_bounds.append((minx + shift, miny, maxx + shift, maxy))

    return list(dict.fromkeys(out_offset_tile_bounds))
