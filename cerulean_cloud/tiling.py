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


def from_base_tiles_create_offset_tiles(
    tiles: List[morecantile.Tile],
) -> List[Tuple[float, float, float, float]]:
    """from a set of base tiles, generate offset tiles"""
    out_offset_tile_bounds = []

    # create offset tile bounds with the up, left translation
    for tile in tiles:
        adj_tile = adjacent_tile(tile, -1, -1)
        minx, maxy = pixel_to_location(adj_tile, 0.5, 0.5)
        maxx, miny = pixel_to_location(tile, 0.5, 0.5)
        out_offset_tile_bounds += [(minx, miny, maxx, maxy)]

    tiles_np = np.array([(tile.x, tile.y, tile.z) for tile in tiles])
    zoom = tiles_np[:, 2].max()
    tilexmin, tilexmax, tileymin, tileymax = supermercado.super_utils.get_range(
        tiles_np
    )
    # create offset tile bounds of the down, left translation
    # (only in the bottom-most boundary)
    for tilex in range(tilexmin, tilexmax + 1):
        tile = morecantile.Tile(tilex, tileymax, zoom)
        adj_tile = adjacent_tile(tile, -1, 1)
        minx, miny = pixel_to_location(adj_tile, 0.5, 0.5)
        maxx, maxy = pixel_to_location(tile, 0.5, 0.5)
        out_offset_tile_bounds += [(minx, miny, maxx, maxy)]

    # create offset tile bounds of the up, ritgh translation
    # (only in the rigth-most boundary)
    for tiley in range(tileymin, tileymax + 1):
        tile = morecantile.Tile(tilexmax, tiley, zoom)
        adj_tile = adjacent_tile(tile, 1, -1)
        maxx, maxy = pixel_to_location(adj_tile, 0.5, 0.5)
        minx, miny = pixel_to_location(tile, 0.5, 0.5)
        out_offset_tile_bounds += [(minx, miny, maxx, maxy)]

    # bottom rigth corner tile
    bottom_rigth = morecantile.Tile(tilexmax, tileymax, zoom)
    adj_tile = adjacent_tile(bottom_rigth, 1, 1)
    minx, maxy = pixel_to_location(bottom_rigth, 0.5, 0.5)
    maxx, miny = pixel_to_location(adj_tile, 0.5, 0.5)
    out_offset_tile_bounds += [(minx, miny, maxx, maxy)]

    return out_offset_tile_bounds
