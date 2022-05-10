"""tiling helpers
Inspired from
https://github.com/mapbox/robosat/blob/master/robosat/tiles.py
"""

from typing import Tuple

import mercantile


def pixel_to_location(
    tile: mercantile.Tile, dx: float, dy: float
) -> Tuple[float, float]:
    """Converts a pixel in a tile to a coordinate.
    mercantile.tiles(west, south, east, north, zooms, truncate=False)

    Args:
      tile: the mercantile tile to calculate the location in.
      dx: the relative x offset in range [0, 1].
      dy: the relative y offset in range [0, 1].

    Returns:
      The coordinate for the pixel in the tile.
    """

    assert 0 <= dx <= 1, "x offset is in [0, 1]"
    assert 0 <= dy <= 1, "y offset is in [0, 1]"

    west, south, east, north = mercantile.bounds(tile)

    def lerp(a, b, c):
        return a + c * (b - a)

    lon = lerp(west, east, dx)
    lat = lerp(south, north, dy)

    return lon, lat


def adjacent_tile(tile: mercantile.Tile, dx: int, dy: int) -> mercantile.Tile:
    """Retrieves an adjacent tile from a tile store.
    Args:
      tile: the original tile to get an adjacent tile for.
      dx: the offset in tile x direction.
      dy: the offset in tile y direction.
    Returns:
      The adjacent tile's image or `None` if it does not exist.
    """

    x, y, z = map(int, [tile.x, tile.y, tile.z])
    other = mercantile.Tile(x=x + dx, y=y + dy, z=z)
    return other
