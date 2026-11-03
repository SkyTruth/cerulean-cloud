"""Scene filtering for historical backfills, independent of the live pipeline."""

import json
from pathlib import Path

from shapely.geometry import GeometryCollection, shape
from shapely.ops import unary_union


def load_ocean_poly(file_path=None):
    if file_path is None:
        file_path = Path(__file__).with_name("OceanGeoJSON_lowres.geojson")
    with open(file_path) as src:
        features = json.load(src)["features"]
    return GeometryCollection(
        [shape(feature["geometry"]).buffer(0) for feature in features]
    ).geoms[0]


def region_from_geojson(geometry):
    """Accept a nonempty FeatureCollection of polygons, like EODAG requests."""
    try:
        if geometry["type"] != "FeatureCollection":
            raise ValueError()
        polygons = [shape(feature["geometry"]) for feature in geometry["features"]]
        if not polygons or any(
            polygon.geom_type not in ("Polygon", "MultiPolygon")
            or polygon.is_empty
            or not polygon.is_valid
            for polygon in polygons
        ):
            raise ValueError()
        return unary_union(polygons)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(
            "geometry must be a nonempty GeoJSON FeatureCollection of valid polygons"
        ) from exc


def is_scene_relevant(footprint, resolution, polarization, ocean_poly, region=None):
    """Require supported resolution, VV imagery, ocean, and optional region overlap."""
    if resolution not in ("H", "M") or polarization not in ("DV", "SV"):
        return False
    scene = shape(footprint)
    return scene.intersects(ocean_poly) and (region is None or scene.intersects(region))
