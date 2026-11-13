"""Ocean mask for historical EODAG discovery."""

import json
from pathlib import Path

from shapely.geometry import GeometryCollection, shape


def load_ocean_poly(file_path=None):
    if file_path is None:
        file_path = Path(__file__).with_name("OceanGeoJSON_lowres.geojson")
    with open(file_path) as src:
        features = json.load(src)["features"]
    return GeometryCollection(
        [shape(feature["geometry"]).buffer(0) for feature in features]
    ).geoms[0]
