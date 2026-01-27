from pathlib import Path
import geopandas as gpd
from joblib import load
import numpy as np
from pyproj import Geod

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _THIS_DIR / "gsp_rf_85_acc_74_F1_20260123.joblib"


def postgis_geography_perimeter(geom):
    """
    Matches ST_Perimeter(geography)
    """
    geod = Geod(ellps="WGS84")

    def ring_length(coords):
        lons, lats = zip(*coords)
        return geod.line_length(lons, lats)

    perimeter = 0.0

    if geom.geom_type == "Polygon":
        # exterior
        perimeter += ring_length(geom.exterior.coords)

        # interior rings (holes)
        for ring in geom.interiors:
            perimeter += ring_length(ring.coords)

    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            perimeter += ring_length(poly.exterior.coords)
            for ring in poly.interiors:
                perimeter += ring_length(ring.coords)

    return perimeter


def add_geom_columns(
    slick_gdf: gpd.GeoDataFrame,
    feature_columns: list[str] = None,
) -> gpd.GeoDataFrame:
    """
    Add geometry-derived feature columns for MultiPolygon geometries.

    This function assumes the geometry column contains MultiPolygon objects
    and computes several area-based features after projecting the data
    to an equal-area CRS.
    """
    slick_gdf = gpd.GeoDataFrame(slick_gdf)

    slick_gdf["geometry_count"] = slick_gdf["geometry"].apply(
        lambda geom: len(geom.geoms)
    )

    slick_gdf_newProj = slick_gdf.to_crs("EPSG:6933")

    slick_gdf_newProj["largest_area"] = slick_gdf_newProj["geometry"].apply(
        lambda geom: max(part.area for part in geom.geoms)
    )

    # Apply to all members of the dataframe
    def median_area(multipoly):
        num_geom = len(multipoly.geoms)
        middle = num_geom // 2
        areas = [part.area for part in multipoly.geoms]
        return sorted(areas)[middle]

    slick_gdf_newProj["median_area"] = slick_gdf_newProj["geometry"].apply(median_area)

    # Total area and perimeter (entire MultiPolygon)
    slick_gdf_newProj["area"] = slick_gdf_newProj.geometry.area
    slick_gdf_newProj["perimeter"] = slick_gdf.geometry.apply(
        postgis_geography_perimeter
    )

    # Polsby–Popper: 4πA / P²
    slick_gdf_newProj["polsby_popper"] = (
        4.0 * np.pi * slick_gdf_newProj["area"] / (slick_gdf_newProj["perimeter"] ** 2)
    )

    # Oriented envelope (minimum rotated rectangle)
    slick_gdf_newProj["oriented_envelope"] = slick_gdf_newProj.geometry.apply(
        lambda g: g.minimum_rotated_rectangle
    )

    # Fill factor: area / area(oriented envelope)
    slick_gdf_newProj["fill_factor"] = (
        slick_gdf_newProj["area"] / slick_gdf_newProj["oriented_envelope"].area
    )

    if feature_columns is not None:
        return slick_gdf_newProj[feature_columns]
    return slick_gdf_newProj


def predict_geometric_slick_potential(
    slick_gdf: gpd.GeoDataFrame,
    model_path: Path | str = _DEFAULT_MODEL_PATH,
):
    """
    Compute geometric slick potential from geometric predictors.

    The model path is resolved relative to this module, not the caller.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Geometric slick potential model not found at: {model_path}"
        )

    rf = load(model_path)

    feature_columns = rf.feature_names_
    X = add_geom_columns(slick_gdf, feature_columns)

    return rf.predict_proba(X)[:, 1]
