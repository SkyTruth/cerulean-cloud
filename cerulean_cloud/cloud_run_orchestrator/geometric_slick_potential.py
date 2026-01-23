from pathlib import Path
import geopandas as gpd
from joblib import load

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _THIS_DIR / "gsp_rf_85_acc_74_F1_20260123.joblib"


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
