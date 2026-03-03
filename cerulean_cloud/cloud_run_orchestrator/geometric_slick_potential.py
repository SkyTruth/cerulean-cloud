from pathlib import Path
import geopandas as gpd
from joblib import load
import numpy as np
from pyproj import Geod
from shapely.geometry import MultiPolygon
from geoalchemy2.shape import to_shape
from datetime import datetime
from sqlalchemy import select, update, and_, bindparam
import cerulean_cloud.database_schema as db

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _THIS_DIR / "gsp_rf_85_acc_74_F1_20260123.joblib"
_GSP_MODEL = None
_GSP_MODEL_PATH = None


def get_gsp_model(model_path: str):
    global _GSP_MODEL, _GSP_MODEL_PATH
    if _GSP_MODEL is None or _GSP_MODEL_PATH != model_path:
        _GSP_MODEL = load(model_path)
        _GSP_MODEL_PATH = model_path
    return _GSP_MODEL


def _to_valid_multipolygon(g):
    g = g.buffer(0)
    if g.geom_type == "Polygon":
        return MultiPolygon([g])
    return g


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
    slick_gdf["geometry"] = slick_gdf["geometry"].apply(_to_valid_multipolygon)

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
    preprocess=True,
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

    rf = get_gsp_model(model_path)

    feature_columns = rf.feature_names_
    X = (
        add_geom_columns(slick_gdf, feature_columns)
        if preprocess
        else slick_gdf[feature_columns]
    )

    return rf.predict_proba(X)[:, 1]


def slicks_to_gdf(slick_rows) -> gpd.GeoDataFrame:
    """
    Build a single GeoDataFrame from an iterable of Slick ORM objects or row objects
    that have .id, .slick_timestamp, .geometry, .aspect_ratio_factor, .machine_confidence.
    """
    records = []
    for s in slick_rows:
        records.append(
            {
                "id": s.id,
                "slick_timestamp": s.slick_timestamp,
                "geometry": to_shape(s.geometry),
                "aspect_ratio_factor": s.aspect_ratio_factor,
                "machine_confidence": s.machine_confidence,
            }
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


async def backfill_gsp_daterange_batched(
    client,
    start_dt: datetime,
    end_dt: datetime,
    *,
    batch_size: int = 2000,
    overwrite: bool = False,
):
    """
    Stable batched backfill for geometric_slick_potential.

    - Uses keyset pagination (id > last_id)
    - Does NOT filter IS NULL in SQL (prevents skipping)
    - Filters NULL rows in Python
    - Bulk UPDATE via executemany
    - Commits per batch
    """

    total_scanned = 0
    total_updated = 0
    last_id = 0

    while True:
        stmt = (
            select(db.Slick)
            .where(
                and_(
                    db.Slick.id > last_id,
                    db.Slick.slick_timestamp >= start_dt,
                    db.Slick.slick_timestamp < end_dt,
                    db.Slick.active.is_(True),
                )
            )
            .order_by(db.Slick.id.asc())
            .limit(batch_size)
        )

        result = await client.session.execute(stmt)
        slicks = result.scalars().all()

        if not slicks:
            break

        total_scanned += len(slicks)

        # ---- Filter rows to update in Python ----
        if overwrite:
            slicks_to_update = slicks
        else:
            slicks_to_update = [
                s for s in slicks if s.geometric_slick_potential is None
            ]

        if slicks_to_update:
            # ---- Build GeoDataFrame ----
            gdf = slicks_to_gdf(slicks_to_update)

            # ---- Predict in batch ----
            preds = predict_geometric_slick_potential(
                gdf,
                preprocess=True,
            )

            # ---- Prepare bulk update payload ----
            payload = []
            for s, p in zip(slicks_to_update, preds):
                if p is None:
                    continue
                fp = float(p)
                if fp != fp:  # guard against NaN
                    continue
                payload.append(
                    {
                        "b_id": int(s.id),
                        "b_gsp": fp,
                    }
                )

            if payload:
                stmt_update = (
                    update(db.Slick)
                    .where(db.Slick.id == bindparam("b_id"))
                    .values(geometric_slick_potential=bindparam("b_gsp"))
                )

                result = await client.session.execute(stmt_update, payload)
                total_updated += len(payload)

                await client.session.commit()

        last_id = slicks[-1].id

        print(
            f"Scanned: {total_scanned:,} | "
            f"Updated: {total_updated:,} | "
            f"Last ID: {last_id}"
        )

    return {
        "total_scanned": total_scanned,
        "total_updated": total_updated,
    }
