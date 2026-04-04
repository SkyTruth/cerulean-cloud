"""Pure helper logic for the sea-ice sync worker."""

from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

DEFAULT_NHSI_BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G02156/GIS/4km"
DEFAULT_CADENCE_DAYS = 1
DEFAULT_ANCHOR_DATE = date(2026, 4, 1)
DEFAULT_SIMPLIFY_TOLERANCE = 0.05
MAX_SOURCE_LOOKBACK_DAYS = 2


def utc_today() -> date:
    """Return the current UTC date."""
    return datetime.now(timezone.utc).date()


def should_run_today(today: date, anchor_date: date, cadence_days: int) -> bool:
    """Return whether the worker should run on the given date."""
    if cadence_days < 1:
        raise ValueError("cadence_days must be >= 1")
    delta_days = (today - anchor_date).days
    return delta_days >= 0 and delta_days % cadence_days == 0


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Split a ``gs://`` URI into bucket and object name."""
    parsed = urlparse(gcs_uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def nhsi_filename_for_day(target_day: date) -> str:
    """Build the NOAA/NHSI filename for the target day."""
    yyyydoy = f"{target_day.year}{target_day.timetuple().tm_yday:03d}"
    return f"ims{yyyydoy}_4km_GIS_v1.3.tif.gz"


def nhsi_source_url_for_day(source_root: str, target_day: date) -> str:
    """Build the source URL for the target day."""
    file_name = nhsi_filename_for_day(target_day)
    yyyydoy = f"{target_day.year}{target_day.timetuple().tm_yday:03d}"

    if (
        "{file_name}" in source_root
        or "{yyyy}" in source_root
        or "{yyyydoy}" in source_root
    ):
        return source_root.format(
            yyyy=target_day.year,
            yyyydoy=yyyydoy,
            file_name=file_name,
        )
    if source_root.endswith(".gz"):
        return source_root
    return f"{source_root.rstrip('/')}/{target_day.year}/{file_name}"


def candidate_source_days(
    today: date, lookback_days: int = MAX_SOURCE_LOOKBACK_DAYS
) -> list[date]:
    """Return source dates to try, newest first."""
    return [today - timedelta(days=offset) for offset in range(lookback_days + 1)]


def build_object_name(gcs_uri: str, source_date: date) -> tuple[str, str]:
    """Build the dated object path the orchestrator expects.

    ``gcs_uri`` must point at a prefix such as ``gs://bucket/extent_vectors/``.
    """
    bucket_name, object_path = parse_gcs_uri(gcs_uri)
    if object_path.endswith(".geojson"):
        raise ValueError(
            "SEA_ICE_MASK_GCS_URI must be a prefix directory, not a file path"
        )
    object_prefix = object_path.rstrip("/")

    if not object_prefix:
        raise ValueError(
            f"SEA_ICE_MASK_GCS_URI must include a prefix directory: {gcs_uri}"
        )

    return bucket_name, f"{object_prefix}/{source_date.isoformat()}_extent.geojson"


def normalize_mask_polygons(gdf):
    """Remove holes and dissolve polygon overlaps for the final mask."""
    import geopandas as gpd
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    from shapely.ops import unary_union

    def polygon_parts(geometry) -> list:
        if geometry is None or geometry.is_empty:
            return []
        if isinstance(geometry, Polygon):
            return [geometry]
        if isinstance(geometry, MultiPolygon):
            return [polygon for polygon in geometry.geoms if not polygon.is_empty]
        if isinstance(geometry, GeometryCollection):
            parts = []
            for part in geometry.geoms:
                parts.extend(polygon_parts(part))
            return parts
        return []

    def remove_holes(geometry):
        polygons = [Polygon(polygon.exterior) for polygon in polygon_parts(geometry)]
        if not polygons:
            return GeometryCollection()
        if len(polygons) == 1:
            return polygons[0]
        return MultiPolygon(polygons)

    if gdf.empty:
        return gdf.copy()

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.map(remove_holes)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    parts = []
    for geometry in gdf.geometry:
        parts.extend(polygon_parts(geometry))

    if not parts:
        return gdf.iloc[0:0].copy()

    dissolved_geometry = unary_union(parts)
    dissolved_parts = polygon_parts(dissolved_geometry)
    if not dissolved_parts:
        return gdf.iloc[0:0].copy()

    template_row = {
        column: gdf.iloc[0][column]
        for column in gdf.columns
        if column != gdf.geometry.name
    }
    return gpd.GeoDataFrame(
        [template_row.copy() for _ in dissolved_parts],
        geometry=dissolved_parts,
        crs=gdf.crs,
    )
