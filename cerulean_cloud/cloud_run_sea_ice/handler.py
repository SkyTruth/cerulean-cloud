"""Cloud Run worker for downloading, processing, and uploading sea-ice extent."""

from __future__ import annotations

import gzip
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI

logger = logging.getLogger("cerulean_cloud.cloud_run_sea_ice")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_NHSI_BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G02156/GIS/4km"
DEFAULT_SIMPLIFY_TOLERANCE = 10_000.0
DEFAULT_SIMPLIFY_SRS = "EPSG:4087"
ICE_CLASS_VALUE = 3
PRECISION_GRID_SIZE = 0.0001
REQUEST_TIMEOUT_SECONDS = 600
MAX_SOURCE_LOOKBACK_DAYS = 14

app = FastAPI(title="Cloud Run sea-ice sync")


@dataclass(frozen=True)
class DownloadedSource:
    """Information about the source raster selected for processing."""

    source_date: date
    source_url: str
    gz_path: Path


@dataclass(frozen=True)
class AvailableSource:
    """Information about an available upstream raster."""

    source_date: date
    source_url: str


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


def iter_source_days(today: date, max_lookback_days: int):
    """Yield candidate source days from newest to oldest."""
    for days_back in range(max_lookback_days):
        yield today - timedelta(days=days_back)


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
        return gdf

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


def download_latest_source(
    source_root: str,
    workdir: Path,
    today: date,
    timeout_seconds: int,
) -> DownloadedSource:
    """Download the NHSI raster for a specific day."""
    import requests

    session = requests.Session()
    source_url = nhsi_source_url_for_day(source_root, today)
    destination = workdir / nhsi_filename_for_day(today)
    logger.info(
        {
            "message": "Attempting NHSI download",
            "source_date": today.isoformat(),
            "source_url": source_url,
        }
    )
    try:
        with session.get(
            source_url,
            stream=True,
            timeout=timeout_seconds,
        ) as response:
            if response.status_code == 404:
                raise FileNotFoundError(
                    f"No NHSI raster available for {today.isoformat()}"
                )
            response.raise_for_status()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as dst:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        dst.write(chunk)
        return DownloadedSource(
            source_date=today,
            source_url=source_url,
            gz_path=destination,
        )
    except requests.RequestException as exc:
        logger.warning(
            {
                "message": "NHSI download attempt failed",
                "source_date": today.isoformat(),
                "source_url": source_url,
                "error": str(exc),
            }
        )
        raise


def find_latest_available_source(
    source_root: str,
    today: date,
    timeout_seconds: int,
    max_lookback_days: int,
) -> AvailableSource:
    """Find the newest available NHSI raster within the lookback window."""
    import requests

    session = requests.Session()
    for source_day in iter_source_days(today, max_lookback_days):
        source_url = nhsi_source_url_for_day(source_root, source_day)
        logger.info(
            {
                "message": "Probing NHSI source",
                "source_date": source_day.isoformat(),
                "source_url": source_url,
            }
        )
        try:
            with session.get(
                source_url,
                stream=True,
                timeout=timeout_seconds,
            ) as response:
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                return AvailableSource(
                    source_date=source_day,
                    source_url=source_url,
                )
        except requests.RequestException as exc:
            logger.warning(
                {
                    "message": "NHSI source probe failed",
                    "source_date": source_day.isoformat(),
                    "source_url": source_url,
                    "error": str(exc),
                }
            )
            raise

    raise FileNotFoundError(
        f"No NHSI raster available from {today.isoformat()} "
        f"within the last {max_lookback_days} day(s)"
    )


def gunzip_file(source_path: Path, destination: Path) -> None:
    """Expand a ``.gz`` file to its uncompressed TIFF."""
    with gzip.open(source_path, "rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def polygonize_ice_raster(source_path: Path):
    """Polygonize only the ice class from the source raster."""
    import geopandas as gpd
    import rasterio
    from rasterio import features
    from shapely.geometry import shape

    with rasterio.open(source_path) as src:
        band = src.read(1)
        ice_mask = band == ICE_CLASS_VALUE
        if not ice_mask.any():
            return gpd.GeoDataFrame({"DN": []}, geometry=[], crs=src.crs)

        geoms = []
        vals = []

        for geom, val in features.shapes(
            band,
            mask=ice_mask,
            transform=src.transform,
            connectivity=8,
        ):
            if int(val) != ICE_CLASS_VALUE:
                continue
            geoms.append(shape(geom))
            vals.append(int(val))

        return gpd.GeoDataFrame({"DN": vals}, geometry=geoms, crs=src.crs)


def remove_land_parts(gdf):
    """Remove polygons whose representative point falls on land."""
    from global_land_mask import globe

    if gdf.empty:
        return gdf

    representative_points = gdf.geometry.representative_point()
    if (
        representative_points.crs is not None
        and representative_points.crs.to_epsg() != 4326
    ):
        representative_points = representative_points.to_crs("EPSG:4326")
    keep_mask = [not globe.is_land(point.y, point.x) for point in representative_points]
    return gdf.loc[keep_mask].copy()


def empty_mask_gdf():
    """Create an empty output GeoDataFrame in GeoJSON CRS."""
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"DN": [], "ice_date": []},
        geometry=[],
        crs="EPSG:4326",
    )


def filter_and_simplify_polygons(
    gdf,
    destination: Path,
    source_date: date,
    simplify_tolerance: float,
    simplify_srs: str,
) -> None:
    """Simplify ice polygons for publication and write GeoJSON output."""
    from shapely import make_valid, set_precision

    if gdf.empty:
        empty_mask_gdf().to_file(destination, driver="GeoJSON")
        return

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.map(make_valid)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        empty_mask_gdf().to_file(destination, driver="GeoJSON")
        return

    gdf["ice_date"] = source_date.isoformat()
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = remove_land_parts(gdf)
    gdf = normalize_mask_polygons(gdf)

    if gdf.empty:
        empty_mask_gdf().to_file(destination, driver="GeoJSON")
        return

    if gdf.crs is not None and gdf.crs.is_projected:
        simplified = gdf
    else:
        simplified = gdf.to_crs(simplify_srs)

    simplified = simplified.copy()
    simplified["geometry"] = simplified.geometry.simplify(
        simplify_tolerance,
        preserve_topology=True,
    )
    if simplified.crs is None or simplified.crs.to_epsg() != 4326:
        simplified = simplified.to_crs("EPSG:4326")
    simplified["geometry"] = simplified.geometry.map(
        lambda geom: set_precision(geom, PRECISION_GRID_SIZE)
    )
    simplified = simplified[
        simplified.geometry.notna() & ~simplified.geometry.is_empty
    ].copy()
    if simplified.empty:
        empty_mask_gdf().to_file(destination, driver="GeoJSON")
        return

    simplified.to_file(destination, driver="GeoJSON")


def process_downloaded_source(
    downloaded_source: DownloadedSource,
    workdir: Path,
    simplify_tolerance: float,
    simplify_srs: str,
) -> Path:
    """Run the NHSI raster processing pipeline and return the final GeoJSON path."""
    uncompressed_tif = workdir / downloaded_source.gz_path.with_suffix("").name
    output_geojson = workdir / "extent.geojson"

    gunzip_file(downloaded_source.gz_path, uncompressed_tif)
    ice_polygons = polygonize_ice_raster(uncompressed_tif)
    filter_and_simplify_polygons(
        ice_polygons,
        output_geojson,
        source_date=downloaded_source.source_date,
        simplify_tolerance=simplify_tolerance,
        simplify_srs=simplify_srs,
    )
    return output_geojson


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Split a GCS URI into bucket name and normalized object prefix."""
    bucket_name, object_prefix = gcs_uri.removeprefix("gs://").split("/", 1)
    return bucket_name, object_prefix.strip("/")


def object_name_for_source_date(object_prefix: str, source_date: date) -> str:
    """Build the output object name for a specific source day."""
    object_basename = f"{source_date.isoformat()}_extent.geojson"
    if not object_prefix:
        return object_basename
    return f"{object_prefix}/{object_basename}"


def vector_output_exists(bucket_name: str, object_name: str) -> bool:
    """Check whether a dated vector output already exists in GCS."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    return blob.exists()


def upload_outputs(
    bucket_name: str,
    object_name: str,
    source_path: Path,
    metadata: dict[str, str],
    if_generation_match: int | None = None,
) -> None:
    """Upload the current object to GCS."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.metadata = metadata
    blob.upload_from_filename(
        str(source_path),
        content_type="application/geo+json",
        if_generation_match=if_generation_match,
    )


def utc_today() -> date:
    """Return the current UTC calendar day."""
    return datetime.now(timezone.utc).date()


def _sync() -> dict[str, str]:
    from google.api_core.exceptions import PreconditionFailed

    mask_gcs_uri = os.environ["SEA_ICE_MASK_GCS_URI"]
    today = utc_today()
    workdir = Path("/tmp/sea-ice-sync")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        available_source = find_latest_available_source(
            DEFAULT_NHSI_BASE_URL,
            today=today,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            max_lookback_days=MAX_SOURCE_LOOKBACK_DAYS,
        )
    except FileNotFoundError as exc:
        logger.info({"message": str(exc), "source_date": today.isoformat()})
        return {
            "status": "skipped",
            "source_date": today.isoformat(),
            "reason": str(exc),
        }

    bucket_name, object_prefix = parse_gcs_uri(mask_gcs_uri)
    object_name = object_name_for_source_date(
        object_prefix,
        available_source.source_date,
    )
    if vector_output_exists(bucket_name, object_name):
        logger.info(
            {
                "message": "Sea-ice extent already vectorized",
                "object_name": object_name,
                "source_date": available_source.source_date.isoformat(),
            }
        )
        return {
            "status": "up_to_date",
            "object_name": object_name,
            "source_date": available_source.source_date.isoformat(),
        }

    downloaded_source = download_latest_source(
        DEFAULT_NHSI_BASE_URL,
        workdir,
        today=available_source.source_date,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )
    output_path = process_downloaded_source(
        downloaded_source,
        workdir,
        simplify_tolerance=DEFAULT_SIMPLIFY_TOLERANCE,
        simplify_srs=DEFAULT_SIMPLIFY_SRS,
    )

    metadata = {
        "source_url": downloaded_source.source_url,
        "mask_gcs_uri": mask_gcs_uri,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_date": downloaded_source.source_date.isoformat(),
        "source_filename": downloaded_source.gz_path.name,
    }
    try:
        upload_outputs(
            bucket_name=bucket_name,
            object_name=object_name,
            source_path=output_path,
            metadata=metadata,
            if_generation_match=0,
        )
    except PreconditionFailed:
        logger.info(
            {
                "message": "Sea-ice extent already uploaded by another run",
                "object_name": object_name,
                "source_date": downloaded_source.source_date.isoformat(),
            }
        )
        return {
            "status": "up_to_date",
            "object_name": object_name,
            "source_date": downloaded_source.source_date.isoformat(),
        }

    logger.info(
        {
            "message": "Uploaded simplified sea-ice extent",
            "object_name": object_name,
            "source_date": downloaded_source.source_date.isoformat(),
        }
    )
    return {
        "status": "ok",
        "object_name": object_name,
        "source_date": downloaded_source.source_date.isoformat(),
    }


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> dict[str, str]:
    """Health check."""
    return {"ping": "pong!"}


@app.get("/sync", description="Run sea-ice sync", tags=["Sea Ice"])
def sync_get() -> dict[str, str]:
    """Run a sea-ice sync from a GET request."""
    return _sync()
