"""Cloud Run worker for downloading, processing, and uploading sea-ice extent."""

from __future__ import annotations

import gzip
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
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

app = FastAPI(title="Cloud Run sea-ice sync")


@dataclass(frozen=True)
class DownloadedSource:
    """Information about the source raster selected for processing."""

    source_date: date
    source_url: str
    gz_path: Path


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
    """Download today's NHSI raster."""
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


def gunzip_file(source_path: Path, destination: Path) -> None:
    """Expand a ``.gz`` file to its uncompressed TIFF."""
    with gzip.open(source_path, "rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def warp_raster_to_epsg4326(source_path: Path, destination: Path) -> None:
    """Reproject the raster to EPSG:4326."""
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject

    with rasterio.open(source_path) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            compress="deflate",
            interleave="band",
            predictor=2,
        )

        with rasterio.open(destination, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                )


def raster_to_geojson(source_path: Path, destination: Path) -> None:
    """Polygonize the raster into GeoJSON features with a ``DN`` property."""
    import geopandas as gpd
    import rasterio
    from rasterio import features
    from shapely.geometry import shape

    with rasterio.open(source_path) as src:
        band = src.read(1)
        geoms = []
        vals = []

        for geom, val in features.shapes(
            band,
            mask=None,
            transform=src.transform,
            connectivity=8,
        ):
            geoms.append(shape(geom))
            vals.append(int(val))

        gdf = gpd.GeoDataFrame({"DN": vals}, geometry=geoms, crs=src.crs)

    gdf.to_file(destination, driver="GeoJSON")


def remove_land_parts(gdf):
    """Remove polygons whose representative point falls on land."""
    from global_land_mask import globe

    if gdf.empty:
        return gdf

    representative_points = gdf.geometry.representative_point()
    keep_mask = [not globe.is_land(point.y, point.x) for point in representative_points]
    return gdf.loc[keep_mask].copy()


def filter_and_simplify_geojson(
    source_path: Path,
    destination: Path,
    source_date: date,
    simplify_tolerance: float,
    simplify_srs: str,
) -> None:
    """Filter to sea-ice polygons and simplify them for publication."""
    import geopandas as gpd
    from shapely import make_valid, set_precision

    gdf = gpd.read_file(source_path)
    gdf = gdf[gdf["DN"] == ICE_CLASS_VALUE].copy()
    if gdf.empty:
        gdf["ice_date"] = None
        gdf.to_file(destination, driver="GeoJSON")
        return

    gdf["geometry"] = gdf.geometry.map(make_valid)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["ice_date"] = source_date.isoformat()
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = remove_land_parts(gdf)
    gdf = normalize_mask_polygons(gdf)

    if gdf.empty:
        gdf.to_file(destination, driver="GeoJSON")
        return

    gdf["geometry"] = gdf.geometry.map(
        lambda geom: set_precision(geom, PRECISION_GRID_SIZE)
    )
    simplified = gdf.to_crs(simplify_srs)
    simplified["geometry"] = simplified.geometry.simplify(
        simplify_tolerance,
        preserve_topology=True,
    )
    simplified = simplified.to_crs("EPSG:4326")
    simplified = simplified[
        simplified.geometry.notna() & ~simplified.geometry.is_empty
    ].copy()
    simplified.to_file(destination, driver="GeoJSON")


def process_downloaded_source(
    downloaded_source: DownloadedSource,
    workdir: Path,
    simplify_tolerance: float,
    simplify_srs: str,
) -> Path:
    """Run the NHSI raster processing pipeline and return the final GeoJSON path."""
    uncompressed_tif = workdir / downloaded_source.gz_path.with_suffix("").name
    warped_tif = workdir / f"{uncompressed_tif.stem}_4326.tif"
    raw_geojson = workdir / f"{warped_tif.stem}.geojson"
    output_geojson = workdir / "extent.geojson"

    gunzip_file(downloaded_source.gz_path, uncompressed_tif)
    warp_raster_to_epsg4326(uncompressed_tif, warped_tif)
    raster_to_geojson(warped_tif, raw_geojson)
    filter_and_simplify_geojson(
        raw_geojson,
        output_geojson,
        source_date=downloaded_source.source_date,
        simplify_tolerance=simplify_tolerance,
        simplify_srs=simplify_srs,
    )
    return output_geojson


def upload_outputs(
    bucket_name: str,
    object_name: str,
    source_path: Path,
    metadata: dict[str, str],
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
    )


def _sync() -> dict[str, str]:
    mask_gcs_uri = os.environ["SEA_ICE_MASK_GCS_URI"]
    today = datetime.now(timezone.utc).date()
    workdir = Path("/tmp/sea-ice-sync")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_source = download_latest_source(
            DEFAULT_NHSI_BASE_URL,
            workdir,
            today=today,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        logger.info({"message": str(exc), "source_date": today.isoformat()})
        return {
            "status": "skipped",
            "source_date": today.isoformat(),
            "reason": str(exc),
        }
    output_path = process_downloaded_source(
        downloaded_source,
        workdir,
        simplify_tolerance=DEFAULT_SIMPLIFY_TOLERANCE,
        simplify_srs=DEFAULT_SIMPLIFY_SRS,
    )

    bucket_name, object_prefix = mask_gcs_uri.removeprefix("gs://").split("/", 1)
    object_name = f"{object_prefix.strip('/')}/{downloaded_source.source_date.isoformat()}_extent.geojson"
    metadata = {
        "source_url": downloaded_source.source_url,
        "mask_gcs_uri": mask_gcs_uri,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_date": downloaded_source.source_date.isoformat(),
        "source_filename": downloaded_source.gz_path.name,
    }
    upload_outputs(
        bucket_name=bucket_name,
        object_name=object_name,
        source_path=output_path,
        metadata=metadata,
    )

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
