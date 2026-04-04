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

from cerulean_cloud.cloud_run_sea_ice.logic import (
    DEFAULT_ANCHOR_DATE,
    DEFAULT_CADENCE_DAYS,
    DEFAULT_NHSI_BASE_URL,
    DEFAULT_SIMPLIFY_TOLERANCE,
    build_object_name,
    candidate_source_days,
    nhsi_filename_for_day,
    nhsi_source_url_for_day,
    normalize_mask_polygons,
    should_run_today,
    utc_today,
)

logger = logging.getLogger("cerulean_cloud.cloud_run_sea_ice")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_SIMPLIFY_SRS = "EPSG:4087"
ICE_CLASS_VALUE = 3
PRECISION_GRID_SIZE = 0.0001

app = FastAPI(title="Cloud Run sea-ice sync")


@dataclass(frozen=True)
class SeaIceSettings:
    """Runtime configuration for the sea-ice worker."""

    source_url: str
    mask_gcs_uri: str
    cadence_days: int
    anchor_date: date
    simplify_tolerance: float
    simplify_srs: str
    request_timeout_seconds: int


@dataclass(frozen=True)
class DownloadedSource:
    """Information about the source raster selected for processing."""

    source_date: date
    source_url: str
    gz_path: Path


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> SeaIceSettings:
    """Load worker settings from environment variables."""
    return SeaIceSettings(
        source_url=DEFAULT_NHSI_BASE_URL,
        mask_gcs_uri=_required_env("SEA_ICE_MASK_GCS_URI"),
        cadence_days=DEFAULT_CADENCE_DAYS,
        anchor_date=DEFAULT_ANCHOR_DATE,
        simplify_tolerance=DEFAULT_SIMPLIFY_TOLERANCE,
        simplify_srs=os.getenv("SEA_ICE_SIMPLIFY_SRS", DEFAULT_SIMPLIFY_SRS),
        request_timeout_seconds=int(
            os.getenv("SEA_ICE_REQUEST_TIMEOUT_SECONDS", "600")
        ),
    )


def _download_response_to_file(response, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as dst:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                dst.write(chunk)


def download_latest_source(
    source_root: str,
    workdir: Path,
    today: date,
    timeout_seconds: int,
) -> DownloadedSource:
    """Download the newest available daily NHSI raster within the lookback window."""
    import requests

    session = requests.Session()
    for candidate_day in candidate_source_days(today):
        source_url = nhsi_source_url_for_day(source_root, candidate_day)
        destination = workdir / nhsi_filename_for_day(candidate_day)
        logger.info(
            {
                "message": "Attempting NHSI download",
                "source_date": candidate_day.isoformat(),
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
                _download_response_to_file(response, destination)
            return DownloadedSource(
                source_date=candidate_day,
                source_url=source_url,
                gz_path=destination,
            )
        except requests.HTTPError as exc:
            response = exc.response
            if response is not None and response.status_code == 404:
                continue
            raise
        except requests.RequestException as exc:
            logger.warning(
                {
                    "message": "NHSI download attempt failed",
                    "source_date": candidate_day.isoformat(),
                    "source_url": source_url,
                    "error": str(exc),
                }
            )

    raise FileNotFoundError(
        "No NHSI raster was available for today or the previous two days"
    )


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
    keep_mask = [
        not globe.is_land(point.y, point.x) if point is not None else False
        for point in representative_points
    ]
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
    if "DN" not in gdf.columns:
        raise ValueError("Vectorized raster is missing the DN field")

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


def _sync(force: bool = False) -> dict[str, object]:
    settings = load_settings()
    today = utc_today()
    if not force and not should_run_today(
        today, settings.anchor_date, settings.cadence_days
    ):
        reason = (
            f"Skipped run on {today.isoformat()} because cadence_days="
            f"{settings.cadence_days} and anchor_date={settings.anchor_date.isoformat()}"
        )
        logger.info({"message": reason})
        return {
            "status": "skipped",
            "ran": False,
            "cadence_days": settings.cadence_days,
            "anchor_date": settings.anchor_date.isoformat(),
            "reason": reason,
        }

    workdir = Path("/tmp/sea-ice-sync")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    downloaded_source = download_latest_source(
        settings.source_url,
        workdir,
        today=today,
        timeout_seconds=settings.request_timeout_seconds,
    )
    output_path = process_downloaded_source(
        downloaded_source,
        workdir,
        simplify_tolerance=settings.simplify_tolerance,
        simplify_srs=settings.simplify_srs,
    )

    bucket_name, object_name = build_object_name(
        settings.mask_gcs_uri,
        downloaded_source.source_date,
    )
    metadata = {
        "source_url": downloaded_source.source_url,
        "mask_gcs_uri": settings.mask_gcs_uri,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cadence_days": str(settings.cadence_days),
        "anchor_date": settings.anchor_date.isoformat(),
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
        "ran": True,
        "cadence_days": settings.cadence_days,
        "anchor_date": settings.anchor_date.isoformat(),
        "object_name": object_name,
    }


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> dict[str, str]:
    """Health check."""
    return {"ping": "pong!"}


@app.get("/sync", description="Run sea-ice sync", tags=["Sea Ice"])
def sync_get(force: bool = False) -> dict[str, object]:
    """Run a sea-ice sync from a GET request."""
    return _sync(force=force)
