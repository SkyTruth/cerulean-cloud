#!/usr/bin/env python3
"""
Publish NOAA MASIE daily sea-ice masks as GCS-backed SEA_ICE AOI FlatGeobufs.

The orchestrator reads these assets through aoi_type.short_name='SEA_ICE'. Each
published file is EPSG:4326 and contains MASK_DATE, NAME, and geometry columns.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import geopandas as gpd
import google.auth
import httpx
from google.cloud import storage

LOGGER = logging.getLogger("publish_sea_ice_aoi")
GCS_WRITE_SCOPE = ("https://www.googleapis.com/auth/devstorage.read_write",)
DEFAULT_FGB_URI_TEMPLATE = (
    "gs://cerulean-cloud-aoi/sea-ice/masie_4km/%Y/masie_ice_r00_v01_%Y%j_4km.fgb"
)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def archive_url_for_date(mask_date: date) -> str:
    return (
        "https://noaadata.apps.nsidc.org/NOAA/G02186/shapefiles/4km/"
        f"{mask_date.year}/masie_ice_r00_v01_{mask_date:%Y%j}_4km.zip"
    )


def gcs_client() -> storage.Client:
    credentials, project = google.auth.default(scopes=GCS_WRITE_SCOPE)
    return storage.Client(project=project, credentials=credentials)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got {uri!r}")
    bucket_and_path = uri[len("gs://") :]
    bucket_name, _, object_name = bucket_and_path.partition("/")
    if not bucket_name or not object_name:
        raise ValueError(f"Expected a GCS object URI, got {uri!r}")
    return bucket_name, object_name


def load_masie_gdf(mask_date: date, tmpdir: Path) -> gpd.GeoDataFrame:
    archive_path = tmpdir / f"masie_ice_r00_v01_{mask_date:%Y%j}_4km.zip"
    with httpx.stream(
        "GET",
        archive_url_for_date(mask_date),
        timeout=60,
        follow_redirects=True,
    ) as response:
        response.raise_for_status()
        with archive_path.open("wb") as dst:
            for chunk in response.iter_bytes():
                dst.write(chunk)

    with ZipFile(archive_path) as archive:
        shp_names = [
            name for name in archive.namelist() if name.lower().endswith(".shp")
        ]
        if len(shp_names) != 1:
            raise ValueError(
                f"Expected exactly one shapefile for {mask_date}, found {len(shp_names)}"
            )
        gdf = gpd.read_file(f"zip://{archive_path}!{shp_names[0]}")

    if gdf.crs is None:
        raise ValueError(f"MASIE archive for {mask_date} has no CRS metadata")
    return gdf


def normalize_aoi_gdf(mask_date: date, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[~gdf.geometry.is_empty]
    if gdf.empty:
        raise ValueError(f"MASIE archive for {mask_date} produced no valid geometry")

    mask_date_text = mask_date.isoformat()
    gdf["MASK_DATE"] = mask_date_text
    gdf["NAME"] = f"NOAA MASIE sea ice {mask_date_text}"
    return gdf[["MASK_DATE", "NAME", "geometry"]]


def publish_mask_date(
    mask_date: date,
    fgb_uri_template: str,
    *,
    dry_run: bool,
) -> str:
    output_uri = mask_date.strftime(fgb_uri_template)
    with TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        gdf = normalize_aoi_gdf(mask_date, load_masie_gdf(mask_date, tmpdir))
        local_fgb = tmpdir / f"masie_ice_r00_v01_{mask_date:%Y%j}_4km.fgb"
        gdf.to_file(local_fgb, driver="FlatGeobuf")

        if dry_run:
            LOGGER.info(
                "dry_run mask_date=%s features=%s output_uri=%s",
                mask_date,
                len(gdf),
                output_uri,
            )
            return output_uri

        bucket_name, object_name = parse_gcs_uri(output_uri)
        client = gcs_client()
        client.bucket(bucket_name).blob(object_name).upload_from_filename(local_fgb)
        LOGGER.info(
            "published mask_date=%s features=%s output_uri=%s",
            mask_date,
            len(gdf),
            output_uri,
        )
    return output_uri


def iter_dates(start_date: date, end_date: date):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish MASIE sea-ice masks as SEA_ICE AOI FlatGeobuf assets."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        help="First mask date to publish, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=None,
        help="Last mask date to publish, inclusive. Defaults to --start-date.",
    )
    parser.add_argument(
        "--fgb-uri-template",
        default=DEFAULT_FGB_URI_TEMPLATE,
        help="strftime-compatible gs:// output template.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build but do not upload."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    end_date = args.end_date or args.start_date
    if end_date < args.start_date:
        raise SystemExit("--end-date must be on or after --start-date.")

    for mask_date in iter_dates(args.start_date, end_date):
        publish_mask_date(
            mask_date,
            args.fgb_uri_template,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
