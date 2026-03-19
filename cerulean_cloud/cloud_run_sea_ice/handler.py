"""Cloud Run worker for downloading, simplifying, and uploading sea-ice extent."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from fastapi import FastAPI
from google.cloud import storage

from cerulean_cloud.cloud_run_sea_ice.logic import (
    build_object_names,
    should_run_today,
    utc_today,
)
from cerulean_cloud.cloud_run_sea_ice.schema import SyncRequest, SyncResponse

logger = logging.getLogger("cerulean_cloud.cloud_run_sea_ice")
logging.basicConfig(level=logging.INFO, format="%(message)s")

VECTOR_EXTENSIONS = (".shp", ".gpkg", ".geojson", ".json", ".fgb")

app = FastAPI(title="Cloud Run sea-ice sync")


@dataclass(frozen=True)
class SeaIceSettings:
    """Runtime configuration for the sea-ice worker."""

    source_url: str
    mask_gcs_uri: str
    cadence_days: int
    anchor_date: date
    simplify_tolerance: float
    simplify_srs: Optional[str]
    source_dataset: Optional[str]
    source_layer: Optional[str]
    request_timeout_seconds: int


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> SeaIceSettings:
    """Load worker settings from environment variables."""
    cadence_days = int(_required_env("SEA_ICE_CADENCE_DAYS"))
    simplify_tolerance = float(_required_env("SEA_ICE_SIMPLIFY_TOLERANCE"))
    return SeaIceSettings(
        source_url=_required_env("SEA_ICE_SOURCE_URL"),
        mask_gcs_uri=_required_env("SEA_ICE_MASK_GCS_URI"),
        cadence_days=cadence_days,
        anchor_date=date.fromisoformat(_required_env("SEA_ICE_ANCHOR_DATE")),
        simplify_tolerance=simplify_tolerance,
        simplify_srs=os.getenv("SEA_ICE_SIMPLIFY_SRS"),
        source_dataset=os.getenv("SEA_ICE_SOURCE_DATASET"),
        source_layer=os.getenv("SEA_ICE_SOURCE_LAYER"),
        request_timeout_seconds=int(
            os.getenv("SEA_ICE_REQUEST_TIMEOUT_SECONDS", "600")
        ),
    )


def _download_path(workdir: Path, source_url: str) -> Path:
    parsed = urlparse(source_url)
    suffix = Path(parsed.path).suffix or ".dat"
    return workdir / f"source{suffix}"


def download_source(source_url: str, destination: Path, timeout_seconds: int) -> None:
    """Download the upstream sea-ice dataset to a local path."""
    logger.info(
        {
            "message": "Downloading sea-ice source",
            "source_url": source_url,
            "destination": str(destination),
        }
    )
    with requests.get(source_url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with destination.open("wb") as dst:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    dst.write(chunk)


def resolve_input_dataset(
    downloaded_path: Path,
    workdir: Path,
    dataset_hint: Optional[str] = None,
) -> Path:
    """Resolve the local vector dataset path for ogr2ogr."""
    if zipfile.is_zipfile(downloaded_path):
        extract_dir = workdir / "source"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(downloaded_path) as archive:
            archive.extractall(extract_dir)
        if dataset_hint:
            hinted_path = extract_dir / dataset_hint
            if hinted_path.exists():
                return hinted_path
            raise FileNotFoundError(
                f"Configured source dataset not found: {hinted_path}"
            )
        for extension in VECTOR_EXTENSIONS:
            matches = sorted(extract_dir.rglob(f"*{extension}"))
            if matches:
                return matches[0]
        raise FileNotFoundError(
            "No supported vector dataset found inside source archive"
        )

    if dataset_hint:
        hinted_path = workdir / dataset_hint
        if hinted_path.exists():
            return hinted_path
        raise FileNotFoundError(f"Configured source dataset not found: {hinted_path}")
    return downloaded_path


def run_ogr2ogr(
    input_dataset: Path,
    output_dataset: Path,
    simplify_tolerance: float,
    simplify_srs: Optional[str] = None,
    source_layer: Optional[str] = None,
) -> None:
    """Run ogr2ogr to simplify the source geometry and emit GeoJSON."""

    def _run(args: list[str]) -> None:
        logger.info({"message": "Running ogr2ogr", "command": args})
        subprocess.run(args, check=True, capture_output=True, text=True)

    if simplify_srs:
        reprojected_path = output_dataset.parent / "reprojected.gpkg"
        reproject_args = [
            "ogr2ogr",
            "-f",
            "GPKG",
            "-t_srs",
            simplify_srs,
            str(reprojected_path),
            str(input_dataset),
        ]
        if source_layer:
            reproject_args.append(source_layer)
        _run(reproject_args)
        input_dataset = reprojected_path
        source_layer = None

    simplify_args = [
        "ogr2ogr",
        "-f",
        "GeoJSON",
        "-makevalid",
        "-simplify",
        str(simplify_tolerance),
        "-t_srs",
        "EPSG:4326",
        str(output_dataset),
        str(input_dataset),
    ]
    if source_layer:
        simplify_args.append(source_layer)
    _run(simplify_args)


def upload_outputs(
    bucket_name: str,
    archive_object: str,
    latest_object: str,
    source_path: Path,
    metadata: dict[str, str],
) -> None:
    """Upload the archive and latest objects to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for object_name in (archive_object, latest_object):
        blob = bucket.blob(object_name)
        blob.metadata = metadata
        blob.upload_from_filename(
            str(source_path),
            content_type="application/geo+json",
        )


def _sync(payload: SyncRequest) -> SyncResponse:
    settings = load_settings()
    today = utc_today()
    if not payload.force and not should_run_today(
        today, settings.anchor_date, settings.cadence_days
    ):
        reason = (
            f"Skipped run on {today.isoformat()} because cadence_days="
            f"{settings.cadence_days} and anchor_date={settings.anchor_date.isoformat()}"
        )
        logger.info({"message": reason})
        return SyncResponse(
            status="skipped",
            ran=False,
            cadence_days=settings.cadence_days,
            anchor_date=settings.anchor_date.isoformat(),
            reason=reason,
        )

    workdir = Path("/tmp/sea-ice-sync")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    download_path = _download_path(workdir, settings.source_url)
    download_source(
        settings.source_url,
        download_path,
        timeout_seconds=settings.request_timeout_seconds,
    )
    input_dataset = resolve_input_dataset(
        download_path,
        workdir,
        dataset_hint=settings.source_dataset,
    )

    output_path = workdir / "extent.geojson"
    run_ogr2ogr(
        input_dataset=input_dataset,
        output_dataset=output_path,
        simplify_tolerance=settings.simplify_tolerance,
        simplify_srs=settings.simplify_srs,
        source_layer=settings.source_layer,
    )

    bucket_name, archive_object, latest_object = build_object_names(
        settings.mask_gcs_uri, today
    )
    metadata = {
        "source_url": settings.source_url,
        "mask_gcs_uri": settings.mask_gcs_uri,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cadence_days": str(settings.cadence_days),
        "anchor_date": settings.anchor_date.isoformat(),
    }
    upload_outputs(
        bucket_name=bucket_name,
        archive_object=archive_object,
        latest_object=latest_object,
        source_path=output_path,
        metadata=metadata,
    )

    logger.info(
        {
            "message": "Uploaded simplified sea-ice extent",
            "archive_object": archive_object,
            "latest_object": latest_object,
        }
    )
    return SyncResponse(
        status="ok",
        ran=True,
        cadence_days=settings.cadence_days,
        anchor_date=settings.anchor_date.isoformat(),
        archive_object=archive_object,
        latest_object=latest_object,
    )


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> dict[str, str]:
    """Health check."""
    return {"ping": "pong!"}


@app.get("/sync", description="Run sea-ice sync", tags=["Sea Ice"])
def sync_get(force: bool = False) -> SyncResponse:
    """Run a sea-ice sync from a GET request."""
    return _sync(SyncRequest(force=force))


@app.post("/sync", description="Run sea-ice sync", tags=["Sea Ice"])
def sync_post(payload: Optional[SyncRequest] = None) -> SyncResponse:
    """Run a sea-ice sync from a POST request."""
    return _sync(payload or SyncRequest())
