"""Pure helper logic for the sea-ice sync worker."""

from datetime import date, datetime, timezone
from pathlib import PurePosixPath
from urllib.parse import urlparse


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


def build_object_names(gcs_uri: str, run_date: date) -> tuple[str, str, str]:
    """Build bucket, archive object, and latest object names from a GCS URI."""
    bucket_name, latest_name = parse_gcs_uri(gcs_uri)
    latest_path = PurePosixPath(latest_name)
    archive_dir = latest_path.parent / "archive"
    archive_name = str(archive_dir / f"{run_date.isoformat()}-{latest_path.name}")
    return bucket_name, archive_name, latest_name
