"""Pure helper logic for the sea-ice sync worker."""

from datetime import date, datetime, timezone
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


def build_object_name(gcs_uri: str) -> tuple[str, str]:
    """Build the bucket and object name from a GCS URI."""
    return parse_gcs_uri(gcs_uri)
