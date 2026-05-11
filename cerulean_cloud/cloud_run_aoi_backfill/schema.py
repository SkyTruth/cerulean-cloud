"""Schema for AOI backfill service endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from cerulean_cloud.cloud_run_aoi_backfill import service


class AssetRequest(BaseModel):
    asset_slug: str
    short_name: str | None = None
    catalog_source: str | None = None


class InspectRequest(AssetRequest):
    version: str = "latest"
    cache_dir: str = service.DEFAULT_CACHE_DIR
    force_download: bool = False
    long_name: str | None = None
    ext_id_field: str | None = None
    display_name_field: str | None = None
    stage_table: str | None = None
    source_url: str | None = None
    citation: str | None = None


class PrepareRequest(InspectRequest):
    batch_size: int = 5000


class RunRequest(AssetRequest):
    max_batches: int | None = 25
    sleep_seconds: float = 0.05
    lock_timeout: str = "1s"
    statement_timeout: str = "10min"


class InspectResponse(BaseModel):
    result: dict[str, Any]


class PrepareResponse(BaseModel):
    short_name: str
    asset_slug: str
    stage_table: str
    dataset_version: str
    batch_size: int
    initial_chunk_count: int
    status: str = "prepared"


class StatusResponse(BaseModel):
    rows: list[dict[str, Any]]


class ValidateResponse(BaseModel):
    rows: list[dict[str, Any]]


class RunResponse(BaseModel):
    short_name: str
    status: str = "submitted"


class FinishResponse(BaseModel):
    short_name: str
    status: str = "finished"
