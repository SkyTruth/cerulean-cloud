"""Cloud Run handler for AOI backfill operations."""

from __future__ import annotations

from typing import Dict

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cerulean_cloud.auth import api_key_auth
from cerulean_cloud.cloud_run_aoi_backfill import service
from cerulean_cloud.cloud_run_aoi_backfill.schema import (
    FinishResponse,
    InspectRequest,
    InspectResponse,
    PrepareRequest,
    PrepareResponse,
    RunRequest,
    RunResponse,
    StatusResponse,
    ValidateResponse,
)


app = FastAPI(title="Cloud Run AOI Backfill", dependencies=[Depends(api_key_auth)])
app.add_middleware(CORSMiddleware, allow_origins=["*"])


def _status_rows(asset_slug: str, short_name: str | None, catalog_source: str | None):
    rows = service.get_backfill_status(
        asset_slug,
        short_name=short_name,
        catalog_source=catalog_source,
    )
    return service.to_plain_python(
        [
            {
                "short_name": row[0],
                "status": row[1],
                "next_slick_id": row[2],
                "max_slick_id_at_start": row[3],
                "slicks_scanned": row[4],
                "matches": row[5],
                "aois_inserted": row[6],
                "links_inserted": row[7],
                "updated_at": row[8],
            }
            for row in rows
        ]
    )


def _validation_rows(asset_slug: str, short_name: str | None, catalog_source: str | None):
    rows = service.validate_backfill(
        asset_slug,
        short_name=short_name,
        catalog_source=catalog_source,
    )
    return service.to_plain_python(
        [
            {"check_name": row[0], "value": row[1]}
            for row in rows
        ]
    )


@app.get("/", description="Health Check", tags=["Health Check"])
def ping() -> Dict[str, str]:
    return {"ping": "pong!"}


@app.post("/inspect", response_model=InspectResponse, tags=["AOI Backfill"])
def inspect(payload: InspectRequest) -> InspectResponse:
    result = service.inspect_asset(
        payload.asset_slug,
        short_name=payload.short_name,
        catalog_source=payload.catalog_source,
        version=payload.version,
        cache_dir=payload.cache_dir,
        force_download=payload.force_download,
        long_name=payload.long_name,
        ext_id_field=payload.ext_id_field,
        display_name_field=payload.display_name_field,
        stage_table=payload.stage_table,
        source_url=payload.source_url,
        citation=payload.citation,
    )
    return InspectResponse(result=result)


@app.post("/prepare", response_model=PrepareResponse, tags=["AOI Backfill"])
def prepare(payload: PrepareRequest) -> PrepareResponse:
    config = service.prepare_backfill(
        payload.asset_slug,
        short_name=payload.short_name,
        catalog_source=payload.catalog_source,
        version=payload.version,
        cache_dir=payload.cache_dir,
        force_download=payload.force_download,
        long_name=payload.long_name,
        ext_id_field=payload.ext_id_field,
        display_name_field=payload.display_name_field,
        stage_table=payload.stage_table,
        source_url=payload.source_url,
        citation=payload.citation,
        batch_size=payload.batch_size,
    )
    return PrepareResponse(
        short_name=config.short_name,
        asset_slug=config.asset_slug,
        stage_table=config.stage_table,
        dataset_version=config.dataset_version,
        batch_size=payload.batch_size,
    )


@app.post("/status", response_model=StatusResponse, tags=["AOI Backfill"])
def status(payload: RunRequest) -> StatusResponse:
    return StatusResponse(
        rows=_status_rows(
            payload.asset_slug, payload.short_name, payload.catalog_source
        )
    )


@app.post("/run", response_model=RunResponse, tags=["AOI Backfill"])
def run(payload: RunRequest) -> RunResponse:
    service.run_backfill(
        payload.asset_slug,
        short_name=payload.short_name,
        catalog_source=payload.catalog_source,
        max_batches=payload.max_batches,
        sleep_seconds=payload.sleep_seconds,
        lock_timeout=payload.lock_timeout,
        statement_timeout=payload.statement_timeout,
    )
    resolved_short_name = payload.short_name or service.slug_to_short_name(
        service.resolve_asset_slug(payload.asset_slug, payload.catalog_source)
    )
    return RunResponse(short_name=resolved_short_name)


@app.post("/validate", response_model=ValidateResponse, tags=["AOI Backfill"])
def validate(payload: RunRequest) -> ValidateResponse:
    return ValidateResponse(
        rows=_validation_rows(
            payload.asset_slug, payload.short_name, payload.catalog_source
        )
    )


@app.post("/finish", response_model=FinishResponse, tags=["AOI Backfill"])
def finish(payload: RunRequest) -> FinishResponse:
    service.finish_backfill(
        payload.asset_slug,
        short_name=payload.short_name,
        catalog_source=payload.catalog_source,
    )
    resolved_short_name = payload.short_name or service.slug_to_short_name(
        service.resolve_asset_slug(payload.asset_slug, payload.catalog_source)
    )
    return FinishResponse(short_name=resolved_short_name)
