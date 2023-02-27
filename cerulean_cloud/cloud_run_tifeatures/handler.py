"""Cloud run main for tifeatures

Run this locally with:
uvicorn --port $PORT --host 0.0.0.0 cerulean_cloud.cloud_run_tifeatures.handler:app

Make sure to set in your environment:
- TIFEATURES_NAME
- TIFEATURES_TEMPLATES
- DATABASE_URL

"""
import logging
from typing import Any, List, Optional

import jinja2
import pydantic
from fastapi import FastAPI
from mangum import Mangum
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from starlette_cramjam.middleware import CompressionMiddleware

from tifeatures import __version__ as tifeatures_version
from tifeatures.db import close_db_connection, connect_to_db, register_table_catalog
from tifeatures.errors import DEFAULT_STATUS_CODES, add_exception_handlers
from tifeatures.factory import Endpoints
from tifeatures.middleware import CacheControlMiddleware
from tifeatures.settings import APISettings, PostgresSettings


settings = APISettings()
postgres_settings = PostgresSettings()

app = FastAPI(
    title=settings.name,
    version=tifeatures_version,
    openapi_url="/api",
    docs_url="/api.html",
)

# custom template directory
templates_location: List[Any] = [
    jinja2.FileSystemLoader("cerulean_cloud/cloud_run_tifeatures/templates/")
]
# default template directory
# templates_location.append(jinja2.PackageLoader(__package__, "templates"))

templates = Jinja2Templates(
    directory="",  # we need to set a dummy directory variable, see https://github.com/encode/starlette/issues/1214
    loader=jinja2.ChoiceLoader(templates_location),
)

# Register endpoints.
endpoints = Endpoints(title=settings.name, templates=templates)
app.include_router(endpoints.router)

# Set all CORS enabled origins
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

app.add_middleware(CacheControlMiddleware, cachecontrol=settings.cachecontrol)
app.add_middleware(CompressionMiddleware)
add_exception_handlers(app, DEFAULT_STATUS_CODES)


@app.on_event("startup")
async def startup_event() -> None:
    """Connect to database on startup."""
    print("using new connection")
    await connect_to_db(app, settings=postgres_settings)
    try:
        await register_table_catalog(
            app,
            schemas=postgres_settings.db_schemas,
            tables=postgres_settings.db_tables,
            spatial=postgres_settings.only_spatial_tables,
        )
    except:  # noqa
        app.state.table_catalog = {}


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Close database connection."""
    await close_db_connection(app)


@app.get("/register", include_in_schema=False)
async def register_table(request: Request):
    """Manually register tables"""
    await register_table_catalog(
        request.app,
        schemas=postgres_settings.db_schemas,
        tables=postgres_settings.db_tables,
        spatial=postgres_settings.only_spatial_tables,
    )


@app.get("/healthz", description="Health Check", tags=["Health Check"])
def ping():
    """Health check."""
    return {"ping": "pong!"}


logging.getLogger("mangum.lifespan").setLevel(logging.ERROR)
logging.getLogger("mangum.http").setLevel(logging.ERROR)

handler = Mangum(app, lifespan="auto")
