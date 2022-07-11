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
from starlette.templating import Jinja2Templates
from starlette_cramjam.middleware import CompressionMiddleware
from tifeatures import __version__ as tifeatures_version
from tifeatures.db import close_db_connection, connect_to_db, register_table_catalog
from tifeatures.errors import DEFAULT_STATUS_CODES, add_exception_handlers
from tifeatures.factory import Endpoints
from tifeatures.layer import FunctionRegistry
from tifeatures.middleware import CacheControlMiddleware
from tifeatures.settings import APISettings


class PostgresSettings(pydantic.BaseSettings):
    """Postgres-specific API settings.
    Attributes:
        postgres_user: postgres username.
        postgres_pass: postgres password.
        postgres_host: hostname for the connection.
        postgres_port: database port.
        postgres_dbname: database name.
    """

    postgres_user: Optional[str]
    postgres_pass: Optional[str]
    postgres_host: Optional[str]
    postgres_port: Optional[str]
    postgres_dbname: Optional[str]

    database_url: Optional[str] = None

    db_min_conn_size: int = 1
    db_max_conn_size: int = 10
    db_max_queries: int = 50000
    db_max_inactive_conn_lifetime: float = 300

    class Config:
        """model config"""

        env_file = ".env"


settings = APISettings()

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

# We add the function registry to the application state
app.state.tifeatures_function_catalog = FunctionRegistry()

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
    await connect_to_db(app, settings=PostgresSettings())
    await register_table_catalog(app)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Close database connection."""
    await close_db_connection(app)


@app.get("/healthz", description="Health Check", tags=["Health Check"])
def ping():
    """Health check."""
    return {"ping": "pong!"}


logging.getLogger("mangum.lifespan").setLevel(logging.ERROR)
logging.getLogger("mangum.http").setLevel(logging.ERROR)

handler = Mangum(app, lifespan="auto")
