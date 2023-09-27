"""Cloud run main for tipg

Run this locally with:
uvicorn --port $PORT --host 0.0.0.0 cerulean_cloud.cloud_run_tipg.handler:app

Make sure to set in your environment:
- tipg_NAME
- tipg_TEMPLATES
- DATABASE_URL

"""
import logging
from typing import Any, List, Optional

import asyncpg
import jinja2
import pydantic
from fastapi import FastAPI
from mangum import Mangum
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from starlette_cramjam.middleware import CompressionMiddleware
from tipg import __version__ as tipg_version
from tipg.collections import register_collection_catalog
from tipg.database import connect_to_db
from tipg.errors import DEFAULT_STATUS_CODES, add_exception_handlers
from tipg.factory import Endpoints
from tipg.middleware import CacheControlMiddleware
from tipg.settings import APISettings, DatabaseSettings

settings = APISettings()
db_settings = DatabaseSettings()


class PostgresSettings(pydantic.BaseSettings):
    """Postgres-specific API settings.

    Note: We can't use PostgresSettings from TiPG because of the weird GCP DB url
          See https://github.com/developmentseed/tipg/issues/32

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


postgres_settings = PostgresSettings()

app = FastAPI(
    title=settings.name,
    version=tipg_version,
    openapi_url="/api",
    docs_url="/api.html",
)

templates_location: List[Any] = [
    jinja2.FileSystemLoader(
        "cerulean_cloud/cloud_run_tipg/templates/"
    ),  # custom template directory
    jinja2.PackageLoader("tipg", "templates"),  # default template directory
]

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
    try:
        await connect_to_db(app, settings=postgres_settings)
        assert getattr(app.state, "pool", None)

        await register_collection_catalog(
            app,
            schemas=db_settings.schemas,
            exclude_table_schemas=db_settings.exclude_table_schemas,
            tables=db_settings.tables,
            exclude_tables=db_settings.exclude_tables,
            exclude_function_schemas=db_settings.exclude_function_schemas,
            functions=db_settings.functions,
            exclude_functions=db_settings.exclude_functions,
            spatial=False,  # False means allow non-spatial tables
        )
    except asyncpg.exceptions.UndefinedObjectError as e:
        # This is the case where TiPG is attempting to start up BEFORE
        # the alembic code has had the opportunity to launch the database
        # You will need to poll the /register endpoint of the tipg URL in order to correctly load the tables
        # i.e. curl https://some-tipg-url.app/register
        app.state.collection_catalog = {}
        print(e)
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Close database connection."""
    if getattr(app.state, "pool", None):
        await app.state.pool.close()


@app.get("/register", include_in_schema=False)
async def register_table(request: Request):
    """Manually register tables"""
    if not getattr(request.app.state, "pool", None):
        await connect_to_db(request.app, settings=postgres_settings)

    assert getattr(request.app.state, "pool", None)
    await register_collection_catalog(
        request.app,
        schemas=db_settings.schemas,
        exclude_table_schemas=db_settings.exclude_table_schemas,
        tables=db_settings.tables,
        exclude_tables=db_settings.exclude_tables,
        exclude_function_schemas=db_settings.exclude_function_schemas,
        functions=db_settings.functions,
        exclude_functions=db_settings.exclude_functions,
        spatial=False,  # False means allow non-spatial tables
    )


@app.get("/healthz", description="Health Check", tags=["Health Check"])
def ping():
    """Health check."""
    return {"ping": "pong!"}


logging.getLogger("mangum.lifespan").setLevel(logging.ERROR)
logging.getLogger("mangum.http").setLevel(logging.ERROR)

handler = Mangum(app, lifespan="auto")
