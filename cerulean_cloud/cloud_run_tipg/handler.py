"""Cloud run main for tipg

Run this locally with:
uvicorn --port $PORT --host 0.0.0.0 cerulean_cloud.cloud_run_tipg.handler:app

Make sure to set in your environment:
- tipg_NAME
- tipg_TEMPLATES
- DATABASE_URL

"""
import logging
import os
from typing import Any, List, Optional

import asyncpg
import jinja2
import pydantic
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from starlette.middleware.base import BaseHTTPMiddleware
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


def extract_table_from_request(request: Request) -> Optional[str]:
    """
    Extract the collection ID (table name) from the URL path of an incoming HTTP request.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        Optional[str]: The collection ID if present in the URL path; otherwise, None.

    Example:
        Given a request object with URL 'http://localhost:8000/collections/my_table/items',
        this function will return 'my_table'.
    """
    path_parts = request.url.path.split("/")

    # Check if the request is related to collections
    if "collections" in path_parts:
        idx = path_parts.index("collections")

        # The 'collectionId' should be the segment immediately following 'collections'
        if len(path_parts) > idx + 1:
            return path_parts[idx + 1]

    # Return None if 'collectionId' is not found
    return None


def get_env_list(env_var: str, default: List[str] = None) -> List[str]:
    """Get a list from an environment variable. Assumes values are comma-separated."""
    raw_value = os.environ.get(env_var)
    if raw_value is None:
        return default if default is not None else []
    return raw_value.split(",")


class AccessControlMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle access control based on the collection ID and an API key.

    This middleware calls `extract_table_from_request` to determine the collection ID
    from the request. It then checks if this collection is in the list of excluded collections.
    If so, it verifies the API key in the request headers. If the API key is invalid,
    it raises an HTTP 403 exception.
    """

    async def dispatch(self, request: Request, call_next):
        """
        The dispatch method to handle the request and execute the middleware logic.

        Args:
            request (Request): The incoming FastAPI request object.
            call_next: The next middleware or endpoint in the processing pipeline.

        Raises:
            HTTPException: If the collection is restricted and an invalid API key is provided.

        Returns:
            Response: The outgoing FastAPI response object.
        """
        table = extract_table_from_request(request)
        excluded_collections = get_env_list("TIPG_DB_EXCLUDE_TABLES") + get_env_list(
            "TIPG_DB_EXCLUDE_FUNCTIONS"
        )
        if table in excluded_collections:
            api_key = request.headers.get("X-API-Key")
            if api_key != "XXX_SECRET_API_KEY":
                raise HTTPException(
                    status_code=403, detail="Access to table restricted"
                )
        response = await call_next(request)
        return response


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

# Custom API key checking for restricted access
app.add_middleware(AccessControlMiddleware)

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
    except asyncpg.exceptions.UndefinedObjectError:
        # This is the case where TiPG is attempting to start up BEFORE
        # the alembic code has had the opportunity to launch the database
        # You will need to poll the /register endpoint of the tipg URL in order to correctly load the tables
        # i.e. curl https://some-tipg-url.app/register
        app.state.collection_catalog = {}


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
