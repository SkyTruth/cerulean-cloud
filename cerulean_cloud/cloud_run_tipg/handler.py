"""Cloud run main for tipg

Run this locally with:
uvicorn --port $PORT --host 0.0.0.0 cerulean_cloud.cloud_run_tipg.handler:app

Make sure to set in your environment:
- tipg_NAME
- tipg_TEMPLATES
- DATABASE_URL

"""

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, List, Optional

import jinja2
import pydantic_settings
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from mangum import Mangum
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette_cramjam.middleware import CompressionMiddleware

from tipg import __version__ as tipg_version
from tipg.collections import register_collection_catalog, Catalog
from tipg.database import close_db_connection, connect_to_db
from tipg.errors import (
    DEFAULT_STATUS_CODES,
    add_exception_handlers,
)
from tipg.factory import Endpoints
from tipg.middleware import CacheControlMiddleware
from tipg.settings import APISettings, DatabaseSettings
from tipg.logger import logger


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
    """
    Turn a list of strings in the .env into a list of strings in the code
    """
    raw_value = os.environ.get(env_var)
    if raw_value is None:
        return default if default is not None else []
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value.split(",")


class CatalogUpdateMiddleware:
    """Middleware to update the catalog cache by calling _initialize_db."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        func: Callable[[FastAPI], Awaitable[None]],
        ttl: int = 300,
    ) -> None:
        self.app = app
        self.func = func  # e.g., _initialize_db
        self.ttl = ttl

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Handle call."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        background: Optional[BackgroundTask] = None

        catalog: Optional[Catalog] = getattr(
            request.app.state, "collection_catalog", None
        )
        last_updated: Optional[datetime] = None

        if catalog:
            last_updated = catalog.get("last_updated")

        should_refresh = (
            not catalog
            or not last_updated
            or datetime.now() > (last_updated + timedelta(seconds=self.ttl))
        )

        if should_refresh:
            logger.debug("Catalog is stale or missing. Triggering background refresh.")
            background = BackgroundTask(self.func, request.app)

        await self.app(scope, receive, send)

        if background:
            await background()


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
        excluded_collections = get_env_list("RESTRICTED_COLLECTIONS")
        if table in excluded_collections:
            # Use something like "from auth import api_key_auth" instead?
            api_key = request.headers.get("X-API-Key")
            if api_key != os.environ.get("SECRET_API_KEY"):
                return JSONResponse(
                    status_code=403,
                    content={
                        "message": f"Access to {table} is restricted.",
                        "request_key": api_key,
                    },
                )
        response = await call_next(request)
        return response


class PostgresSettings(pydantic_settings.BaseSettings):
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

    postgres_user: Optional[str] = None
    postgres_pass: Optional[str] = None
    postgres_host: Optional[str] = None
    postgres_port: Optional[str] = None
    postgres_dbname: Optional[str] = None

    database_url: Optional[str] = None

    db_min_conn_size: int = 1
    db_max_conn_size: int = 10
    db_max_queries: int = 50000
    db_max_inactive_conn_lifetime: float = 300

    class Config:
        """model config"""

        env_file = ".env"


postgres_settings = PostgresSettings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _initialize_db(app)

    yield
    # shutdown
    await close_db_connection(app)
    if getattr(app.state, "pool", None):
        await app.state.pool.close()


app = FastAPI(
    title=settings.name,
    version=tipg_version,
    openapi_url="/api",
    docs_url="/api.html",
    lifespan=lifespan,
)

templates_location: List[Any] = [
    jinja2.FileSystemLoader(
        "cerulean_cloud/cloud_run_tipg/templates/"
    ),  # custom template directory
    jinja2.PackageLoader("tipg", "templates"),  # default template directory
]

templates = Jinja2Templates(directory="cerulean_cloud/cloud_run_tipg/templates/")
templates.env.loader = jinja2.ChoiceLoader(templates_location)

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


@app.get("/register", include_in_schema=False)
async def register_table(request: Request):
    """Manually register tables"""
    await _initialize_db(request.app)
    return {
        "status": "ok",
        "registered": list(request.app.state.collection_catalog.keys()),
    }


async def _initialize_db(app: FastAPI):
    """Common DB setup: connect and register catalog."""
    await connect_to_db(
        app,
        settings=postgres_settings,
        schemas=db_settings.schemas,
    )
    await register_collection_catalog(app, db_settings=db_settings)


app.add_middleware(
    CatalogUpdateMiddleware,
    func=_initialize_db,
    ttl=300,
)


@app.get("/health", description="Health Check", tags=["Health Check"])
def ping():
    """Health check."""
    return {"ping": "pong!"}


@app.get("/robots.txt", description="Robots.txt", tags=["Robots.txt"])
def get_robots_txt():
    """Return the robots.txt file that controls web crawler access to the API.

    This endpoint serves the robots.txt file which provides instructions to web crawlers
    about which parts of the API they are allowed to access. The file should be placed
    in the root directory of the project.

    Returns:
        FileResponse: The robots.txt file content
    """
    return FileResponse("robots.txt")


logging.getLogger("mangum.lifespan").setLevel(logging.ERROR)
logging.getLogger("mangum.http").setLevel(logging.ERROR)

handler = Mangum(app, lifespan="auto")
