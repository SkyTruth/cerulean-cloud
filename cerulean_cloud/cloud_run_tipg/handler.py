"""Cloud run main for tipg

Run this locally with:
uvicorn --port $PORT --host 0.0.0.0 cerulean_cloud.cloud_run_tipg.handler:app

Make sure to set in your environment:
- tipg_NAME
- tipg_TEMPLATES
- DATABASE_URL

"""

import logging
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import jinja2
import pydantic_settings
from fastapi import FastAPI
from fastapi.responses import FileResponse
from mangum import Mangum
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from starlette_cramjam.middleware import CompressionMiddleware
from tipg import __version__ as tipg_version
from tipg.collections import register_collection_catalog
from tipg.database import close_db_connection, connect_to_db
from tipg.errors import DEFAULT_STATUS_CODES, add_exception_handlers
from tipg.factory import Endpoints
from tipg.middleware import CacheControlMiddleware, CatalogUpdateMiddleware
from tipg.settings import APISettings, DatabaseSettings

settings = APISettings()
db_settings = DatabaseSettings()
# Missing allowlists expose no collections. TiPG otherwise discovers everything.
db_settings.tables = db_settings.tables or []
db_settings.functions = db_settings.functions or []


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

app.add_middleware(CacheControlMiddleware, cachecontrol=settings.cachecontrol)
app.add_middleware(CompressionMiddleware)
app.add_middleware(
    CatalogUpdateMiddleware,
    func=register_collection_catalog,
    ttl=300,  # 5 minutes, or adjust as needed
    db_settings=db_settings,  # passed as **kwargs
)
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
