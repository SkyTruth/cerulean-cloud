"""
pytest configuration for the cerulean_cloud package.
"""

import os
import re
import shutil
import sys
from pathlib import Path

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError
from pytest_postgresql import factories
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

import cerulean_cloud.database_schema as database_schema
from cerulean_cloud.database_client import get_engine


REPO_ROOT = Path(__file__).resolve().parents[2]
CERULEAN_CLOUD_PATH = REPO_ROOT / "cerulean_cloud"
CERU_CI = REPO_ROOT / ".conda" / "ceru-ci"
SEQUENCE_DEFAULT_RE = re.compile(r"nextval\('([^']+)'::regclass\)")


def _prepend_env_path(env_var: str, path: Path) -> None:
    """Ensure `path` is first in a path-like env var without duplicating it."""
    if not path.exists():
        return
    current_value = os.environ.get(env_var, "")
    entries = [entry for entry in current_value.split(os.pathsep) if entry]
    path_str = str(path)
    entries = [entry for entry in entries if entry != path_str]
    os.environ[env_var] = os.pathsep.join([path_str, *entries])


def _ensure_postgis_extension_layout() -> None:
    """
    Conda's macOS `postgis` package can place extension SQL/control files under
    `share/postgresql/extension` while PostgreSQL 18 looks in `share/extension`.
    Mirror those files if needed so `CREATE EXTENSION postgis` works in pytest.
    """
    source_dir = CERU_CI / "share" / "postgresql" / "extension"
    target_dir = CERU_CI / "share" / "extension"
    if (target_dir / "postgis.control").exists() or not (
        source_dir / "postgis.control"
    ).exists():
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("postgis*")):
        target = target_dir / source.name
        if target.exists():
            continue
        relative_source = Path(os.path.relpath(source, start=target.parent))
        try:
            target.symlink_to(relative_source)
        except OSError:
            shutil.copy2(source, target)


_prepend_env_path("PATH", CERU_CI / "bin")
_prepend_env_path("DYLD_FALLBACK_LIBRARY_PATH", CERU_CI / "lib")
_ensure_postgis_extension_layout()

# add stack path to enable relative imports from stack
if str(CERULEAN_CLOUD_PATH) not in sys.path:
    sys.path.append(str(CERULEAN_CLOUD_PATH))


PG_CTL = CERU_CI / "bin" / "pg_ctl"
if PG_CTL.exists():
    postgresql_proc = factories.postgresql_proc(executable=str(PG_CTL))
    postgresql = factories.postgresql("postgresql_proc")


def _iter_metadata_sequence_names() -> list[str]:
    """Extract sequence names referenced only via server_default nextval(...)."""
    sequence_names: list[str] = []
    seen: set[str] = set()
    for table in database_schema.Base.metadata.sorted_tables:
        for column in table.columns:
            default = getattr(column.server_default, "arg", None)
            if default is None:
                continue
            default_text = getattr(default, "text", str(default))
            match = SEQUENCE_DEFAULT_RE.search(default_text)
            if not match:
                continue
            sequence_name = match.group(1)
            if sequence_name in seen:
                continue
            seen.add(sequence_name)
            sequence_names.append(sequence_name)
    return sequence_names


METADATA_SEQUENCE_NAMES = _iter_metadata_sequence_names()
MANAGED_TABLES = [
    table
    for table in database_schema.Base.metadata.sorted_tables
    if table.name != "spatial_ref_sys"
]
POSTGIS_AVAILABLE: bool | None = None


async def _create_metadata_sequences(conn) -> None:
    for sequence_name in METADATA_SEQUENCE_NAMES:
        await conn.execute(sa.text(f'CREATE SEQUENCE IF NOT EXISTS "{sequence_name}"'))


def _is_missing_postgis_error(exc: BaseException) -> bool:
    return 'extension "postgis" is not available' in str(exc)


async def _try_enable_postgis(engine) -> bool:
    if os.environ.get("CERULEAN_TEST_DISABLE_POSTGIS"):
        return False

    async with engine.connect() as conn:
        try:
            await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS postgis"))
            await conn.commit()
            return True
        except DBAPIError as exc:
            await conn.rollback()
            if _is_missing_postgis_error(exc):
                return False
            raise


async def _install_spatial_compat_schema(conn) -> None:
    """
    Install the minimal PostgreSQL schema needed by AOI database tests when the
    runner has PostgreSQL but not PostGIS.

    GitHub-hosted runners expose PostgreSQL binaries without PostGIS. These
    tests still need to exercise asyncpg, SQLAlchemy sessions, PostgreSQL JSONB,
    arrays, conflict handling, and FK-backed data paths there. Real PostGIS runs
    continue through the metadata schema above.
    """
    statements = [
        """
        CREATE OR REPLACE FUNCTION public.ST_GeogFromText(value text)
        RETURNS bytea
        LANGUAGE sql
        IMMUTABLE
        AS $$ SELECT convert_to(value, 'UTF8') $$;
        """,
        """
        CREATE OR REPLACE FUNCTION public.ST_GeomFromEWKT(value text)
        RETURNS bytea
        LANGUAGE sql
        IMMUTABLE
        AS $$ SELECT convert_to(value, 'UTF8') $$;
        """,
        """
        CREATE TABLE public.users (
            id bigserial PRIMARY KEY,
            "firstName" text,
            "lastName" text,
            name text,
            email text NOT NULL UNIQUE,
            "emailVerified" boolean,
            image text,
            role text,
            organization text,
            "organizationType" jsonb,
            location text,
            "emailConsent" boolean,
            banned boolean,
            "banReason" text,
            "banExpires" timestamp without time zone,
            "createdAt" timestamp without time zone DEFAULT now(),
            "updatedAt" timestamp without time zone DEFAULT now()
        )
        """,
        """
        CREATE TABLE public.permission (
            id bigserial PRIMARY KEY,
            short_name text UNIQUE,
            long_name text
        )
        """,
        """
        CREATE TABLE public.aoi_access_type (
            id integer PRIMARY KEY,
            short_name text NOT NULL UNIQUE,
            prop_keys text[] NOT NULL
        )
        """,
        """
        CREATE TABLE public.aoi_type (
            id bigserial PRIMARY KEY,
            table_name text,
            long_name text,
            short_name text NOT NULL UNIQUE,
            source_url text,
            citation text,
            update_time timestamp without time zone DEFAULT now(),
            filter_toggle boolean DEFAULT NULL,
            owner bigint REFERENCES public.users(id),
            read_perm bigint REFERENCES public.permission(id),
            access_type text REFERENCES public.aoi_access_type(short_name),
            properties jsonb DEFAULT NULL
        )
        """,
        """
        CREATE TABLE public.aoi (
            id bigserial PRIMARY KEY,
            type bigint NOT NULL REFERENCES public.aoi_type(id),
            name text NOT NULL,
            ext_id text DEFAULT NULL,
            geometry bytea DEFAULT NULL
        )
        """,
        """
        CREATE TABLE public.aoi_user (
            aoi_id bigint PRIMARY KEY REFERENCES public.aoi(id),
            "user" bigint REFERENCES public.users(id),
            create_time timestamp without time zone DEFAULT now(),
            geometry bytea DEFAULT NULL
        )
        """,
        """
        CREATE TABLE public.trigger (
            id bigserial PRIMARY KEY,
            trigger_time timestamp without time zone NOT NULL DEFAULT now(),
            scene_count integer,
            filtered_scene_count integer,
            trigger_logs text NOT NULL,
            trigger_type varchar(200) NOT NULL
        )
        """,
        """
        CREATE TABLE public.model (
            id integer PRIMARY KEY,
            type text NOT NULL,
            file_path text NOT NULL,
            layers text[] NOT NULL,
            cls_map json NOT NULL,
            name text,
            tile_width_m integer NOT NULL,
            tile_width_px integer NOT NULL,
            epochs integer,
            thresholds json NOT NULL,
            backbone_size integer,
            pixel_f1 double precision,
            instance_f1 double precision,
            update_time timestamp without time zone DEFAULT now()
        )
        """,
        """
        CREATE TABLE public.orchestrator_run (
            id bigserial PRIMARY KEY,
            inference_start_time timestamp without time zone NOT NULL,
            inference_end_time timestamp without time zone NOT NULL,
            base_tiles integer,
            offset_tiles integer,
            git_hash text,
            git_tag varchar(200),
            zoom integer,
            scale integer,
            sea_ice_date date,
            dataset_versions jsonb,
            success boolean,
            inference_run_logs text NOT NULL,
            geometry bytea NOT NULL,
            trigger bigint NOT NULL REFERENCES public.trigger(id),
            model integer NOT NULL REFERENCES public.model(id),
            sentinel1_grd bigint
        )
        """,
        """
        CREATE TABLE public.slick (
            id bigserial PRIMARY KEY,
            slick_timestamp timestamp without time zone NOT NULL,
            geometry bytea NOT NULL,
            active boolean NOT NULL,
            orchestrator_run bigint NOT NULL REFERENCES public.orchestrator_run(id),
            create_time timestamp without time zone NOT NULL DEFAULT now(),
            inference_idx integer NOT NULL,
            cls integer,
            hitl_cls integer,
            machine_confidence double precision,
            precursor_slicks bigint[],
            notes text,
            centerlines json,
            aspect_ratio_factor double precision,
            length double precision,
            area double precision,
            perimeter double precision,
            centroid bytea,
            polsby_popper double precision,
            fill_factor double precision,
            geom_3857_simplified bytea,
            centroid_3857 bytea,
            geom_3857 bytea,
            geometry_count smallint,
            largest_area double precision,
            median_area double precision,
            geometric_slick_potential double precision
        )
        """,
        """
        CREATE TABLE public.slick_to_aoi (
            slick bigint NOT NULL REFERENCES public.slick(id) ON DELETE CASCADE,
            aoi bigint NOT NULL REFERENCES public.aoi(id) ON DELETE CASCADE,
            PRIMARY KEY (slick, aoi)
        )
        """,
    ]
    for statement in statements:
        await conn.execute(sa.text(statement))


@pytest.fixture
def engine(postgresql):
    connection = (
        f"postgresql+asyncpg://{postgresql.info.user}:@"
        f"{postgresql.info.host}:{postgresql.info.port}/{postgresql.info.dbname}"
    )
    return get_engine(connection)


@pytest_asyncio.fixture
async def setup_database(engine):
    global POSTGIS_AVAILABLE
    POSTGIS_AVAILABLE = await _try_enable_postgis(engine)

    async with engine.begin() as conn:
        if POSTGIS_AVAILABLE:
            await _create_metadata_sequences(conn)
            await conn.run_sync(
                lambda sync_conn: database_schema.Base.metadata.create_all(
                    sync_conn, tables=MANAGED_TABLES
                )
            )
        else:
            await _install_spatial_compat_schema(conn)

    yield

    async with engine.begin() as conn:
        if POSTGIS_AVAILABLE:
            await conn.run_sync(
                lambda sync_conn: database_schema.Base.metadata.drop_all(
                    sync_conn, tables=list(reversed(MANAGED_TABLES))
                )
            )
        else:
            await conn.execute(sa.text("DROP SCHEMA public CASCADE"))
            await conn.execute(sa.text("CREATE SCHEMA public"))


@pytest_asyncio.fixture
async def postgis_available(setup_database):
    return POSTGIS_AVAILABLE


@pytest_asyncio.fixture
async def db_session(setup_database, engine):
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield async_session
