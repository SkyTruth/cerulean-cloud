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


async def _create_metadata_sequences(conn) -> None:
    for sequence_name in METADATA_SEQUENCE_NAMES:
        await conn.execute(sa.text(f'CREATE SEQUENCE IF NOT EXISTS "{sequence_name}"'))


@pytest.fixture
def engine(postgresql):
    connection = (
        f"postgresql+asyncpg://{postgresql.info.user}:@"
        f"{postgresql.info.host}:{postgresql.info.port}/{postgresql.info.dbname}"
    )
    return get_engine(connection)


@pytest_asyncio.fixture
async def setup_database(engine):
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS postgis"))
        await _create_metadata_sequences(conn)
        await conn.run_sync(
            lambda sync_conn: database_schema.Base.metadata.create_all(
                sync_conn, tables=MANAGED_TABLES
            )
        )

    yield

    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: database_schema.Base.metadata.drop_all(
                sync_conn, tables=list(reversed(MANAGED_TABLES))
            )
        )


@pytest_asyncio.fixture
async def db_session(setup_database, engine):
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield async_session
