"""Public TiPG collections must be explicitly allowlisted."""

import importlib
from pathlib import Path

import httpx
import pytest

PG_CTL = Path(__file__).resolve().parents[2] / ".conda" / "ceru-ci" / "bin" / "pg_ctl"


@pytest.mark.skipif(
    not PG_CTL.exists(), reason="canonical ceru-ci PostgreSQL pg_ctl is unavailable"
)
@pytest.mark.asyncio
async def test_catalog_uses_allowlists(postgresql, monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        f"postgresql://{postgresql.info.user}@{postgresql.info.host}:"
        f"{postgresql.info.port}/{postgresql.info.dbname}",
    )
    monkeypatch.setenv("TIPG_DB_TABLES", '["public.slick_plus"]')
    monkeypatch.delenv("TIPG_DB_FUNCTIONS", raising=False)
    monkeypatch.setenv("TIPG_DB_ONLY_SPATIAL_TABLES", "false")
    monkeypatch.setenv("TIPG_DB_SPATIAL_EXTENT", "false")
    monkeypatch.setenv("TIPG_DB_DATETIME_EXTENT", "false")
    with postgresql.cursor() as cursor:
        cursor.execute(
            """
            CREATE EXTENSION postgis;
            CREATE TABLE slick_plus (id integer PRIMARY KEY);
            CREATE TABLE new_private_table (id integer PRIMARY KEY);
            CREATE FUNCTION new_private_function(cls_id bigint DEFAULT 1)
                RETURNS TABLE(id integer) LANGUAGE sql AS $$ SELECT 1 $$;
            """
        )
    postgresql.commit()

    from cerulean_cloud.cloud_run_tipg import handler

    handler = importlib.reload(handler)
    async with handler.lifespan(handler.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=handler.app), base_url="http://test"
        ) as client:
            response = await client.get("/collections?f=json")
            assert [c["id"] for c in response.json()["collections"]] == [
                "public.slick_plus"
            ]
            response = await client.get("/collections/public.new_private_table/items")
            assert response.status_code == 404
