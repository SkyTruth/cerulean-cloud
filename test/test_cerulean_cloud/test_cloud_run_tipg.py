"""Public TiPG collections must be explicitly allowlisted."""

import ast
import importlib
import json
from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("allowlists", ["deployed", "missing", "empty"])
async def test_catalog_and_endpoints_use_allowlists(
    postgresql, monkeypatch, allowlists
):
    stack_path = Path(__file__).resolve().parents[2] / "stack/cloud_run_tipg.py"
    for node in ast.walk(ast.parse(stack_path.read_text())):
        if isinstance(node, ast.Call):
            kwargs = {keyword.arg: keyword.value for keyword in node.keywords}
            name = kwargs.get("name")
            if isinstance(name, ast.Constant) and name.value in {
                "TIPG_DB_TABLES",
                "TIPG_DB_FUNCTIONS",
            }:
                if allowlists == "missing":
                    monkeypatch.delenv(name.value, raising=False)
                else:
                    values = (
                        ast.literal_eval(kwargs["value"].args[0])
                        if allowlists == "deployed"
                        else []
                    )
                    monkeypatch.setenv(name.value, json.dumps(values))

    monkeypatch.setenv(
        "DATABASE_URL",
        f"postgresql://{postgresql.info.user}@{postgresql.info.host}:"
        f"{postgresql.info.port}/{postgresql.info.dbname}",
    )
    monkeypatch.setenv("TIPG_DB_ONLY_SPATIAL_TABLES", "false")
    monkeypatch.setenv("TIPG_DB_SPATIAL_EXTENT", "false")
    monkeypatch.setenv("TIPG_DB_DATETIME_EXTENT", "false")
    with postgresql.cursor() as cursor:
        cursor.execute(
            """
            CREATE EXTENSION postgis;
            CREATE TABLE slick_plus (id integer PRIMARY KEY);
            INSERT INTO slick_plus VALUES (1);
            CREATE VIEW source_plus AS SELECT id FROM slick_plus;
            CREATE TABLE users (id integer PRIMARY KEY);
            CREATE TABLE verifications (id integer PRIMARY KEY);
            CREATE TABLE aoi (id integer PRIMARY KEY);
            CREATE TABLE new_private_table (id integer PRIMARY KEY);
            CREATE VIEW new_private_view AS SELECT id FROM new_private_table;
            CREATE FUNCTION get_slick_subclses(cls_id bigint DEFAULT 1)
                RETURNS TABLE(id integer) LANGUAGE sql AS $$ SELECT 1 $$;
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
            assert response.status_code == 200
            expected = (
                {"public.slick_plus", "public.source_plus", "public.get_slick_subclses"}
                if allowlists == "deployed"
                else set()
            )
            assert {item["id"] for item in response.json()["collections"]} == expected
            for collection in expected:
                assert (
                    await client.get(f"/collections/{collection}?f=json")
                ).status_code == 200
            for collection in (
                "public.users",
                "public.verifications",
                "public.aoi",
                "public.new_private_table",
                "public.new_private_view",
                "public.new_private_function",
            ):
                for suffix in (
                    "",
                    "/items",
                    "/queryables",
                    "/tiles/WebMercatorQuad/0/0/0",
                ):
                    response = await client.get(
                        f"/collections/{collection}{suffix}",
                        headers={"X-API-Key": "previous-api-key"},
                    )
                    assert response.status_code == 404
