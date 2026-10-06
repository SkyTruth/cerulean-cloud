"""Regression coverage for TiPG catalog refresh without a public admin endpoint."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import tipg.collections

from cerulean_cloud.cloud_run_tipg import handler


@pytest.fixture
def database_mocks(monkeypatch):
    pool = SimpleNamespace(close=AsyncMock())

    async def connect(app, **kwargs):
        app.state.pool = pool

    connect_mock = AsyncMock(side_effect=connect)
    index_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(handler, "connect_to_db", connect_mock)
    monkeypatch.setattr(handler, "close_db_connection", AsyncMock())
    monkeypatch.setattr(tipg.collections, "pg_get_collection_index", index_mock)
    monkeypatch.setenv("SECRET_API_KEY", "test-api-key")
    monkeypatch.setenv("RESTRICTED_COLLECTIONS", '["public.slick"]')
    return connect_mock, index_mock, pool


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "test-api-key"}])
async def test_register_endpoint_is_removed(database_mocks, method, headers):
    connect_mock, index_mock, pool = database_mocks
    async with handler.lifespan(handler.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=handler.app),
            base_url="http://test",
        ) as client:
            for _ in range(3):
                response = await client.request(method, "/register", headers=headers)
                assert response.status_code == 404

        connect_mock.assert_awaited_once()
        index_mock.assert_awaited_once_with(pool, settings=handler.db_settings)
        assert handler.app.state.pool is pool


@pytest.mark.asyncio
async def test_catalog_refresh_reuses_startup_pool(database_mocks):
    connect_mock, index_mock, pool = database_mocks
    new_collection = SimpleNamespace(id="public.slick_plus")
    index_mock.side_effect = [[], [new_collection]]

    async with handler.lifespan(handler.app):
        assert handler.app.state.collection_catalog["collections"] == {}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=handler.app),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"ping": "pong!"}
            index_mock.assert_awaited_once()

            handler.app.state.collection_catalog["last_updated"] = (
                datetime.now() - timedelta(minutes=6)
            )
            response = await client.get("/health")
            assert response.status_code == 200

        assert handler.app.state.collection_catalog["collections"] == {
            "public.slick_plus": new_collection
        }
        assert index_mock.await_count == 2
        index_mock.assert_awaited_with(pool, settings=handler.db_settings)
        connect_mock.assert_awaited_once()
        assert handler.app.state.pool is pool
