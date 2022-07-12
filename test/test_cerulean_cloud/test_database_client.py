"""Test database client"""
import json

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, sessionmaker

import cerulean_cloud.database_schema as database_schema
from cerulean_cloud.database_client import DatabaseClient, get_engine
from cerulean_cloud.titiler_client import TitilerClient


def test_get_engine(postgresql):
    connection = f"postgresql+asyncpg://{postgresql.info.user}:@{postgresql.info.host}:{postgresql.info.port}/{postgresql.info.dbname}"
    engine = get_engine(connection)
    assert Session(engine)


@pytest.fixture
def engine(postgresql):
    connection = f"postgresql+asyncpg://{postgresql.info.user}:@{postgresql.info.host}:{postgresql.info.port}/{postgresql.info.dbname}"
    engine = get_engine(connection)
    return engine


@pytest.fixture
async def setup_database(engine):

    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION postgis"))
        await conn.run_sync(database_schema.Base.metadata.create_all)

    yield

    async with engine.begin() as conn:
        await conn.run_sync(database_schema.Base.metadata.drop_all)


@pytest.fixture
async def db_session(setup_database, engine):
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield async_session


@pytest.mark.asyncio
async def test_create_model(db_session):
    async with db_session() as session:
        async with session.begin():
            session.add(database_schema.Model(name="Jane Doe", file_path="true"))

        model = await session.execute(
            sa.select(database_schema.Model).filter_by(name="Jane Doe")
        )
        model = model.scalars().first()

        assert model.name == "Jane Doe"


@pytest.mark.asyncio
async def test_create_s1l(setup_database, engine):
    titiler_client = TitilerClient("some_url")
    async with DatabaseClient(engine) as db_client:
        async with db_client.session.begin():
            with open("test/test_cerulean_cloud/fixtures/productInfo.json") as src:
                info = json.load(src)
            sentinel1_grd = await db_client.get_sentinel1_grd(
                info["id"],
                info,
                titiler_client.get_base_tile_url(info["id"], rescale=(0, 100)),
            )
    assert sentinel1_grd.scene_id == info["id"]
