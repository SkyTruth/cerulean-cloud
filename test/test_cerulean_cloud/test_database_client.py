"""Test database client"""
import json
from datetime import datetime

import geojson
import pytest
import sqlalchemy as sa
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, box
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


@pytest.mark.asyncio
async def test_create_slick(setup_database, engine):
    async with DatabaseClient(engine) as db_client:
        titiler_client = TitilerClient("some_url")
        out_fc = geojson.FeatureCollection(
            features=[
                geojson.Feature(
                    geometry=box(1, 2, 3, 4),
                    properties={"classification": 1, "confidence": 0.99},
                ),
                geojson.Feature(
                    geometry=box(1, 2, 3, 4),
                    properties={"classification": 2, "confidence": 0.99},
                ),
            ]
        )
        async with db_client.session.begin():
            geom = MultiPolygon([box(*[1, 2, 3, 4])])
            eezs = [
                database_schema.Eez(mrgid=1, geoname="test", geometry=from_shape(geom)),
                database_schema.Eez(mrgid=1, geoname="test", geometry=from_shape(geom)),
            ]
            db_client.session.add_all(eezs)
            db_client.session.add_all(
                [database_schema.Trigger(trigger_logs="", trigger_type="MANUAL")]
            )

            with open("test/test_cerulean_cloud/fixtures/productInfo.json") as src:
                info = json.load(src)
            db_client.session.add(
                database_schema.Trigger(trigger_logs="", trigger_type="MANUAL")
            )
            db_client.session.add(
                database_schema.Model(file_path="model_path", name="model_path")
            )
            sentinel1_grd = await db_client.get_sentinel1_grd(
                info["id"],
                info,
                titiler_client.get_base_tile_url(info["id"], rescale=(0, 100)),
            )
            trigger = await db_client.get_trigger(1)
            model = await db_client.get_model("model_path")
            orchestrator_run = db_client.add_orchestrator(
                datetime.now(),
                datetime.now(),
                1,
                1,
                "",
                "",
                "",
                1,
                1,
                [1, 2, 3, 4],
                trigger,
                model,
                sentinel1_grd,
                None,
                None,
            )

        for feat in out_fc.features:
            async with db_client.session.begin():
                slick = await db_client.add_slick_with_eez(
                    dict(feat), orchestrator_run, sentinel1_grd.start_time
                )
                print(f"Added last eez for slick {slick}")

        slicks = await db_client.session.execute(sa.select(database_schema.Slick))
        all_slicks = slicks.scalars().all()
        assert len(all_slicks) == 2
        assert all_slicks[0].machine_confidence == 0.99


@pytest.mark.asyncio
async def test_update_orchestrator(setup_database, engine):
    titiler_client = TitilerClient("some_url")
    async with DatabaseClient(engine) as db_client:
        async with db_client.session.begin():
            with open("test/test_cerulean_cloud/fixtures/productInfo.json") as src:
                info = json.load(src)
            db_client.session.add(
                database_schema.Trigger(trigger_logs="", trigger_type="MANUAL")
            )
            db_client.session.add(
                database_schema.Model(file_path="model_path", name="model_path")
            )
            sentinel1_grd = await db_client.get_sentinel1_grd(
                info["id"],
                info,
                titiler_client.get_base_tile_url(info["id"], rescale=(0, 100)),
            )
            trigger = await db_client.get_trigger()
            model = await db_client.get_model("model_path")
            orchestrator_run = db_client.add_orchestrator(
                datetime.now(),
                datetime.now(),
                1,
                1,
                "",
                "",
                "",
                1,
                1,
                [1, 2, 3, 4],
                trigger,
                model,
                sentinel1_grd,
                None,
                None,
            )
            db_client.session.add(orchestrator_run)

        async with db_client.session.begin():
            orchestrator_run.success = True
            orchestrator_run.inference_end_time = datetime.now()

        o_r = await db_client.session.execute(
            sa.select(database_schema.OrchestratorRun).filter_by(id=orchestrator_run.id)
        )
        o_r = o_r.scalars().first()
        assert o_r.success is True
