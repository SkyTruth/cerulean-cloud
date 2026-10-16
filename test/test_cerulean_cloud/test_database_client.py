"""Test database client"""

import json
from datetime import datetime

import geojson
import pytest
import sqlalchemy as sa
from shapely.geometry import box
from sqlalchemy.orm import Session

import cerulean_cloud.database_schema as database_schema
from cerulean_cloud.database_client import DatabaseClient, get_engine
from cerulean_cloud.titiler_client import TitilerClient


def make_model(**overrides):
    """Create a minimally valid model row for the current schema."""
    model_kwargs = {
        "type": "MASKRCNN",
        "file_path": "model_path",
        "layers": ["VV"],
        "cls_map": {"OIL": 1},
        "name": "model_path",
        "tile_width_m": 256,
        "tile_width_px": 256,
        "thresholds": {"OIL": 0.5},
    }
    model_kwargs.update(overrides)
    return database_schema.Model(**model_kwargs)


def test_get_engine(postgresql):
    connection = f"postgresql+asyncpg://{postgresql.info.user}:@{postgresql.info.host}:{postgresql.info.port}/{postgresql.info.dbname}"
    engine = get_engine(connection)
    assert Session(engine)


@pytest.mark.asyncio
async def test_create_model(db_session):
    async with db_session() as session:
        async with session.begin():
            session.add(
                make_model(
                    name="Jane Doe",
                    file_path="true",
                )
            )

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
            sentinel1_grd = await db_client.get_or_insert_sentinel1_grd(
                info["id"],
                info,
                titiler_client.get_base_tile_url(info["id"], rescale=(0, 255)),
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
                    properties={"inf_idx": 1, "machine_confidence": 0.99},
                ),
                geojson.Feature(
                    geometry=box(1, 2, 3, 4),
                    properties={"inf_idx": 2, "machine_confidence": 0.99},
                ),
            ]
        )
        async with db_client.session.begin():
            db_client.session.add_all(
                [database_schema.Trigger(trigger_logs="", trigger_type="MANUAL")]
            )

            with open("test/test_cerulean_cloud/fixtures/productInfo.json") as src:
                info = json.load(src)
            db_client.session.add(
                database_schema.Trigger(trigger_logs="", trigger_type="MANUAL")
            )
            db_client.session.add(make_model())
            sentinel1_grd = await db_client.get_or_insert_sentinel1_grd(
                info["id"],
                info,
                titiler_client.get_base_tile_url(info["id"], rescale=(0, 255)),
            )
            trigger = await db_client.get_trigger(1)
            model = await db_client.get_db_model("model_path")
            orchestrator_run = await db_client.add_orchestrator(
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
            )

        for feat in out_fc.features:
            async with db_client.session.begin():
                slick = await db_client.add_slick(
                    orchestrator_run,
                    sentinel1_grd.start_time,
                    dict(feat).get("geometry"),
                    dict(feat).get("properties").get("inf_idx"),
                    dict(feat).get("properties").get("machine_confidence"),
                    None,
                    None,
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
            db_client.session.add(make_model())
            sentinel1_grd = await db_client.get_or_insert_sentinel1_grd(
                info["id"],
                info,
                titiler_client.get_base_tile_url(info["id"], rescale=(0, 255)),
            )
            trigger = await db_client.get_trigger()
            model = await db_client.get_db_model("model_path")
            orchestrator_run = await db_client.add_orchestrator(
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
