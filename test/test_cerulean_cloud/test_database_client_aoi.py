"""Focused AOI tests for DatabaseClient."""

import pytest
import sqlalchemy as sa
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, box

import cerulean_cloud.database_schema as database_schema
from cerulean_cloud.database_client import (
    AmbiguousAOIError,
    DatabaseClient,
)


@pytest.mark.asyncio
async def test_get_aoi_access_configs_reads_properties_json(db_session):
    async with db_session() as session:
        async with session.begin():
            session.add_all(
                [
                    database_schema.AoiAccessType(
                        id=1,
                        short_name="GCS",
                        prop_keys=["fgb_uri", "pmt_uri", "dataset_version"],
                    ),
                    database_schema.AoiAccessType(
                        id=2,
                        short_name="DB_LOCAL",
                        prop_keys=["table_name", "geog_col", "ext_id_col"],
                    ),
                    database_schema.AoiType(
                        id=1,
                        table_name="aoi_eez",
                        short_name="EEZ",
                        filter_toggle=True,
                        access_type="GCS",
                        properties={
                            "fgb_uri": "gs://cerulean-cloud-aoi/eez-mr/eez_v12.fgb",
                            "pmt_uri": "gs://cerulean-cloud-aoi/eez-mr/eez_v12.pmt",
                            "dataset_version": "2026-04-23",
                        },
                    ),
                    database_schema.AoiType(
                        id=4,
                        table_name="aoi_user",
                        short_name="USER",
                        filter_toggle=False,
                        access_type="DB_LOCAL",
                        properties={
                            "table_name": "aoi_user",
                            "geog_col": "geometry",
                            "ext_id_col": "aoi_id",
                        },
                    ),
                ]
            )

        db_client = DatabaseClient(session.bind)
        db_client.session = session

        configs = await db_client.get_aoi_access_configs()

        assert configs == [
            {
                "key": "EEZ",
                "geometry_source_uri": "gs://cerulean-cloud-aoi/eez-mr/eez_v12.fgb",
                "ext_id_field": "MRGID",
                "name_field": "GEONAME",
                "pmtiles_uri": "gs://cerulean-cloud-aoi/eez-mr/eez_v12.pmt",
                "dataset_version": "2026-04-23",
                "filter_toggle": True,
                "read_perm": None,
            }
        ]

        with pytest.raises(NotImplementedError, match="supports only GCS-backed"):
            await db_client.get_aoi_access_configs(["USER"])


@pytest.mark.asyncio
async def test_resolve_single_aoi_id_raises_on_duplicate_ext_ids(db_session):
    async with db_session() as session:
        geom = MultiPolygon([box(1, 2, 3, 4)])
        async with session.begin():
            session.add(
                database_schema.AoiType(
                    id=1,
                    table_name="aoi_eez",
                    short_name="EEZ",
                )
            )
            session.add_all(
                [
                    database_schema.Aoi(
                        type=1,
                        name="EEZ 1",
                        ext_id="5679",
                        geometry=from_shape(geom),
                    ),
                    database_schema.Aoi(
                        type=1,
                        name="EEZ 2",
                        ext_id="5679",
                        geometry=from_shape(geom),
                    ),
                ]
            )

        db_client = DatabaseClient(session.bind)
        db_client.session = session

        with pytest.raises(AmbiguousAOIError, match="Multiple AOIs found"):
            await db_client.resolve_single_aoi_id("EEZ", "5679")


@pytest.mark.asyncio
async def test_create_user_aoi_inserts_parent_and_child_geometry(db_session):
    async with db_session() as session:
        async with session.begin():
            session.add(
                database_schema.AoiType(
                    id=4,
                    table_name="aoi_user",
                    short_name="USER",
                )
            )
            session.add(
                database_schema.Users(
                    id=1,
                    email="tester@example.com",
                )
            )

        db_client = DatabaseClient(session.bind)
        db_client.session = session

        async with session.begin():
            aoi_user = await db_client.create_user_aoi(
                user_id=1,
                name="Test AOI",
                geometry=box(1, 2, 3, 4),
                ext_id="user-aoi-1",
            )
            aoi_id = getattr(aoi_user, "id", None) or aoi_user.aoi_id

        parent_result = await session.execute(
            sa.text(
                """
                SELECT type, name, ext_id, geometry IS NOT NULL AS has_geometry
                FROM public.aoi
                WHERE id = :aoi_id
                """
            ),
            {"aoi_id": aoi_id},
        )
        parent_row = parent_result.mappings().one()

        child_result = await session.execute(
            sa.text(
                """
                SELECT "user", geometry IS NOT NULL AS has_geometry
                FROM public.aoi_user
                WHERE aoi_id = :aoi_id
                """
            ),
            {"aoi_id": aoi_id},
        )
        child_row = child_result.mappings().one()

        assert parent_row["type"] == 4
        assert parent_row["name"] == "Test AOI"
        assert parent_row["ext_id"] == "user-aoi-1"
        assert parent_row["has_geometry"] is True
        assert child_row["user"] == 1
        assert child_row["has_geometry"] is True
