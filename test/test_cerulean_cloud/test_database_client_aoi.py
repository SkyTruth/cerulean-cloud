"""Focused AOI tests for DatabaseClient."""

from datetime import datetime

import pandas as pd
import pytest
import sqlalchemy as sa
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, box

import cerulean_cloud.database_schema as database_schema
from cerulean_cloud.database_client import (
    AmbiguousAOIError,
    DatabaseClient,
    InstanceNotFoundError,
    _iter_aoi_match_payload,
)


def _make_model(**overrides):
    model_kwargs = {
        "id": 1,
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


async def _add_slick_fixture(session, slick_id: int = 1):
    geom = box(1, 2, 3, 4)
    session.add(database_schema.Trigger(id=1, trigger_logs="", trigger_type="MANUAL"))
    session.add(_make_model())
    session.add(
        database_schema.OrchestratorRun(
            id=1,
            inference_start_time=datetime(2026, 1, 1),
            inference_end_time=datetime(2026, 1, 1),
            inference_run_logs="",
            geometry=from_shape(geom),
            trigger=1,
            model=1,
        )
    )
    session.add(
        database_schema.Slick(
            id=slick_id,
            slick_timestamp=datetime(2026, 1, 1),
            geometry=from_shape(MultiPolygon([geom])),
            active=True,
            orchestrator_run=1,
            inference_idx=1,
            cls=1,
        )
    )
    await session.flush()


def test_aoi_match_payload_preserves_integer_like_ext_ids_as_text():
    assert list(
        _iter_aoi_match_payload(
            {
                "CORAL": [
                    {"ext_id": 1.0, "name": "Coral 1"},
                    12.0,
                    "007",
                    18.5,
                    None,
                ]
            }
        )
    ) == [
        {
            "aoi_type_short_name": "CORAL",
            "ext_id": "1",
            "name": "Coral 1",
            "is_rich_match": True,
        },
        {
            "aoi_type_short_name": "CORAL",
            "ext_id": "12",
            "name": "12",
            "is_rich_match": False,
        },
        {
            "aoi_type_short_name": "CORAL",
            "ext_id": "007",
            "name": "007",
            "is_rich_match": False,
        },
        {
            "aoi_type_short_name": "CORAL",
            "ext_id": "18.5",
            "name": "18.5",
            "is_rich_match": False,
        },
    ]


@pytest.mark.asyncio
async def test_get_aoi_accessor_configs_reads_properties_json(db_session):
    async with db_session() as session:
        async with session.begin():
            session.add_all(
                [
                    database_schema.AoiAccessType(
                        id=1,
                        short_name="SHARED_DATASET",
                        prop_keys=[
                            "asset_slug",
                            "ext_id_field",
                            "display_name_field",
                        ],
                    ),
                    database_schema.AoiAccessType(
                        id=2,
                        short_name="DB_LOCAL",
                        prop_keys=[
                            "table_name",
                            "geog_col",
                            "ext_id_col",
                            "display_name_field",
                        ],
                    ),
                    database_schema.AoiAccessType(
                        id=3,
                        short_name="DB_REMOTE",
                        prop_keys=[
                            "db_conn_secret_name",
                            "table_name",
                            "geog_col",
                            "ext_id_col",
                            "display_name_field",
                        ],
                    ),
                    database_schema.AoiType(
                        id=1,
                        table_name="aoi_eez",
                        short_name="EEZ",
                        filter_toggle=True,
                        access_type="SHARED_DATASET",
                        properties={
                            "asset_slug": "marine-regions-eez",
                            "ext_id_field": "MRGID",
                            "display_name_field": "GEONAME",
                        },
                    ),
                    database_schema.AoiType(
                        id=2,
                        table_name="aoi_iho",
                        short_name="IHO",
                        filter_toggle=False,
                        access_type="SHARED_DATASET",
                        properties={
                            "asset_slug": "iho-world-seas",
                            "ext_id_field": "MRGID",
                            "display_name_field": "NAME",
                        },
                    ),
                    database_schema.AoiType(
                        id=3,
                        table_name="aoi_mpa",
                        short_name="MPA",
                        filter_toggle=True,
                        access_type="SHARED_DATASET",
                        properties={
                            "asset_slug": "wdpa-marine",
                            "ext_id_field": "SITE_ID",
                            "display_name_field": "NAME",
                        },
                    ),
                    database_schema.AoiType(
                        id=4,
                        table_name="local_aoi",
                        short_name="LOCAL",
                        filter_toggle=False,
                        access_type="DB_LOCAL",
                        properties={
                            "table_name": "local_aoi",
                            "geog_col": "geometry",
                            "ext_id_col": "aoi_id",
                        },
                    ),
                    database_schema.AoiType(
                        id=5,
                        table_name="remote_aoi",
                        short_name="REMOTE",
                        filter_toggle=False,
                        access_type="DB_REMOTE",
                        properties={
                            "db_conn_secret_name": "remote-aoi-db",
                            "table_name": "remote_schema.remote_aoi",
                            "geog_col": "geometry",
                            "ext_id_col": "remote_id",
                            "display_name_field": "remote_name",
                        },
                    ),
                    database_schema.AoiType(
                        id=6,
                        table_name="display_aoi",
                        short_name="DISPLAY",
                        filter_toggle=False,
                        slick_to_aoi_enabled=False,
                        access_type="SHARED_DATASET",
                        properties={
                            "asset_slug": "display-only",
                            "ext_id_field": "display_id",
                            "display_name_field": "display_name",
                        },
                    ),
                ]
            )

        db_client = DatabaseClient(session.bind)
        db_client.session = session

        all_configs = await db_client.get_aoi_accessor_configs()
        assert [config["access_type"] for config in all_configs] == [
            "SHARED_DATASET",
            "SHARED_DATASET",
            "SHARED_DATASET",
            "DB_LOCAL",
            "DB_REMOTE",
            "SHARED_DATASET",
        ]

        scene_configs = await db_client.get_slick_to_aoi_accessor_configs()
        assert [config["short_name"] for config in scene_configs] == [
            "EEZ",
            "IHO",
            "MPA",
            "LOCAL",
            "REMOTE",
        ]

        configs = await db_client.get_aoi_accessor_configs(
            access_types=["SHARED_DATASET"]
        )

        assert [config["short_name"] for config in configs] == [
            "EEZ",
            "IHO",
            "MPA",
            "DISPLAY",
        ]
        config = configs[0]
        assert config == {
            "short_name": "EEZ",
            "access_type": "SHARED_DATASET",
            "properties": {
                "asset_slug": "marine-regions-eez",
                "ext_id_field": "MRGID",
                "display_name_field": "GEONAME",
            },
            "filter_toggle": True,
            "slick_to_aoi_enabled": True,
            "read_perm": None,
        }
        assert configs[-1]["slick_to_aoi_enabled"] is False

        local_configs = await db_client.get_aoi_accessor_configs(["LOCAL"])
        assert local_configs == [
            {
                "short_name": "LOCAL",
                "access_type": "DB_LOCAL",
                "properties": {
                    "table_name": "local_aoi",
                    "geog_col": "geometry",
                    "ext_id_col": "aoi_id",
                },
                "filter_toggle": False,
                "slick_to_aoi_enabled": True,
                "read_perm": None,
            }
        ]

        remote_configs = await db_client.get_aoi_accessor_configs(["REMOTE"])
        assert remote_configs == [
            {
                "short_name": "REMOTE",
                "access_type": "DB_REMOTE",
                "properties": {
                    "db_conn_secret_name": "remote-aoi-db",
                    "table_name": "remote_schema.remote_aoi",
                    "geog_col": "geometry",
                    "ext_id_col": "remote_id",
                    "display_name_field": "remote_name",
                },
                "filter_toggle": False,
                "slick_to_aoi_enabled": True,
                "read_perm": None,
            }
        ]


@pytest.mark.asyncio
async def test_resolve_single_aoi_id_raises_on_duplicate_ext_ids(db_session):
    async with db_session() as session:
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
                    ),
                    database_schema.Aoi(
                        type=1,
                        name="EEZ 2",
                        ext_id="5679",
                    ),
                ]
            )

        db_client = DatabaseClient(session.bind)
        db_client.session = session

        with pytest.raises(AmbiguousAOIError, match="Multiple AOIs found"):
            await db_client.resolve_single_aoi_id("EEZ", "5679")


@pytest.mark.asyncio
async def test_get_or_insert_aoi_upserts_by_type_and_ext_id(db_session):
    async with db_session() as session:
        async with session.begin():
            session.add(
                database_schema.AoiType(
                    id=3,
                    table_name="aoi_mpa",
                    short_name="MPA",
                )
            )
            await session.execute(
                sa.text(
                    "CREATE UNIQUE INDEX uq_test_aoi_type_ext_id "
                    "ON public.aoi(type, ext_id)"
                )
            )

        db_client = DatabaseClient(session.bind)
        db_client.session = session

        async with session.begin():
            first = await db_client.get_or_insert_aoi(
                "MPA",
                "789",
                "MPA One",
            )
            second = await db_client.get_or_insert_aoi(
                "MPA",
                "789",
                "Different Name",
            )

        result = await session.execute(
            sa.text(
                """
                SELECT
                    id,
                    name,
                    geometry IS NULL AS parent_geometry_is_null,
                    COUNT(*) OVER () AS row_count
                FROM public.aoi
                WHERE type = 3 AND ext_id = '789'
                """
            )
        )
        row = result.mappings().one()

        assert first["id"] == second["id"]
        assert row["id"] == first["id"]
        assert row["name"] == "Different Name"
        assert row["parent_geometry_is_null"] is True
        assert row["row_count"] == 1


@pytest.mark.asyncio
async def test_insert_slick_to_aoi_uses_smallest_duplicate_aoi_id(db_session):
    async with db_session() as session:
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
                        name="EEZ duplicate high",
                        ext_id="5679",
                    ),
                    database_schema.Aoi(
                        type=1,
                        name="EEZ duplicate low",
                        ext_id="5679",
                    ),
                ]
            )
            await _add_slick_fixture(session)
            await session.flush()
            aoi_ids = [
                row[0]
                for row in (
                    await session.execute(
                        sa.text(
                            """
                            SELECT id
                            FROM public.aoi
                            WHERE type = 1 AND ext_id = '5679'
                            ORDER BY id
                            """
                        )
                    )
                ).all()
            ]

        db_client = DatabaseClient(session.bind)
        db_client.session = session

        async with session.begin():
            slick_aoi_df = pd.DataFrame(
                [
                    {
                        "slick_id": 1,
                        "aoi_ext_ids": {"EEZ": ["5679"]},
                    }
                ]
            )
            inserted_count = await db_client.insert_slick_to_aoi_from_dataframe(
                slick_aoi_df
            )
            duplicate_inserted_count = (
                await db_client.insert_slick_to_aoi_from_dataframe(slick_aoi_df)
            )

        result = await session.execute(
            sa.text("SELECT slick, aoi FROM public.slick_to_aoi")
        )
        row = result.mappings().one()

        assert inserted_count == 1
        assert duplicate_inserted_count == 0
        assert row["slick"] == 1
        assert row["aoi"] == min(aoi_ids)


@pytest.mark.asyncio
async def test_insert_slick_to_aoi_upserts_rich_aoi_matches(db_session):
    async with db_session() as session:
        async with session.begin():
            session.add(
                database_schema.AoiType(
                    id=3,
                    table_name="aoi_mpa",
                    short_name="MPA",
                )
            )
            await session.execute(
                sa.text(
                    "CREATE UNIQUE INDEX uq_test_aoi_type_ext_id_matches "
                    "ON public.aoi(type, ext_id)"
                )
            )
            await _add_slick_fixture(session)

        db_client = DatabaseClient(session.bind)
        db_client.session = session

        async with session.begin():
            inserted_count = await db_client.insert_slick_to_aoi_from_dataframe(
                pd.DataFrame(
                    [
                        {
                            "slick_id": 1,
                            "aoi_matches": {
                                "MPA": [
                                    {
                                        "ext_id": "789",
                                        "name": "MPA One",
                                    }
                                ]
                            },
                        }
                    ]
                )
            )

        result = await session.execute(
            sa.text(
                """
                SELECT
                    a.type,
                    a.name,
                    a.ext_id,
                    a.geometry IS NULL AS parent_geometry_is_null,
                    sta.slick
                FROM public.slick_to_aoi sta
                JOIN public.aoi a ON a.id = sta.aoi
                """
            )
        )
        row = result.mappings().one()

        assert inserted_count == 1
        assert row["type"] == 3
        assert row["name"] == "MPA One"
        assert row["ext_id"] == "789"
        assert row["parent_geometry_is_null"] is True
        assert row["slick"] == 1


@pytest.mark.asyncio
async def test_insert_slick_to_aoi_legacy_ext_ids_require_existing_aoi(db_session):
    async with db_session() as session:
        async with session.begin():
            session.add(
                database_schema.AoiType(
                    id=3,
                    table_name="aoi_mpa",
                    short_name="MPA",
                )
            )
            await _add_slick_fixture(session)

        db_client = DatabaseClient(session.bind)
        db_client.session = session

        async with session.begin():
            with pytest.raises(InstanceNotFoundError, match="AOI ext_id values"):
                await db_client.insert_slick_to_aoi_from_dataframe(
                    pd.DataFrame(
                        [
                            {
                                "slick_id": 1,
                                "aoi_ext_ids": {"MPA": ["missing"]},
                            }
                        ]
                    )
                )


@pytest.mark.asyncio
async def test_aoi_methods_raise_for_unknown_aoi_type(db_session):
    async with db_session() as session:
        db_client = DatabaseClient(session.bind)
        db_client.session = session

        with pytest.raises(InstanceNotFoundError, match="AOI type not found"):
            await db_client.get_or_insert_aoi(
                "UNKNOWN",
                "1",
                "Unknown AOI",
            )

        with pytest.raises(InstanceNotFoundError, match="AOI type\\(s\\) not found"):
            await db_client.insert_slick_to_aoi_from_dataframe(
                pd.DataFrame(
                    [
                        {
                            "slick_id": 1,
                            "aoi_ext_ids": {"UNKNOWN": ["1"]},
                        }
                    ]
                )
            )
