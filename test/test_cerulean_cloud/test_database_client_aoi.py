"""Focused AOI tests for DatabaseClient."""

from datetime import datetime
from pathlib import Path

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
        )
    )
    await session.flush()


@pytest.mark.asyncio
async def test_get_aoi_access_configs_reads_properties_json(db_session):
    async with db_session() as session:
        async with session.begin():
            session.add_all(
                [
                    database_schema.AoiAccessType(
                        id=1,
                        short_name="GCS",
                        prop_keys=[
                            "fgb_uri",
                            "pmt_uri",
                            "dataset_version",
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
                        access_type="GCS",
                        properties={
                            "fgb_uri": "gs://cerulean-cloud-aoi/eez-mr/eez_v12.fgb",
                            "pmt_uri": "gs://cerulean-cloud-aoi/eez-mr/eez_v12.pmt",
                            "dataset_version": "2026-04-23",
                            "ext_id_field": "MRGID",
                            "display_name_field": "GEONAME",
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
                ]
            )

        db_client = DatabaseClient(session.bind)
        db_client.session = session

        all_configs = await db_client.get_aoi_access_configs()
        assert [config["access_type"] for config in all_configs] == [
            "GCS",
            "DB_LOCAL",
            "DB_REMOTE",
        ]

        configs = await db_client.get_aoi_access_configs(access_types=["GCS"])

        assert len(configs) == 1
        config = configs[0]
        assert config == {
            "short_name": "EEZ",
            "access_type": "GCS",
            "properties": {
                "fgb_uri": "gs://cerulean-cloud-aoi/eez-mr/eez_v12.fgb",
                "pmt_uri": "gs://cerulean-cloud-aoi/eez-mr/eez_v12.pmt",
                "dataset_version": "2026-04-23",
                "ext_id_field": "MRGID",
                "display_name_field": "GEONAME",
            },
            "filter_toggle": True,
            "read_perm": None,
        }

        local_configs = await db_client.get_aoi_access_configs(["USER"])
        assert local_configs == [
            {
                "short_name": "USER",
                "access_type": "DB_LOCAL",
                "properties": {
                    "table_name": "aoi_user",
                    "geog_col": "geometry",
                    "ext_id_col": "aoi_id",
                },
                "filter_toggle": False,
                "read_perm": None,
            }
        ]

        remote_configs = await db_client.get_aoi_access_configs(["REMOTE"])
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
async def test_create_user_aoi_inserts_child_geometry_only(db_session):
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
                SELECT type, name, ext_id
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
        assert child_row["user"] == 1
        assert child_row["has_geometry"] is True


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
            inserted_count = await db_client.insert_slick_to_aoi_from_dataframe(
                pd.DataFrame(
                    [
                        {
                            "slick_id": 1,
                            "aoi_ext_ids": {"EEZ": ["5679"]},
                        }
                    ]
                )
            )

        result = await session.execute(
            sa.text("SELECT slick, aoi FROM public.slick_to_aoi")
        )
        row = result.mappings().one()

        assert inserted_count == 1
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
async def test_insert_slick_to_aoi_does_not_create_missing_user_aoi(db_session):
    async with db_session() as session:
        async with session.begin():
            session.add(
                database_schema.AoiType(
                    id=4,
                    table_name="aoi_user",
                    short_name="USER",
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
                                "aoi_matches": {
                                    "USER": [
                                        {
                                            "ext_id": "missing-user-aoi",
                                            "name": "Missing user AOI",
                                        }
                                    ]
                                },
                            }
                        ]
                    )
                )


def test_aoi_access_sql_contracts_are_kept_in_sync():
    repo_root = Path(__file__).resolve().parents[2]
    migration_text = (
        repo_root
        / "alembic/versions/1f70e7d0c5b1_add_aoi_access_type_and_dataset_versions.py"
    ).read_text()
    rollback_text = (
        repo_root / "scripts/manual_aoi_ext_id_and_slick_plus_aoi_ids_rollback.sql"
    ).read_text()
    current_branch_sql = [
        (
            repo_root / "scripts/manual_aoi_ext_id_and_slick_plus_aoi_ids.sql"
        ).read_text(),
        migration_text.split("def downgrade():", 1)[0],
    ]
    tipg_text = (repo_root / "stack/cloud_run_tipg.py").read_text()

    for sql_text in current_branch_sql:
        assert "aoi_chunks" not in sql_text
        assert "ck_aoi_type_access_properties" in sql_text
        assert "NULLIF(properties->>'fgb_uri', '') IS NOT NULL" in sql_text
        assert "properties->>'fgb_uri' LIKE 'gs://%'" not in sql_text
        assert "db_conn_str" not in sql_text
        assert "NULLIF(properties->>'db_conn_secret_name', '') IS NOT NULL" in sql_text
        assert "CREATE OR REPLACE VIEW public.aoi_type_public" in sql_text
        assert (
            "ALTER COLUMN table_name DROP NOT NULL" in sql_text
            or '"table_name",\n        existing_type=sa.Text(),\n        nullable=True'
            in sql_text
        )
        assert "ALTER COLUMN geometry DROP NOT NULL" in sql_text
        assert "SET table_name = NULL" not in sql_text
        assert "DROP COLUMN geometry" not in sql_text

        public_view_sql = sql_text.split(
            "CREATE OR REPLACE VIEW public.aoi_type_public", 1
        )[1].split(";", 1)[0]
        assert "read_permission.short_name = 'any'" in public_view_sql
        assert "COALESCE(filter_toggle, FALSE) IS TRUE" not in public_view_sql
        assert "public_properties" not in public_view_sql
        assert "jsonb_build_object" not in public_view_sql
        for sensitive_key in [
            "db_conn_secret_name",
            "fgb_uri",
            "pmt_uri",
            "style",
            "table_name",
            "geog_col",
            "ext_id_col",
        ]:
            assert sensitive_key not in public_view_sql

        aoi_ids_sql = sql_text.split("json_object_agg(aoi_ids.short_name", 1)[1].split(
            ") AS aoi_ids",
            1,
        )[0]
        assert "short_name IN ('EEZ', 'IHO', 'MPA')" not in aoi_ids_sql

    assert "DROP CONSTRAINT IF EXISTS ck_aoi_type_access_properties" in rollback_text
    assert "DROP VIEW IF EXISTS public.aoi_type_public" in rollback_text
    assert "ALTER COLUMN geometry SET NOT NULL" not in rollback_text
    assert "ALTER COLUMN table_name SET NOT NULL" not in rollback_text
    assert "SET table_name = COALESCE" not in rollback_text
    assert "WHEN 'EEZ' THEN 'aoi_eez'" not in rollback_text
    assert (
        "ALTER TABLE public.aoi\n    DROP COLUMN IF EXISTS geometry"
        not in rollback_text
    )

    datetime_table_config = tipg_text.split("for datetime_table in [", 1)[1].split(
        "]", 1
    )[0]
    assert '"aoi_type_public",' in datetime_table_config
    assert '"aoi_type",' not in datetime_table_config
    assert '"public.aoi_type"' in tipg_text


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
                geometry=box(1, 2, 3, 4),
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
