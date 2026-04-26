from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pytest
import sqlalchemy as sa
from shapely import wkb
from shapely.geometry import box

from cerulean_cloud.cloud_run_orchestrator.aoi_join import (
    AOIJoiner,
    BaseAoiAccessor,
    DbBaseAoiAccessor,
    DbLocalAoiAccessor,
    DbRemoteAoiAccessor,
    GCSAoiAccessor,
    build_aoi_accessor,
)


SCENE_TIME = datetime(2026, 1, 1)


def _scene_bounds(bounds=(-1, -1, 3, 3)):
    return tuple(bounds)


def test_gcs_accessor_owns_gcs_config_parsing():
    accessor = build_aoi_accessor(
        {
            "short_name": "CUSTOM",
            "access_type": "GCS",
            "properties": {
                "fgb_uri": "gs://bucket/custom.fgb",
                "dataset_version": "custom-v1",
                "ext_id_field": "CUSTOM_ID",
                "display_name_field": "DISPLAY_NAME",
            },
        }
    )

    assert isinstance(accessor, GCSAoiAccessor)
    assert accessor.short_name == "CUSTOM"
    assert accessor.fgb_uri == "gs://bucket/custom.fgb"
    assert accessor.ext_id_field == "CUSTOM_ID"
    assert accessor.display_name_field == "DISPLAY_NAME"
    assert accessor.dataset_version == "custom-v1"


def test_db_accessors_own_db_config_parsing():
    local_accessor = build_aoi_accessor(
        {
            "short_name": "USER",
            "access_type": "DB_LOCAL",
            "properties": {
                "table_name": "aoi_user",
                "geog_col": "geometry",
                "ext_id_col": "aoi_id",
            },
        },
        local_engine=object(),
    )
    remote_accessor = build_aoi_accessor(
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
        }
    )

    assert isinstance(local_accessor, DbLocalAoiAccessor)
    assert local_accessor.short_name == "USER"
    assert local_accessor.table_name == "aoi_user"
    assert local_accessor.geog_col == "geometry"
    assert local_accessor.ext_id_col == "aoi_id"

    assert isinstance(remote_accessor, DbRemoteAoiAccessor)
    assert remote_accessor.short_name == "REMOTE"
    assert remote_accessor.db_conn_secret_name == "remote-aoi-db"
    assert remote_accessor.table_name == "remote_schema.remote_aoi"
    assert remote_accessor.geog_col == "geometry"
    assert remote_accessor.ext_id_col == "remote_id"
    assert remote_accessor.display_name_field == "remote_name"


def test_accessors_let_missing_config_fields_raise_at_the_source():
    with pytest.raises(KeyError, match="fgb_uri"):
        build_aoi_accessor({"short_name": "CUSTOM", "access_type": "GCS"})

    with pytest.raises(KeyError, match="table_name"):
        build_aoi_accessor({"short_name": "LOCAL", "access_type": "DB_LOCAL"})

    with pytest.raises(KeyError, match="db_conn_secret_name"):
        build_aoi_accessor(
            {
                "short_name": "REMOTE",
                "access_type": "DB_REMOTE",
                "properties": {
                    "table_name": "remote_schema.remote_aoi",
                    "geog_col": "geometry",
                    "ext_id_col": "remote_id",
                },
            }
        )

    with pytest.raises(ValueError, match="db_conn_secret_name"):
        build_aoi_accessor(
            {
                "short_name": "REMOTE",
                "access_type": "DB_REMOTE",
                "properties": {
                    "db_conn_str": "postgresql://example/remote",
                    "table_name": "remote_schema.remote_aoi",
                    "geog_col": "geometry",
                    "ext_id_col": "remote_id",
                },
            }
        )

    with pytest.raises(NotImplementedError, match="Unsupported AOI access_type"):
        build_aoi_accessor({"short_name": "UNKNOWN", "access_type": "S3"})


@pytest.mark.asyncio
async def test_aoi_joiner_loads_candidates_once_and_skips_null_ext_ids(monkeypatch):
    read_calls = []

    def fake_read_file(path, bbox=None):
        read_calls.append({"path": path, "bbox": bbox})
        return gpd.GeoDataFrame(
            {
                "CUSTOM_ID": ["aoi-1", None, "aoi-2"],
                "DISPLAY_NAME": ["AOI 1", "No ID", "AOI 2"],
                "geometry": [
                    box(0, 0, 2, 2),
                    box(1, 1, 2, 2),
                    box(10, 10, 11, 11),
                ],
            },
            crs="EPSG:4326",
        )

    monkeypatch.setattr(gpd, "read_file", fake_read_file)
    monkeypatch.setattr(
        GCSAoiAccessor,
        "_download_aoi_dataset",
        lambda self: "/tmp/custom.fgb",
    )

    accessor = build_aoi_accessor(
        {
            "short_name": "CUSTOM",
            "access_type": "GCS",
            "properties": {
                "fgb_uri": "gs://bucket/custom.fgb",
                "ext_id_field": "CUSTOM_ID",
                "display_name_field": "DISPLAY_NAME",
            },
        }
    )
    joiner = AOIJoiner(accessors=[accessor])
    slicks = gpd.GeoDataFrame(
        geometry=[box(1, 1, 1.5, 1.5), box(10.2, 10.2, 10.4, 10.4)],
        crs="EPSG:4326",
    )

    matches = await joiner.compute_aoi_matches(
        slicks,
        scene_bounds=_scene_bounds((-1, -1, 12, 12)),
        scene_time=SCENE_TIME,
    )
    assert matches == [
        {"CUSTOM": [{"ext_id": "aoi-1", "name": "AOI 1"}]},
        {"CUSTOM": [{"ext_id": "aoi-2", "name": "AOI 2"}]},
    ]
    assert len(read_calls) == 1
    assert read_calls[0]["path"] == "/tmp/custom.fgb"
    assert read_calls[0]["bbox"] == (-1.0, -1.0, 12.0, 12.0)

    assert (
        await joiner.compute_aoi_matches(
            slicks,
            scene_bounds=_scene_bounds((-1, -1, 12, 12)),
            scene_time=SCENE_TIME,
        )
        == matches
    )
    assert len(read_calls) == 1

    assert await joiner.compute_aoi_matches(
        slicks,
        scene_bounds=_scene_bounds((-1, -1, 12, 12)),
        scene_time=datetime(2026, 1, 2),
    )
    assert len(read_calls) == 2


def test_aoi_gcs_cache_key_includes_dataset_version(monkeypatch, tmp_path):
    downloaded_paths = []

    class FakeBlob:
        def __init__(self, object_name):
            self.object_name = object_name

        def download_to_filename(self, local_path):
            downloaded_paths.append(Path(local_path))
            Path(local_path).write_text(self.object_name)

    class FakeBucket:
        def blob(self, object_name):
            return FakeBlob(object_name)

    class FakeStorageClient:
        def __init__(self, project, credentials):
            pass

        def bucket(self, bucket_name):
            return FakeBucket()

    monkeypatch.setattr(GCSAoiAccessor, "_get_gcs_credentials", lambda self: object())
    monkeypatch.setattr(
        "cerulean_cloud.cloud_run_orchestrator.aoi_join.storage.Client",
        FakeStorageClient,
    )

    row_v1 = {
        "short_name": "CUSTOM",
        "access_type": "GCS",
        "properties": {
            "fgb_uri": "gs://bucket/custom.fgb",
            "ext_id_field": "CUSTOM_ID",
            "dataset_version": "v1",
        },
    }
    row_v2 = {
        "short_name": "CUSTOM",
        "access_type": "GCS",
        "properties": {
            "fgb_uri": "gs://bucket/custom.fgb",
            "ext_id_field": "CUSTOM_ID",
            "dataset_version": "v2",
        },
    }

    accessor_v1 = GCSAoiAccessor(row_v1)
    accessor_v1.cache_dir = tmp_path
    accessor_v2 = GCSAoiAccessor(row_v2)
    accessor_v2.cache_dir = tmp_path

    path_v1 = accessor_v1._download_aoi_dataset()
    path_v2 = accessor_v2._download_aoi_dataset()

    assert path_v1 != path_v2
    assert len(downloaded_paths) == 2
    assert all(not path.exists() for path in downloaded_paths)


def test_aoi_gcs_cache_download_is_atomic(monkeypatch, tmp_path):
    class FailingBlob:
        def download_to_filename(self, local_path):
            Path(local_path).write_text("partial")
            raise RuntimeError("download failed")

    class FakeBucket:
        def blob(self, object_name):
            return FailingBlob()

    class FakeStorageClient:
        def __init__(self, project, credentials):
            pass

        def bucket(self, bucket_name):
            return FakeBucket()

    monkeypatch.setattr(GCSAoiAccessor, "_get_gcs_credentials", lambda self: object())
    monkeypatch.setattr(
        "cerulean_cloud.cloud_run_orchestrator.aoi_join.storage.Client",
        FakeStorageClient,
    )

    accessor = GCSAoiAccessor(
        {
            "short_name": "CUSTOM",
            "access_type": "GCS",
            "properties": {
                "fgb_uri": "gs://bucket/custom.fgb",
                "ext_id_field": "CUSTOM_ID",
            },
        }
    )
    accessor.cache_dir = tmp_path

    with pytest.raises(RuntimeError, match="download failed"):
        accessor._download_aoi_dataset()

    assert list(tmp_path.iterdir()) == []


def test_aoi_gcs_accessor_accepts_local_file_paths(tmp_path):
    local_fgb = tmp_path / "custom.fgb"
    local_fgb.write_text("placeholder")
    accessor = GCSAoiAccessor(
        {
            "short_name": "CUSTOM",
            "access_type": "GCS",
            "properties": {
                "fgb_uri": str(local_fgb),
                "ext_id_field": "CUSTOM_ID",
            },
        }
    )

    assert accessor._download_aoi_dataset() == str(local_fgb)


async def _create_test_aoi_table(engine, table_name="test_aoi", *, postgis=True):
    async with engine.begin() as conn:
        await conn.execute(sa.text(f"DROP TABLE IF EXISTS public.{table_name}"))
        if postgis:
            await conn.execute(
                sa.text(
                    f"""
                    CREATE TABLE public.{table_name} (
                        ext_id text PRIMARY KEY,
                        display_name text,
                        geometry geometry(Polygon, 4326)
                    )
                    """
                )
            )
            await conn.execute(
                sa.text(
                    f"""
                    INSERT INTO public.{table_name}
                        (ext_id, display_name, geometry)
                    VALUES
                        (
                            'local-1',
                            'Local AOI 1',
                            ST_MakeEnvelope(0, 0, 2, 2, 4326)
                        ),
                        (
                            'local-2',
                            'Local AOI 2',
                            ST_MakeEnvelope(10, 10, 11, 11, 4326)
                        )
                    """
                )
            )
            return

        await conn.execute(
            sa.text(
                f"""
                CREATE TABLE public.{table_name} (
                    ext_id text PRIMARY KEY,
                    display_name text,
                    geometry bytea
                )
                """
            )
        )
        await conn.execute(
            sa.text(
                f"""
                INSERT INTO public.{table_name}
                    (ext_id, display_name, geometry)
                VALUES
                    ('local-1', 'Local AOI 1', :geom_1),
                    ('local-2', 'Local AOI 2', :geom_2)
                """
            ),
            {
                "geom_1": wkb.dumps(box(0, 0, 2, 2)),
                "geom_2": wkb.dumps(box(10, 10, 11, 11)),
            },
        )


async def _load_candidates_without_postgis(accessor, engine, scene_bounds):
    minx, miny, maxx, maxy = scene_bounds
    display_col = accessor.display_name_field or accessor.ext_id_col
    async with engine.connect() as conn:
        result = await conn.execute(
            sa.text(
                f"""
                SELECT
                    {accessor.ext_id_col}::text AS ext_id,
                    {display_col}::text AS name,
                    {accessor.geog_col} AS geometry
                FROM {accessor.table_name}
                WHERE {accessor.ext_id_col} IS NOT NULL
                """
            )
        )
        rows = result.mappings().all()

    records = []
    bbox = box(minx, miny, maxx, maxy)
    for row in rows:
        geometry = wkb.loads(bytes(row["geometry"]))
        if geometry.intersects(bbox):
            records.append(
                {
                    "ext_id": row["ext_id"],
                    "name": row["name"] or row["ext_id"],
                    "geometry": geometry,
                }
            )
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


@pytest.mark.asyncio
async def test_db_local_accessor_queries_scene_bbox(
    setup_database, engine, postgis_available, monkeypatch
):
    await _create_test_aoi_table(engine, postgis=postgis_available)
    if not postgis_available:
        monkeypatch.setattr(
            DbBaseAoiAccessor,
            "_load_candidates_from_engine",
            _load_candidates_without_postgis,
        )
    accessor = DbLocalAoiAccessor(
        {
            "short_name": "LOCAL",
            "access_type": "DB_LOCAL",
            "properties": {
                "table_name": "public.test_aoi",
                "geog_col": "geometry",
                "ext_id_col": "ext_id",
                "display_name_field": "display_name",
            },
        },
        engine,
    )

    candidates = await accessor.load_candidates(_scene_bounds(), SCENE_TIME)

    assert candidates["ext_id"].tolist() == ["local-1"]
    assert candidates["name"].tolist() == ["Local AOI 1"]
    assert candidates.crs.to_string() == "EPSG:4326"
    assert candidates.geometry.iloc[0].intersects(box(1, 1, 1.5, 1.5))


@pytest.mark.asyncio
async def test_db_remote_accessor_uses_configured_remote_engine(
    monkeypatch, setup_database, engine, postgis_available
):
    await _create_test_aoi_table(
        engine, table_name="remote_test_aoi", postgis=postgis_available
    )
    if not postgis_available:
        monkeypatch.setattr(
            DbBaseAoiAccessor,
            "_load_candidates_from_engine",
            _load_candidates_without_postgis,
        )
    requested_conn_strings = []
    requested_secret_names = []

    def fake_secret_resolver(secret_name):
        requested_secret_names.append(secret_name)
        return "postgresql://example/remote"

    def fake_engine(self):
        requested_conn_strings.append(self._resolved_db_conn_str())
        return engine

    monkeypatch.setattr(DbRemoteAoiAccessor, "_engine", fake_engine)
    accessor = DbRemoteAoiAccessor(
        {
            "short_name": "REMOTE",
            "access_type": "DB_REMOTE",
            "properties": {
                "table_name": "public.remote_test_aoi",
                "geog_col": "geometry",
                "ext_id_col": "ext_id",
                "display_name_field": "display_name",
                "db_conn_secret_name": "remote-aoi-db",
            },
        },
        secret_resolver=fake_secret_resolver,
    )

    candidates = await accessor.load_candidates(_scene_bounds(), SCENE_TIME)

    assert requested_secret_names == ["remote-aoi-db"]
    assert requested_conn_strings == ["postgresql://example/remote"]
    assert candidates["ext_id"].tolist() == ["local-1"]
    assert candidates["name"].tolist() == ["Local AOI 1"]


@pytest.mark.asyncio
async def test_aoi_joiner_merges_mixed_accessor_results():
    class FakeAccessor(BaseAoiAccessor):
        def __init__(self, short_name, ext_id):
            super().__init__({"short_name": short_name})
            self.ext_id = ext_id

        async def matches_for_scene(self, scene_bounds, scene_time, slick_gdf):
            return [
                {
                    self.short_name: [
                        {"ext_id": self.ext_id, "name": self.ext_id.upper()}
                    ]
                }
                for _ in range(len(slick_gdf))
            ]

    slicks = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    # TODO: add antimeridian bbox-splitting coverage when the AOI join path carries
    # exact scene footprint or split wrapped bounds.
    joiner = AOIJoiner(
        accessors=[
            FakeAccessor("GCS_LAYER", "gcs-1"),
            FakeAccessor("LOCAL_LAYER", "local-1"),
            FakeAccessor("REMOTE_LAYER", "remote-1"),
        ]
    )

    assert await joiner.compute_aoi_matches(
        slicks,
        scene_bounds=_scene_bounds((-1, -1, 2, 2)),
        scene_time=SCENE_TIME,
    ) == [
        {
            "GCS_LAYER": [{"ext_id": "gcs-1", "name": "GCS-1"}],
            "LOCAL_LAYER": [{"ext_id": "local-1", "name": "LOCAL-1"}],
            "REMOTE_LAYER": [{"ext_id": "remote-1", "name": "REMOTE-1"}],
        }
    ]
