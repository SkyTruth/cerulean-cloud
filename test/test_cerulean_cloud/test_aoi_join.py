import geopandas as gpd
import pytest
import sqlalchemy as sa
from shapely import wkb
from shapely.geometry import box

from cerulean_cloud.cloud_run_orchestrator.aoi_join import (
    AOIAccessConfig,
    AOIJoiner,
    DbBaseAoiAccessor,
    DbLocalAoiAccessor,
    DbRemoteAoiAccessor,
    GCSAoiAccessor,
    build_aoi_accessor,
)


def test_aoi_access_config_from_properties_is_data_driven():
    config = AOIAccessConfig.from_mapping(
        {
            "short_name": "CUSTOM",
            "access_type": "GCS",
            "filter_toggle": True,
            "read_perm": 3,
            "properties": {
                "fgb_uri": "/tmp/custom.fgb",
                "pmt_uri": "gs://bucket/custom.pmtiles",
                "dataset_version": "custom-v1",
                "ext_id_field": "CUSTOM_ID",
                "display_name_field": "DISPLAY_NAME",
            },
        }
    )

    assert config.key == "CUSTOM"
    assert config.fgb_uri == "/tmp/custom.fgb"
    assert config.ext_id_field == "CUSTOM_ID"
    assert config.name_field == "DISPLAY_NAME"
    assert config.pmtiles_uri == "gs://bucket/custom.pmtiles"
    assert config.dataset_version == "custom-v1"
    assert config.filter_toggle is True
    assert config.read_perm == 3


def test_aoi_access_config_supports_db_local_properties():
    config = AOIAccessConfig.from_mapping(
        {
            "short_name": "USER",
            "access_type": "DB_LOCAL",
            "properties": {
                "table_name": "aoi_user",
                "geog_col": "geometry",
                "ext_id_col": "aoi_id",
            },
        }
    )

    assert config.key == "USER"
    assert config.access_type == "DB_LOCAL"
    assert config.table_name == "aoi_user"
    assert config.geometry_column == "geometry"
    assert config.ext_id_column == "aoi_id"


def test_aoi_access_config_supports_db_remote_properties():
    config = AOIAccessConfig.from_mapping(
        {
            "short_name": "REMOTE",
            "access_type": "DB_REMOTE",
            "properties": {
                "db_conn_str": "postgresql://example/remote",
                "table_name": "remote_schema.remote_aoi",
                "geog_col": "geometry",
                "ext_id_col": "remote_id",
                "name_col": "remote_name",
            },
        }
    )

    assert config.key == "REMOTE"
    assert config.access_type == "DB_REMOTE"
    assert config.db_conn_str == "postgresql://example/remote"
    assert config.table_name == "remote_schema.remote_aoi"
    assert config.geometry_column == "geometry"
    assert config.ext_id_column == "remote_id"
    assert config.name_field == "remote_name"


def test_aoi_access_config_requires_access_type_specific_fields():
    with pytest.raises(ValueError, match="GCS AOI type 'CUSTOM'"):
        AOIAccessConfig.from_mapping({"short_name": "CUSTOM", "access_type": "GCS"})

    with pytest.raises(ValueError, match="DB_LOCAL AOI type 'LOCAL'"):
        AOIAccessConfig.from_mapping({"short_name": "LOCAL", "access_type": "DB_LOCAL"})

    with pytest.raises(ValueError, match="db_conn_str"):
        AOIAccessConfig.from_mapping(
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


def test_aoi_accessor_factory_selects_access_pattern_classes():
    gcs_config = AOIAccessConfig(
        key="CUSTOM",
        access_type="GCS",
        fgb_uri="/tmp/custom.fgb",
        ext_id_field="CUSTOM_ID",
    )
    local_config = AOIAccessConfig(
        key="LOCAL",
        access_type="DB_LOCAL",
        table_name="public.local_aoi",
        geometry_column="geometry",
        ext_id_column="ext_id",
    )
    remote_config = AOIAccessConfig(
        key="REMOTE",
        access_type="DB_REMOTE",
        table_name="remote_schema.remote_aoi",
        geometry_column="geometry",
        ext_id_column="remote_id",
        db_conn_str="postgresql://example/remote",
    )

    assert isinstance(build_aoi_accessor(gcs_config), GCSAoiAccessor)
    assert isinstance(
        build_aoi_accessor(local_config, local_engine=object()), DbLocalAoiAccessor
    )
    assert isinstance(build_aoi_accessor(remote_config), DbRemoteAoiAccessor)

    unknown_config = AOIAccessConfig(key="UNKNOWN", access_type="S3")
    with pytest.raises(NotImplementedError, match="Unsupported AOI access_type"):
        build_aoi_accessor(unknown_config)


def test_db_remote_accessor_fails_before_query_without_connection_string():
    config = AOIAccessConfig(
        key="REMOTE",
        access_type="DB_REMOTE",
        table_name="remote_schema.remote_aoi",
        geometry_column="geometry",
        ext_id_column="remote_id",
    )

    with pytest.raises(ValueError, match="requires db_conn_str"):
        DbRemoteAoiAccessor(config)


@pytest.mark.asyncio
async def test_aoi_joiner_loads_aoi_candidates_once_per_scene(monkeypatch):
    read_calls = []

    def fake_read_file(path, bbox=None):
        read_calls.append({"path": path, "bbox": bbox})
        return gpd.GeoDataFrame(
            {
                "CUSTOM_ID": ["aoi-1", "aoi-2"],
                "DISPLAY_NAME": ["AOI 1", "AOI 2"],
                "geometry": [box(0, 0, 2, 2), box(10, 10, 11, 11)],
            },
            crs="EPSG:4326",
        )

    monkeypatch.setattr(gpd, "read_file", fake_read_file)

    joiner = AOIJoiner(
        scene_bounds=box(-1, -1, 12, 12),
        aoi_access_configs=[
            {
                "short_name": "CUSTOM",
                "access_type": "GCS",
                "properties": {
                    "fgb_uri": "/tmp/custom.fgb",
                    "ext_id_field": "CUSTOM_ID",
                    "display_name_field": "DISPLAY_NAME",
                },
            }
        ],
    )
    slicks = gpd.GeoDataFrame(
        geometry=[box(1, 1, 1.5, 1.5), box(10.2, 10.2, 10.4, 10.4)],
        crs="EPSG:4326",
    )

    assert len(read_calls) == 0
    assert await joiner.compute_aoi_intersect(slicks) == [
        {"CUSTOM": ["aoi-1"]},
        {"CUSTOM": ["aoi-2"]},
    ]
    assert len(read_calls) == 1
    assert read_calls[0]["path"] == "/tmp/custom.fgb"
    assert read_calls[0]["bbox"] == (-1.0, -1.0, 12.0, 12.0)
    matches = await joiner.compute_aoi_matches(slicks)
    assert matches[0]["CUSTOM"][0]["ext_id"] == "aoi-1"
    assert matches[0]["CUSTOM"][0]["name"] == "AOI 1"
    assert matches[0]["CUSTOM"][0]["geometry"].equals(box(0, 0, 2, 2))
    assert matches[1]["CUSTOM"][0]["ext_id"] == "aoi-2"
    assert matches[1]["CUSTOM"][0]["name"] == "AOI 2"
    assert len(read_calls) == 1


def test_aoi_gcs_cache_key_includes_dataset_version(monkeypatch, tmp_path):
    downloaded_paths = []

    class FakeBlob:
        def __init__(self, object_name):
            self.object_name = object_name

        def download_to_filename(self, local_path):
            downloaded_paths.append(local_path)
            with open(local_path, "w") as fp:
                fp.write(self.object_name)

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

    config_v1 = AOIAccessConfig(
        key="CUSTOM",
        fgb_uri="gs://bucket/custom.fgb",
        ext_id_field="CUSTOM_ID",
        dataset_version="v1",
    )
    config_v2 = AOIAccessConfig(
        key="CUSTOM",
        fgb_uri="gs://bucket/custom.fgb",
        ext_id_field="CUSTOM_ID",
        dataset_version="v2",
    )

    accessor_v1 = GCSAoiAccessor(config_v1)
    accessor_v1.cache_dir = tmp_path
    accessor_v2 = GCSAoiAccessor(config_v2)
    accessor_v2.cache_dir = tmp_path

    path_v1 = accessor_v1._download_aoi_dataset()
    path_v2 = accessor_v2._download_aoi_dataset()

    assert path_v1 != path_v2
    assert len(downloaded_paths) == 2


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
    minx, miny, maxx, maxy = tuple(scene_bounds.bounds)
    table_name = accessor.config.table_name
    ext_id_column = accessor.config.ext_id_column
    name_column = accessor.config.name_field or ext_id_column
    geometry_column = accessor.config.geometry_column
    async with engine.connect() as conn:
        result = await conn.execute(
            sa.text(
                f"""
                SELECT
                    {ext_id_column}::text AS ext_id,
                    {name_column}::text AS name,
                    {geometry_column} AS geometry
                FROM {table_name}
                WHERE {ext_id_column} IS NOT NULL
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
        AOIAccessConfig(
            key="LOCAL",
            access_type="DB_LOCAL",
            table_name="public.test_aoi",
            geometry_column="geometry",
            ext_id_column="ext_id",
            name_field="display_name",
        ),
        engine,
    )

    candidates = await accessor.load_candidates(box(-1, -1, 3, 3))

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

    def fake_get_remote_aoi_engine(db_conn_str):
        requested_conn_strings.append(db_conn_str)
        return engine

    monkeypatch.setattr(
        "cerulean_cloud.cloud_run_orchestrator.aoi_join.get_remote_aoi_engine",
        fake_get_remote_aoi_engine,
    )
    accessor = DbRemoteAoiAccessor(
        AOIAccessConfig(
            key="REMOTE",
            access_type="DB_REMOTE",
            table_name="public.remote_test_aoi",
            geometry_column="geometry",
            ext_id_column="ext_id",
            name_field="display_name",
            db_conn_str="postgresql://example/remote",
        )
    )

    candidates = await accessor.load_candidates(box(-1, -1, 3, 3))

    assert requested_conn_strings == ["postgresql://example/remote"]
    assert candidates["ext_id"].tolist() == ["local-1"]
    assert candidates["name"].tolist() == ["Local AOI 1"]


@pytest.mark.asyncio
async def test_aoi_joiner_merges_mixed_accessor_results():
    class FakeAccessor:
        def __init__(self, key, ext_id):
            self.config = AOIAccessConfig(key=key)
            self.ext_id = ext_id

        async def compute_matches(self, slick_gdf, scene_bounds):
            return [
                {
                    self.config.key: [
                        {
                            "ext_id": self.ext_id,
                            "name": self.ext_id.upper(),
                            "geometry": box(0, 0, 1, 1),
                        }
                    ]
                }
                for _ in range(len(slick_gdf))
            ]

    slicks = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    joiner = AOIJoiner(
        scene_bounds=box(-1, -1, 2, 2),
        accessors=[
            FakeAccessor("GCS_LAYER", "gcs-1"),
            FakeAccessor("LOCAL_LAYER", "local-1"),
            FakeAccessor("REMOTE_LAYER", "remote-1"),
        ],
    )

    matches = await joiner.compute_aoi_matches(slicks)

    assert matches == [
        {
            "GCS_LAYER": [
                {"ext_id": "gcs-1", "name": "GCS-1", "geometry": box(0, 0, 1, 1)}
            ],
            "LOCAL_LAYER": [
                {
                    "ext_id": "local-1",
                    "name": "LOCAL-1",
                    "geometry": box(0, 0, 1, 1),
                }
            ],
            "REMOTE_LAYER": [
                {
                    "ext_id": "remote-1",
                    "name": "REMOTE-1",
                    "geometry": box(0, 0, 1, 1),
                }
            ],
        }
    ]
