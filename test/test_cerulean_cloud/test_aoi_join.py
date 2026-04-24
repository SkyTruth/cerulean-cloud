import geopandas as gpd
from shapely.geometry import box

from cerulean_cloud.cloud_run_orchestrator.aoi_join import AOIAccessConfig, AOIJoiner


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
    assert config.geometry_source_uri == "/tmp/custom.fgb"
    assert config.ext_id_field == "CUSTOM_ID"
    assert config.name_field == "DISPLAY_NAME"
    assert config.pmtiles_uri == "gs://bucket/custom.pmtiles"
    assert config.dataset_version == "custom-v1"
    assert config.filter_toggle is True
    assert config.read_perm == 3


def test_aoi_joiner_loads_aoi_candidates_once_per_scene(monkeypatch):
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

    assert len(read_calls) == 1
    assert read_calls[0]["path"] == "/tmp/custom.fgb"
    assert read_calls[0]["bbox"] == (-1.0, -1.0, 12.0, 12.0)
    assert joiner.compute_aoi_intersect(slicks) == [
        {"CUSTOM": ["aoi-1"]},
        {"CUSTOM": ["aoi-2"]},
    ]
    matches = joiner.compute_aoi_matches(slicks)
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

    monkeypatch.setattr(AOIJoiner, "_get_gcs_credentials", lambda self: object())
    monkeypatch.setattr(
        "cerulean_cloud.cloud_run_orchestrator.aoi_join.storage.Client",
        FakeStorageClient,
    )

    joiner = object.__new__(AOIJoiner)
    joiner.cache_dir = tmp_path
    joiner.gcp_project = None

    config_v1 = AOIAccessConfig(
        key="CUSTOM",
        geometry_source_uri="gs://bucket/custom.fgb",
        ext_id_field="CUSTOM_ID",
        dataset_version="v1",
    )
    config_v2 = AOIAccessConfig(
        key="CUSTOM",
        geometry_source_uri="gs://bucket/custom.fgb",
        ext_id_field="CUSTOM_ID",
        dataset_version="v2",
    )

    path_v1 = joiner._download_aoi_dataset(config_v1)
    path_v2 = joiner._download_aoi_dataset(config_v2)

    assert path_v1 != path_v2
    assert len(downloaded_paths) == 2
