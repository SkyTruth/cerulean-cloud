import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKFILL_SQL = REPO_ROOT / "scripts/backfill_shared_dataset_aoi.sql"
BACKFILL_SERVICE = REPO_ROOT / "cerulean_cloud/cloud_run_aoi_backfill/service.py"


def load_wrapper_module():
    spec = importlib.util.spec_from_file_location(
        "backfill_shared_dataset_aoi",
        BACKFILL_SERVICE,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shared_dataset_aoi_backfill_sql_keeps_online_safety_contract():
    sql_text = BACKFILL_SQL.read_text()

    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_slick_to_aoi_slick" in sql_text
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_slick_to_aoi_aoi" in sql_text
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_aoi_type_ext_id" in sql_text
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_slick_geom" in sql_text
    assert "CREATE SCHEMA IF NOT EXISTS maintenance" in sql_text
    assert "Staging table must not live in public" in sql_text
    assert "SET LOCAL lock_timeout" in sql_text
    assert "SET LOCAL statement_timeout" in sql_text
    assert "ST_Subdivide" in sql_text
    assert "ON CONFLICT DO NOTHING" in sql_text
    assert "ST_Intersects" in sql_text
    assert "cleanup_shared_dataset_aoi_backfills" in sql_text
    assert "split_required" in sql_text
    assert "DROP TABLE IF EXISTS" in sql_text
    assert "ALTER TABLE maintenance.shared_dataset_aoi_backfill_chunk" in sql_text
    assert "ADD COLUMN IF NOT EXISTS runtime_seconds double precision" in sql_text
    assert (
        "ADD COLUMN IF NOT EXISTS slick_to_aoi_buffer_m double precision NOT NULL DEFAULT 0"
        in sql_text
    )
    assert (
        "sub_batches integer NOT NULL DEFAULT 0,\n    runtime_seconds double precision,"
        in sql_text
    )
    assert "geom geometry(MultiPolygon, 4326) NOT NULL\n        )" in sql_text

    forbidden_fragments = [
        "TRUNCATE public.",
        "DROP TABLE public.",
        "LOCK TABLE public.slick",
        "LOCK TABLE public.slick_to_aoi",
        "LOCK TABLE public.aoi",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in sql_text


def test_shared_dataset_aoi_backfill_keeps_aoi_hidden_until_manual_toggle():
    sql_text = BACKFILL_SQL.read_text()

    preparation_sql = sql_text.split(
        "CREATE OR REPLACE FUNCTION maintenance.process_shared_dataset_aoi_backfill_chunk",
        1,
    )[0]
    finish_sql = sql_text.split(
        "CREATE OR REPLACE PROCEDURE maintenance.finish_shared_dataset_aoi_backfill", 1
    )[1]

    assert "filter_toggle," in preparation_sql
    assert "read_perm," in preparation_sql
    assert "access_type," in preparation_sql
    assert "FALSE," in preparation_sql
    assert "v_read_perm_id," in preparation_sql
    assert "'SHARED_DATASET'" in preparation_sql
    assert "filter_toggle = FALSE" in preparation_sql
    assert "access_type = 'SHARED_DATASET'" in preparation_sql

    assert "filter_toggle = FALSE" in finish_sql
    assert "manual UI enablement" in finish_sql
    assert "DROP TABLE IF EXISTS" in finish_sql


def test_shared_dataset_aoi_backfill_sql_snapshots_and_uses_buffer():
    sql_text = BACKFILL_SQL.read_text()

    assert "slick_to_aoi_buffer_m double precision NOT NULL DEFAULT 0" in sql_text
    assert "(properties->>'slick_to_aoi_buffer_m')::double precision" in sql_text
    assert "r.slick_to_aoi_buffer_m" in sql_text
    assert "ST_Buffer(" in sql_text
    assert "ST_Transform(geom, 8857)" in sql_text
    assert "s.geometry::geometry && COALESCE(" in sql_text
    assert "ST_MakeEnvelope(p_minx, p_miny, p_maxx, p_maxy, 4326)" in sql_text
    assert "WHERE ST_Intersects(slick_geom, aoi_geom)" in sql_text


def test_backfill_wrapper_derives_safe_names_from_asset_slug():
    wrapper = load_wrapper_module()

    assert (
        wrapper.slug_to_short_name("gfw-fixed-infrastructure")
        == "GFW_FIXED_INFRASTRUCTURE"
    )
    assert wrapper.slug_to_short_name("2026-demo") == "AOI_2026_DEMO"
    assert wrapper.slug_to_stage_table("GFW_FIXED_INFRASTRUCTURE") == (
        "maintenance.aoi_stage_gfw_fixed_infrastructure"
    )
    assert len(wrapper.slug_to_stage_table("X" * 100).split(".")[1]) <= 63


def test_backfill_wrapper_resolves_gcs_uri_to_asset_slug_from_catalog():
    wrapper = load_wrapper_module()
    csv_text = "\n".join(
        [
            "asset_slug,title,category,subcategory,status,access_tier,owner,"
            "update_cadence,canonical_path,canonical_format,available_formats,"
            "metadata_paths,has_pmtiles,has_geojson,has_csv,source,license,notes",
            "petrodata,PETRODATA Petroleum Fields,300,310,active,public,"
            "SkyTruth,manual,gs://bucket/path/petrodata/latest/petrodata.fgb,"
            "fgb,fgb;pmtiles,README.md,true,false,false,source,license,notes",
        ]
    )
    with TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "catalog.csv"
        catalog_path.write_text(csv_text)

        assert (
            wrapper.resolve_asset_slug(
                "gs://bucket/path/petrodata/latest/petrodata.fgb",
                str(catalog_path),
            )
            == "petrodata"
        )


def test_backfill_wrapper_derives_catalog_metadata():
    wrapper = load_wrapper_module()
    csv_text = "\n".join(
        [
            "asset_slug,title,category,subcategory,status,access_tier,owner,"
            "update_cadence,canonical_path,canonical_format,available_formats,"
            "metadata_paths,has_pmtiles,has_geojson,has_csv,source,license,citation,notes",
            "petrodata,PETRODATA Petroleum Fields,300,310,active,public,"
            "SkyTruth,manual,gs://bucket/path/petrodata/latest/petrodata.fgb,"
            "fgb,fgb;pmtiles,README.md,true,false,false,source summary,"
            "follow source terms,PRIO PETRODATA v1.2,notes",
        ]
    )
    with TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "catalog.csv"
        catalog_path.write_text(csv_text)
        asset = wrapper.get_catalog_asset("petrodata", str(catalog_path))

        assert wrapper.derive_catalog_citation(asset) == "PRIO PETRODATA v1.2"
        assert (
            wrapper.normalize_dataset_version("petrodata", "petrodata@", "latest") == ""
        )


def test_backfill_wrapper_citation_is_empty_when_missing():
    wrapper = load_wrapper_module()
    csv_text = "\n".join(
        [
            "asset_slug,title,category,subcategory,status,access_tier,owner,"
            "update_cadence,canonical_path,canonical_format,available_formats,"
            "metadata_paths,has_pmtiles,has_geojson,has_csv,source,license,citation,notes",
            "petrodata,PETRODATA Petroleum Fields,300,310,active,public,"
            "SkyTruth,manual,gs://bucket/path/petrodata/latest/petrodata.fgb,"
            "fgb,fgb;pmtiles,README.md,true,false,false,PRIO PETRODATA v1.2,"
            "follow source terms,,notes",
        ]
    )
    with TemporaryDirectory() as tmp_dir:
        catalog_path = Path(tmp_dir) / "catalog.csv"
        catalog_path.write_text(csv_text)
        asset = wrapper.get_catalog_asset("petrodata", str(catalog_path))

        assert wrapper.derive_catalog_citation(asset) == ""


def test_backfill_wrapper_inspects_and_enforces_dataset_size(tmp_path):
    wrapper = load_wrapper_module()
    path = tmp_path / "dataset.fgb"
    path.touch()
    path.open("r+b").truncate(wrapper.LARGE_DATASET_WARN_BYTES + 1)

    result = wrapper.inspect_dataset_size(path)

    assert result["dataset_size_bytes"] == path.stat().st_size
    assert "dataset_size_warning" in result
    assert "chunked AOI staging" in result["dataset_size_warning"]


def test_backfill_wrapper_builds_chunk_plan(monkeypatch, tmp_path):
    wrapper = load_wrapper_module()
    path = tmp_path / "dataset.fgb"
    path.touch()
    path.open("r+b").truncate(wrapper.TARGET_CHUNK_BYTES * 5)

    monkeypatch.setattr(
        wrapper,
        "_dataset_metadata",
        lambda _: {
            "feature_count": wrapper.TARGET_CHUNK_FEATURES * 2,
            "bounds": (0.0, 0.0, 10.0, 10.0),
            "crs": "EPSG:4326",
        },
    )

    result = wrapper.build_chunk_plan(path)

    assert result["grid_side"] >= 2
    assert result["target_chunk_count"] == len(result["chunks"])
    assert result["target_chunk_count"] >= 4


def test_backfill_wrapper_splits_chunk_bbox():
    wrapper = load_wrapper_module()

    children = wrapper.split_chunk_bbox(
        wrapper.ChunkSpec(
            chunk_index=1,
            minx=0.0,
            miny=0.0,
            maxx=8.0,
            maxy=4.0,
            split_depth=0,
        )
    )

    assert len(children) == 4
    assert {child.split_depth for child in children} == {1}


def test_backfill_wrapper_infers_fields_when_unambiguous():
    wrapper = load_wrapper_module()

    assert wrapper.infer_ext_id_field(["Name", "MRGID", "geometry"]) == "MRGID"
    assert wrapper.infer_ext_id_field(["source_layer", "PRIMKEY", "NAME"]) == "PRIMKEY"
    assert wrapper.infer_display_name_field(["Name", "MRGID"], "MRGID") == "Name"
    assert wrapper.infer_display_name_field(["MRGID"], "MRGID") is None


def test_backfill_wrapper_strips_nul_bytes_from_stage_text():
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    wrapper = load_wrapper_module()
    gdf = geopandas.GeoDataFrame(
        {
            "MRGID": ["abc\x00def"],
            "Name": ["name\x00with\x00nul"],
            "geometry": [Polygon([(0, 0), (0, 1), (1, 1), (0, 0)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    normalized = wrapper.normalize_stage_gdf(
        gdf,
        wrapper.AoiConfig(
            asset_slug="mpa",
            short_name="MPA",
            long_name="MPA",
            ext_id_field="MRGID",
            display_name_field="Name",
            stage_table="maintenance.aoi_stage_mpa",
            dataset_version="latest",
            source_url="",
            citation="",
        ),
    )

    assert normalized.iloc[0]["ext_id"] == "abcdef"
    assert normalized.iloc[0]["name"] == "namewithnul"


def test_backfill_wrapper_normalizes_integer_like_ext_ids():
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import Polygon

    wrapper = load_wrapper_module()
    gdf = geopandas.GeoDataFrame(
        {
            "MRGID": [1.0, 12.0, "007", 18.5],
            "geometry": [Polygon([(0, 0), (0, 1), (1, 1), (0, 0)])] * 4,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    normalized = wrapper.normalize_stage_gdf(
        gdf,
        wrapper.AoiConfig(
            asset_slug="mpa",
            short_name="MPA",
            long_name="MPA",
            ext_id_field="MRGID",
            display_name_field=None,
            stage_table="maintenance.aoi_stage_mpa",
            dataset_version="latest",
            source_url="",
            citation="",
        ),
    )

    assert normalized["ext_id"].tolist() == ["1", "12", "007", "18.5"]
    assert normalized["name"].tolist() == ["1", "12", "007", "18.5"]


def test_backfill_wrapper_requires_ext_id_override_when_ambiguous():
    wrapper = load_wrapper_module()

    with pytest.raises(ValueError, match="Multiple plausible ext_id fields"):
        wrapper.infer_ext_id_field(["id", "objectid", "name"])

    with pytest.raises(ValueError, match="Could not infer ext_id field"):
        wrapper.infer_ext_id_field(["name", "country"])


def test_backfill_wrapper_rejects_public_or_unsafe_stage_table():
    wrapper = load_wrapper_module()

    assert wrapper.parse_table_name("maintenance.aoi_stage_demo") == (
        "maintenance",
        "aoi_stage_demo",
    )
    with pytest.raises(ValueError, match="must not live in public"):
        wrapper.parse_table_name("public.aoi_stage_demo")
    with pytest.raises(ValueError, match="Unsafe SQL identifier"):
        wrapper.parse_table_name("maintenance.aoi-stage-demo")
