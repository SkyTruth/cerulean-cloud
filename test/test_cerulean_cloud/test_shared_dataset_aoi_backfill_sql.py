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
    assert "v_stage_schema = 'public'" in sql_text
    assert "RAISE EXCEPTION 'Staging table must not live in public" in sql_text
    assert "SET LOCAL lock_timeout" in sql_text
    assert "SET LOCAL statement_timeout" in sql_text
    assert "COMMIT;" in sql_text
    assert "pg_try_advisory_lock" in sql_text
    assert "pg_sleep" in sql_text
    assert "ON CONFLICT DO NOTHING" in sql_text
    assert "ST_Intersects" in sql_text
    assert "ST_IsValid" in sql_text
    assert "GeometryType(geom) NOT IN ('POLYGON', 'MULTIPOLYGON')" in sql_text
    assert "RAISE NOTICE 'Staging table % has % duplicate ext_id values" in sql_text

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
        "CREATE OR REPLACE PROCEDURE maintenance.run_shared_dataset_aoi_backfill", 1
    )[0]
    finish_sql = sql_text.split(
        "CREATE OR REPLACE PROCEDURE maintenance.finish_shared_dataset_aoi_backfill", 1
    )[1]

    assert "filter_toggle,\n        read_perm,\n        access_type" in preparation_sql
    assert (
        "FALSE,\n        v_read_perm_id,\n        'SHARED_DATASET'" in preparation_sql
    )
    assert "filter_toggle = FALSE" in preparation_sql
    assert "access_type = 'SHARED_DATASET'" in preparation_sql

    assert "filter_toggle = FALSE" in finish_sql
    assert "manual UI enablement" in finish_sql


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


def test_backfill_wrapper_infers_fields_when_unambiguous():
    wrapper = load_wrapper_module()

    assert wrapper.infer_ext_id_field(["Name", "MRGID", "geometry"]) == "MRGID"
    assert wrapper.infer_ext_id_field(["source_layer", "PRIMKEY", "NAME"]) == "PRIMKEY"
    assert wrapper.infer_display_name_field(["Name", "MRGID"], "MRGID") == "Name"
    assert wrapper.infer_display_name_field(["MRGID"], "MRGID") is None


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
