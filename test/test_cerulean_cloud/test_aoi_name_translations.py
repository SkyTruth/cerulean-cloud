"""Tests for curated AOI name translation seed data."""

import csv
import hashlib
import importlib.util
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "docs/aoi_name_translations.csv"
MIGRATION_PATH = REPO_ROOT / "alembic/versions/9c73b5c1d2e4_add_aoi_name_i18n.py"


def _load_csv_rows():
    csv_text = CSV_PATH.read_text(encoding="utf-8")
    return list(
        csv.DictReader(
            line for line in csv_text.splitlines() if not line.startswith("#")
        )
    )


def _load_migration_module():
    spec = importlib.util.spec_from_file_location("aoi_name_i18n", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aoi_name_translation_csv_covers_curated_eez_and_iho_names():
    rows = _load_csv_rows()

    assert Counter(row["aoi_type_short_name"] for row in rows) == {
        "EEZ": 282,
        "IHO": 102,
    }
    assert {row["source_locale"] for row in rows} == {"en"}
    assert {row["aoi_type_short_name"] for row in rows}.isdisjoint({"USER", "MPA"})
    aoi_keys = {(row["aoi_type_short_name"], row["mrgid"]) for row in rows}
    assert len(aoi_keys) == len(rows)
    translation_locales = {"es", "fr", "pt", "pt-br", "id", "sw"}
    assert translation_locales.issubset(rows[0].keys())

    for row in rows:
        assert row["mrgid"]
        assert row["source_name"]
        assert (
            row["source_checksum"]
            == hashlib.md5(row["source_name"].encode("utf-8")).hexdigest()
        )
        for locale in translation_locales:
            assert row[locale].strip(), (
                f"Missing {locale} translation for "
                f"{row['aoi_type_short_name']}:{row['mrgid']}"
            )


def test_aoi_name_i18n_migration_uses_current_aoi_keys():
    migration_text = MIGRATION_PATH.read_text(encoding="utf-8")
    tipg_text = (REPO_ROOT / "stack/cloud_run_tipg.py").read_text(encoding="utf-8")

    assert 'down_revision = "d6c7b48d9f11"' in migration_text
    assert '"aoi_i18n"' in migration_text
    assert 'sa.ForeignKey("aoi.id", ondelete="CASCADE")' in migration_text
    assert 'sa.ForeignKey("supported_locale.code")' in migration_text
    assert 'TRANSLATABLE_AOI_TYPES = {"EEZ", "IHO"}' in migration_text
    assert "aoi_eez.mrgid::text AS mrgid" in migration_text
    assert "aoi_iho.mrgid::text AS mrgid" in migration_text
    assert "aoi.ext_id" not in migration_text
    assert '"MPA"' not in migration_text
    assert '"USER"' not in migration_text
    assert '"public.aoi_i18n"' in tipg_text


def test_aoi_name_i18n_migration_loads_translation_csv():
    migration = _load_migration_module()
    locales, rows = migration._load_aoi_name_translation_csv()

    assert locales == ["es", "fr", "pt", "pt-br", "id", "sw"]
    assert len(rows) == 384
