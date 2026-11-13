"""Add vocabulary i18n tables

Revision ID: d6c7b48d9f11
Revises: b3d1f7c2a9e4
Create Date: 2026-03-12 10:30:00.000000

"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "d6c7b48d9f11"
down_revision = "b3d1f7c2a9e4"
branch_labels = None
depends_on = None

STATUS_CHECK = "status IN ('draft', 'reviewed', 'published')"
QUALITY_CHECK = "quality IN ('human', 'machine', 'machine_reviewed')"
TRANSLATION_CSV_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "vocabulary_translations.csv"
)
LOCALE_METADATA_PREFIX = "# locale:"
TRANSLATION_SOURCE_COLUMNS = (
    "source_locale",
    "entity_type",
    "entity_id",
    "context_group",
    "context_key",
    "field_name",
    "source_text",
    "source_checksum",
)
DEFAULT_SOURCE_LOCALE = "en"
TRANSLATION_CONFIG = {
    "cls": {
        "base_table_name": "cls",
        "table_name": "cls_i18n",
        "id_column": "cls_id",
        "id_type": sa.Integer(),
        "fields": ("long_name", "description"),
    },
    "tag": {
        "base_table_name": "tag",
        "table_name": "tag_i18n",
        "id_column": "tag_id",
        "id_type": sa.BigInteger(),
        "fields": ("long_name", "description", "citation"),
    },
    "aoi_type": {
        "base_table_name": "aoi_type",
        "table_name": "aoi_type_i18n",
        "id_column": "aoi_type_id",
        "id_type": sa.BigInteger(),
        "fields": ("long_name", "citation"),
    },
    "source_type": {
        "base_table_name": "source_type",
        "table_name": "source_type_i18n",
        "id_column": "source_type_id",
        "id_type": sa.BigInteger(),
        "fields": ("long_name", "citation"),
    },
    "frequency": {
        "base_table_name": "frequency",
        "table_name": "frequency_i18n",
        "id_column": "frequency_id",
        "id_type": sa.BigInteger(),
        "fields": ("long_name",),
    },
    "permission": {
        "base_table_name": "permission",
        "table_name": "permission_i18n",
        "id_column": "permission_id",
        "id_type": sa.BigInteger(),
        "fields": ("long_name",),
    },
    "layer": {
        "base_table_name": "layer",
        "table_name": "layer_i18n",
        "id_column": "layer_id",
        "id_type": sa.Integer(),
        "fields": ("long_name", "notes", "citation"),
    },
}


def _create_locale_index(table_name: str) -> None:
    op.create_index(
        f"idx_{table_name}_locale_published",
        table_name,
        ["locale"],
        postgresql_where=sa.text("status = 'published'"),
    )


def _i18n_seed_table(entity_type: str):
    config = TRANSLATION_CONFIG[entity_type]
    columns = [
        sa.column(config["id_column"], config["id_type"]),
        sa.column("locale", sa.Text()),
    ]
    columns.extend(sa.column(field, sa.Text()) for field in config["fields"])
    columns.extend(
        [
            sa.column("status", sa.Text()),
            sa.column("quality", sa.Text()),
            sa.column("source_checksum", sa.Text()),
            sa.column("updated_by", sa.BigInteger()),
        ]
    )
    return sa.table(config["table_name"], *columns)


def _lookup_table(entity_type: str):
    config = TRANSLATION_CONFIG[entity_type]
    return sa.table(
        config["base_table_name"],
        sa.column("id", config["id_type"]),
        sa.column("short_name", sa.Text()),
    )


def _load_entity_id_map(entity_type: str):
    table = _lookup_table(entity_type)
    bind = op.get_bind()
    rows = bind.execute(sa.select(table.c.short_name, table.c.id)).fetchall()
    return {row[0]: row[1] for row in rows}


def _build_supported_locale_rows(translation_locales, locale_metadata):
    metadata_by_code = {}
    for metadata in locale_metadata:
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "Locale metadata rows in translation CSV must decode to JSON objects."
            )
        code = metadata.get("code")
        if not code:
            raise RuntimeError(
                "Locale metadata rows in translation CSV must include a code."
            )
        if code in metadata_by_code:
            raise RuntimeError(
                f"Duplicate locale metadata row for code={code!r} in translation CSV."
            )
        metadata_by_code[code] = metadata

    supported_codes = {DEFAULT_SOURCE_LOCALE, *translation_locales}
    unexpected_codes = sorted(set(metadata_by_code) - supported_codes)
    if unexpected_codes:
        raise RuntimeError(
            "Locale metadata rows found without matching translation columns: "
            + ", ".join(unexpected_codes)
        )

    locale_rows = []
    default_count = 0
    for sort_order, code in enumerate(
        [DEFAULT_SOURCE_LOCALE, *translation_locales], start=1
    ):
        metadata = metadata_by_code.get(code, {})
        if code != code.lower():
            raise RuntimeError(
                f"Locale code {code!r} must be lowercase to satisfy supported_locale constraints."
            )

        is_default = bool(metadata.get("is_default", code == DEFAULT_SOURCE_LOCALE))
        fallback_code = metadata.get("fallback_code")
        if fallback_code is None and not is_default:
            fallback_code = DEFAULT_SOURCE_LOCALE
        if is_default and fallback_code is not None:
            raise RuntimeError(
                f"Locale code {code!r} cannot be default and have a fallback_code."
            )
        if is_default:
            default_count += 1

        locale_rows.append(
            {
                "code": code,
                "english_name": metadata.get("english_name", code),
                "native_name": metadata.get("native_name", code),
                "text_direction": metadata.get("text_direction", "ltr"),
                "fallback_code": fallback_code,
                "is_default": is_default,
                "is_active": bool(metadata.get("is_active", True)),
                "sort_order": sort_order,
                "notes": metadata.get(
                    "notes",
                    (
                        "Canonical source language stored in base vocabulary tables."
                        if code == DEFAULT_SOURCE_LOCALE
                        else "Translation locale discovered from the CSV header."
                    ),
                ),
            }
        )

    if default_count != 1:
        raise RuntimeError(
            f"supported_locale seed requires exactly one default locale, found {default_count}."
        )

    return locale_rows


def _load_translation_csv():
    if not TRANSLATION_CSV_PATH.exists():
        raise RuntimeError(
            f"Missing translation seed CSV required by migration: {TRANSLATION_CSV_PATH}"
        )

    locale_metadata = []
    csv_lines = []
    with TRANSLATION_CSV_PATH.open(encoding="utf-8") as csv_file:
        for raw_line in csv_file:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith(LOCALE_METADATA_PREFIX):
                try:
                    locale_metadata.append(
                        json.loads(stripped[len(LOCALE_METADATA_PREFIX) :].strip())
                    )
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Locale metadata rows in translation CSV must be valid JSON."
                    ) from exc
                continue
            if stripped.startswith("#"):
                continue
            csv_lines.append(raw_line)

    if not csv_lines:
        raise RuntimeError(
            f"Translation seed CSV does not contain any translation rows: {TRANSLATION_CSV_PATH}"
        )

    reader = csv.DictReader(csv_lines)
    if reader.fieldnames is None:
        raise RuntimeError(
            f"Translation seed CSV is missing a header row: {TRANSLATION_CSV_PATH}"
        )

    missing_columns = sorted(set(TRANSLATION_SOURCE_COLUMNS) - set(reader.fieldnames))
    if missing_columns:
        raise RuntimeError(
            "Translation seed CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    translation_locales = [
        field_name
        for field_name in reader.fieldnames
        if field_name not in TRANSLATION_SOURCE_COLUMNS
    ]
    if not translation_locales:
        raise RuntimeError(
            "Translation seed CSV must include at least one locale column beyond the source fields."
        )
    if DEFAULT_SOURCE_LOCALE in translation_locales:
        raise RuntimeError(
            f"Base locale {DEFAULT_SOURCE_LOCALE!r} must not be duplicated as a translation column."
        )

    translation_rows = list(reader)
    locale_rows = _build_supported_locale_rows(translation_locales, locale_metadata)
    return translation_locales, locale_rows, translation_rows


def _build_translation_seed_rows(translation_rows, translation_locales):
    entity_id_maps = {
        entity_type: _load_entity_id_map(entity_type)
        for entity_type in TRANSLATION_CONFIG
    }
    grouped = defaultdict(dict)
    skipped_keys = set()
    for row in translation_rows:
        entity_type = row["entity_type"]
        context_key = row["context_key"]
        entity_id = entity_id_maps[entity_type].get(context_key)
        if entity_id is None:
            skipped_keys.add((entity_type, context_key))
            continue
        field_name = row["field_name"]
        for locale in translation_locales:
            group = grouped[(entity_type, entity_id, locale)]
            group["source_checksum"] = row["source_checksum"]
            value = row[locale] or None
            if value is not None:
                group[field_name] = value

    if skipped_keys:
        skipped_list = ", ".join(
            f"{entity_type}:{context_key}"
            for entity_type, context_key in sorted(skipped_keys)
        )
        print(
            "Skipping translation seed rows for vocabulary entries not present in this "
            f"database: {skipped_list}"
        )

    seed_rows = {entity_type: [] for entity_type in TRANSLATION_CONFIG}
    for (entity_type, entity_id, locale), values in sorted(grouped.items()):
        config = TRANSLATION_CONFIG[entity_type]
        payload = {
            config["id_column"]: entity_id,
            "locale": locale,
            "status": "published",
            "quality": "human",
            "source_checksum": values["source_checksum"],
            "updated_by": None,
        }
        for field in config["fields"]:
            payload[field] = values.get(field)
        if any(payload[field] is not None for field in config["fields"]):
            seed_rows[entity_type].append(payload)

    return seed_rows


def _seed_supported_locales(locale_rows) -> None:
    locale_table = sa.table(
        "supported_locale",
        sa.column("code", sa.Text()),
        sa.column("english_name", sa.Text()),
        sa.column("native_name", sa.Text()),
        sa.column("text_direction", sa.Text()),
        sa.column("fallback_code", sa.Text()),
        sa.column("is_default", sa.Boolean()),
        sa.column("is_active", sa.Boolean()),
        sa.column("sort_order", sa.Integer()),
        sa.column("notes", sa.Text()),
    )
    remaining_rows = list(locale_rows)
    inserted_codes = set()
    while remaining_rows:
        ready_rows = [
            row
            for row in remaining_rows
            if row["fallback_code"] is None or row["fallback_code"] in inserted_codes
        ]
        if not ready_rows:
            unresolved = ", ".join(
                f"{row['code']}->{row['fallback_code']}" for row in remaining_rows
            )
            raise RuntimeError(
                "supported_locale metadata contains unresolved fallback references: "
                f"{unresolved}"
            )
        op.bulk_insert(locale_table, ready_rows)
        inserted_codes.update(row["code"] for row in ready_rows)
        remaining_rows = [
            row for row in remaining_rows if row["code"] not in inserted_codes
        ]


def _seed_translation_tables(translation_rows, translation_locales) -> None:
    seed_rows = _build_translation_seed_rows(translation_rows, translation_locales)
    for entity_type, rows in seed_rows.items():
        if rows:
            op.bulk_insert(_i18n_seed_table(entity_type), rows)


def upgrade() -> None:
    """Add locale roster and i18n tables for controlled vocabulary."""
    translation_locales, locale_rows, translation_rows = _load_translation_csv()
    op.create_table(
        "supported_locale",
        sa.Column("code", sa.Text(), primary_key=True),
        sa.Column("english_name", sa.Text(), nullable=False),
        sa.Column("native_name", sa.Text(), nullable=False),
        sa.Column("text_direction", sa.Text(), nullable=False, server_default="ltr"),
        sa.Column("fallback_code", sa.Text(), sa.ForeignKey("supported_locale.code")),
        sa.Column(
            "is_default", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("notes", sa.Text()),
        sa.CheckConstraint("code = lower(code)", name="ck_supported_locale_code_lower"),
        sa.CheckConstraint(
            "text_direction IN ('ltr', 'rtl')",
            name="ck_supported_locale_direction",
        ),
        sa.CheckConstraint(
            "fallback_code IS NULL OR fallback_code <> code",
            name="ck_supported_locale_fallback_self",
        ),
        sa.CheckConstraint(
            "NOT is_default OR fallback_code IS NULL",
            name="ck_supported_locale_default_fallback",
        ),
    )
    op.create_index(
        "uq_supported_locale_single_default",
        "supported_locale",
        ["is_default"],
        unique=True,
        postgresql_where=sa.text("is_default"),
    )
    op.create_index(
        "uq_supported_locale_sort_order",
        "supported_locale",
        ["sort_order"],
        unique=True,
    )
    _seed_supported_locales(locale_rows)

    op.create_table(
        "cls_i18n",
        sa.Column(
            "cls_id",
            sa.Integer(),
            sa.ForeignKey("cls.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "locale",
            sa.Text(),
            sa.ForeignKey("supported_locale.code"),
            nullable=False,
        ),
        sa.Column("long_name", sa.Text()),
        sa.Column("description", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False, server_default="published"),
        sa.Column("quality", sa.Text(), nullable=False, server_default="human"),
        sa.Column("source_checksum", sa.Text(), nullable=False),
        sa.Column(
            "updated_by",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("cls_id", "locale"),
        sa.CheckConstraint(STATUS_CHECK, name="ck_cls_i18n_status"),
        sa.CheckConstraint(QUALITY_CHECK, name="ck_cls_i18n_quality"),
        sa.CheckConstraint(
            "num_nonnulls(long_name, description) > 0",
            name="ck_cls_i18n_has_translation",
        ),
    )
    _create_locale_index("cls_i18n")

    op.create_table(
        "tag_i18n",
        sa.Column(
            "tag_id",
            sa.BigInteger(),
            sa.ForeignKey("tag.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "locale",
            sa.Text(),
            sa.ForeignKey("supported_locale.code"),
            nullable=False,
        ),
        sa.Column("long_name", sa.Text()),
        sa.Column("description", sa.Text()),
        sa.Column("citation", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False, server_default="published"),
        sa.Column("quality", sa.Text(), nullable=False, server_default="human"),
        sa.Column("source_checksum", sa.Text(), nullable=False),
        sa.Column(
            "updated_by",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("tag_id", "locale"),
        sa.CheckConstraint(STATUS_CHECK, name="ck_tag_i18n_status"),
        sa.CheckConstraint(QUALITY_CHECK, name="ck_tag_i18n_quality"),
        sa.CheckConstraint(
            "num_nonnulls(long_name, description, citation) > 0",
            name="ck_tag_i18n_has_translation",
        ),
    )
    _create_locale_index("tag_i18n")

    op.create_table(
        "aoi_type_i18n",
        sa.Column(
            "aoi_type_id",
            sa.BigInteger(),
            sa.ForeignKey("aoi_type.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "locale",
            sa.Text(),
            sa.ForeignKey("supported_locale.code"),
            nullable=False,
        ),
        sa.Column("long_name", sa.Text()),
        sa.Column("citation", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False, server_default="published"),
        sa.Column("quality", sa.Text(), nullable=False, server_default="human"),
        sa.Column("source_checksum", sa.Text(), nullable=False),
        sa.Column(
            "updated_by",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("aoi_type_id", "locale"),
        sa.CheckConstraint(STATUS_CHECK, name="ck_aoi_type_i18n_status"),
        sa.CheckConstraint(QUALITY_CHECK, name="ck_aoi_type_i18n_quality"),
        sa.CheckConstraint(
            "num_nonnulls(long_name, citation) > 0",
            name="ck_aoi_type_i18n_has_translation",
        ),
    )
    _create_locale_index("aoi_type_i18n")

    op.create_table(
        "source_type_i18n",
        sa.Column(
            "source_type_id",
            sa.BigInteger(),
            sa.ForeignKey("source_type.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "locale",
            sa.Text(),
            sa.ForeignKey("supported_locale.code"),
            nullable=False,
        ),
        sa.Column("long_name", sa.Text()),
        sa.Column("citation", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False, server_default="published"),
        sa.Column("quality", sa.Text(), nullable=False, server_default="human"),
        sa.Column("source_checksum", sa.Text(), nullable=False),
        sa.Column(
            "updated_by",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("source_type_id", "locale"),
        sa.CheckConstraint(STATUS_CHECK, name="ck_source_type_i18n_status"),
        sa.CheckConstraint(QUALITY_CHECK, name="ck_source_type_i18n_quality"),
        sa.CheckConstraint(
            "num_nonnulls(long_name, citation) > 0",
            name="ck_source_type_i18n_has_translation",
        ),
    )
    _create_locale_index("source_type_i18n")

    op.create_table(
        "frequency_i18n",
        sa.Column(
            "frequency_id",
            sa.BigInteger(),
            sa.ForeignKey("frequency.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "locale",
            sa.Text(),
            sa.ForeignKey("supported_locale.code"),
            nullable=False,
        ),
        sa.Column("long_name", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False, server_default="published"),
        sa.Column("quality", sa.Text(), nullable=False, server_default="human"),
        sa.Column("source_checksum", sa.Text(), nullable=False),
        sa.Column(
            "updated_by",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("frequency_id", "locale"),
        sa.CheckConstraint(STATUS_CHECK, name="ck_frequency_i18n_status"),
        sa.CheckConstraint(QUALITY_CHECK, name="ck_frequency_i18n_quality"),
        sa.CheckConstraint(
            "long_name IS NOT NULL",
            name="ck_frequency_i18n_has_translation",
        ),
    )
    _create_locale_index("frequency_i18n")

    op.create_table(
        "permission_i18n",
        sa.Column(
            "permission_id",
            sa.BigInteger(),
            sa.ForeignKey("permission.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "locale",
            sa.Text(),
            sa.ForeignKey("supported_locale.code"),
            nullable=False,
        ),
        sa.Column("long_name", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False, server_default="published"),
        sa.Column("quality", sa.Text(), nullable=False, server_default="human"),
        sa.Column("source_checksum", sa.Text(), nullable=False),
        sa.Column(
            "updated_by",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("permission_id", "locale"),
        sa.CheckConstraint(STATUS_CHECK, name="ck_permission_i18n_status"),
        sa.CheckConstraint(QUALITY_CHECK, name="ck_permission_i18n_quality"),
        sa.CheckConstraint(
            "long_name IS NOT NULL",
            name="ck_permission_i18n_has_translation",
        ),
    )
    _create_locale_index("permission_i18n")

    op.create_table(
        "layer_i18n",
        sa.Column(
            "layer_id",
            sa.Integer(),
            sa.ForeignKey("layer.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "locale",
            sa.Text(),
            sa.ForeignKey("supported_locale.code"),
            nullable=False,
        ),
        sa.Column("long_name", sa.Text()),
        sa.Column("notes", sa.Text()),
        sa.Column("citation", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False, server_default="published"),
        sa.Column("quality", sa.Text(), nullable=False, server_default="human"),
        sa.Column("source_checksum", sa.Text(), nullable=False),
        sa.Column(
            "updated_by",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("layer_id", "locale"),
        sa.CheckConstraint(STATUS_CHECK, name="ck_layer_i18n_status"),
        sa.CheckConstraint(QUALITY_CHECK, name="ck_layer_i18n_quality"),
        sa.CheckConstraint(
            "num_nonnulls(long_name, notes, citation) > 0",
            name="ck_layer_i18n_has_translation",
        ),
    )
    _create_locale_index("layer_i18n")
    _seed_translation_tables(translation_rows, translation_locales)


def downgrade() -> None:
    """Remove vocabulary i18n tables and locale roster."""
    op.drop_table("layer_i18n")
    op.drop_table("permission_i18n")
    op.drop_table("frequency_i18n")
    op.drop_table("source_type_i18n")
    op.drop_table("aoi_type_i18n")
    op.drop_table("tag_i18n")
    op.drop_table("cls_i18n")
    op.drop_table("supported_locale")
