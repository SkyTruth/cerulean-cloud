"""Add vocabulary i18n tables

Revision ID: d6c7b48d9f11
Revises: 8f0c0f3f1f6d
Create Date: 2026-03-12 10:30:00.000000

"""

import csv
from collections import defaultdict
from pathlib import Path

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "d6c7b48d9f11"
down_revision = "8f0c0f3f1f6d"
branch_labels = None
depends_on = None

STATUS_CHECK = "status IN ('draft', 'reviewed', 'published')"
QUALITY_CHECK = "quality IN ('human', 'machine', 'machine_reviewed')"
TRANSLATION_LOCALES = ("es", "fr", "pt", "id")
TRANSLATION_CSV_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "vocabulary_translations_es_fr_pt_id.csv"
)
TRANSLATION_CONFIG = {
    "cls": {
        "table_name": "cls_i18n",
        "id_column": "cls_id",
        "id_type": sa.Integer(),
        "fields": ("long_name", "description"),
    },
    "tag": {
        "table_name": "tag_i18n",
        "id_column": "tag_id",
        "id_type": sa.BigInteger(),
        "fields": ("long_name", "description", "citation"),
    },
    "aoi_type": {
        "table_name": "aoi_type_i18n",
        "id_column": "aoi_type_id",
        "id_type": sa.BigInteger(),
        "fields": ("long_name", "citation"),
    },
    "source_type": {
        "table_name": "source_type_i18n",
        "id_column": "source_type_id",
        "id_type": sa.BigInteger(),
        "fields": ("long_name", "citation"),
    },
    "frequency": {
        "table_name": "frequency_i18n",
        "id_column": "frequency_id",
        "id_type": sa.BigInteger(),
        "fields": ("long_name",),
    },
    "permission": {
        "table_name": "permission_i18n",
        "id_column": "permission_id",
        "id_type": sa.BigInteger(),
        "fields": ("long_name",),
    },
    "layer": {
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


def _build_translation_seed_rows():
    if not TRANSLATION_CSV_PATH.exists():
        raise RuntimeError(
            f"Missing translation seed CSV required by migration: {TRANSLATION_CSV_PATH}"
        )

    grouped = defaultdict(dict)
    with TRANSLATION_CSV_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            entity_type = row["entity_type"]
            entity_id = int(row["entity_id"])
            field_name = row["field_name"]
            for locale in TRANSLATION_LOCALES:
                group = grouped[(entity_type, entity_id, locale)]
                group["source_checksum"] = row["source_checksum"]
                value = row[locale] or None
                if value is not None:
                    group[field_name] = value

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
        seed_rows[entity_type].append(payload)

    return seed_rows


def _seed_translation_tables() -> None:
    seed_rows = _build_translation_seed_rows()
    for entity_type, rows in seed_rows.items():
        if rows:
            op.bulk_insert(_i18n_seed_table(entity_type), rows)


def upgrade() -> None:
    """Add locale roster and i18n tables for controlled vocabulary."""
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
    op.bulk_insert(
        locale_table,
        [
            {
                "code": "en",
                "english_name": "English",
                "native_name": "English",
                "text_direction": "ltr",
                "fallback_code": None,
                "is_default": True,
                "is_active": True,
                "sort_order": 1,
                "notes": "Canonical source language stored in base vocabulary tables.",
            },
            {
                "code": "es",
                "english_name": "Spanish",
                "native_name": "Espanol",
                "text_direction": "ltr",
                "fallback_code": "en",
                "is_default": False,
                "is_active": True,
                "sort_order": 2,
                "notes": "Initial Cerulean translation roster.",
            },
            {
                "code": "fr",
                "english_name": "French",
                "native_name": "Francais",
                "text_direction": "ltr",
                "fallback_code": "en",
                "is_default": False,
                "is_active": True,
                "sort_order": 3,
                "notes": "Initial Cerulean translation roster.",
            },
            {
                "code": "pt",
                "english_name": "Portuguese",
                "native_name": "Portugues",
                "text_direction": "ltr",
                "fallback_code": "en",
                "is_default": False,
                "is_active": True,
                "sort_order": 4,
                "notes": "Initial Cerulean translation roster.",
            },
            {
                "code": "id",
                "english_name": "Indonesian",
                "native_name": "Bahasa Indonesia",
                "text_direction": "ltr",
                "fallback_code": "en",
                "is_default": False,
                "is_active": True,
                "sort_order": 5,
                "notes": "Requested as Bahasa; stored as locale code id.",
            },
        ],
    )

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
    _seed_translation_tables()


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
