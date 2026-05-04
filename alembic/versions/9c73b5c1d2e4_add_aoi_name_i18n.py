"""Add AOI name i18n

Revision ID: 9c73b5c1d2e4
Revises: d6c7b48d9f11
Create Date: 2026-04-28 11:55:00.000000

"""

import csv
import hashlib
from pathlib import Path

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "9c73b5c1d2e4"
down_revision = "d6c7b48d9f11"
branch_labels = None
depends_on = None

STATUS_CHECK = "status IN ('draft', 'reviewed', 'published')"
QUALITY_CHECK = "quality IN ('human', 'machine', 'machine_reviewed')"
DEFAULT_SOURCE_LOCALE = "en"
LOCALE_METADATA_PREFIX = "# locale:"
AOI_NAME_TRANSLATION_CSV_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "aoi_name_translations.csv"
)
AOI_NAME_SOURCE_COLUMNS = (
    "source_locale",
    "aoi_type_short_name",
    "mrgid",
    "source_name",
    "source_checksum",
)
TRANSLATABLE_AOI_TYPES = {"EEZ", "IHO"}


def _aoi_i18n_seed_table():
    return sa.table(
        "aoi_i18n",
        sa.column("aoi_id", sa.BigInteger()),
        sa.column("locale", sa.Text()),
        sa.column("name", sa.Text()),
        sa.column("status", sa.Text()),
        sa.column("quality", sa.Text()),
        sa.column("source_checksum", sa.Text()),
        sa.column("updated_by", sa.BigInteger()),
    )


def _source_checksum(source_name: str) -> str:
    return hashlib.md5(source_name.encode("utf-8")).hexdigest()


def _load_aoi_name_translation_csv():
    if not AOI_NAME_TRANSLATION_CSV_PATH.exists():
        raise RuntimeError(
            "Missing AOI name translation seed CSV required by migration: "
            f"{AOI_NAME_TRANSLATION_CSV_PATH}"
        )

    csv_lines = []
    with AOI_NAME_TRANSLATION_CSV_PATH.open(encoding="utf-8") as csv_file:
        for raw_line in csv_file:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith(LOCALE_METADATA_PREFIX):
                continue
            if stripped.startswith("#"):
                continue
            csv_lines.append(raw_line)

    if not csv_lines:
        raise RuntimeError(
            "AOI name translation CSV does not contain any translation rows: "
            f"{AOI_NAME_TRANSLATION_CSV_PATH}"
        )

    reader = csv.DictReader(csv_lines)
    if reader.fieldnames is None:
        raise RuntimeError(
            "AOI name translation CSV is missing a header row: "
            f"{AOI_NAME_TRANSLATION_CSV_PATH}"
        )

    missing_columns = sorted(set(AOI_NAME_SOURCE_COLUMNS) - set(reader.fieldnames))
    if missing_columns:
        raise RuntimeError(
            "AOI name translation CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    translation_locales = [
        field_name
        for field_name in reader.fieldnames
        if field_name not in AOI_NAME_SOURCE_COLUMNS
    ]
    if DEFAULT_SOURCE_LOCALE in translation_locales:
        raise RuntimeError(
            f"Base locale {DEFAULT_SOURCE_LOCALE!r} must not be duplicated as a "
            "translation column."
        )

    rows = list(reader)
    for row in rows:
        source_locale = row["source_locale"]
        if source_locale != DEFAULT_SOURCE_LOCALE:
            raise RuntimeError(
                "AOI name translation CSV only supports source_locale="
                f"{DEFAULT_SOURCE_LOCALE!r}; found {source_locale!r}."
            )
        aoi_type_short_name = row["aoi_type_short_name"]
        if aoi_type_short_name not in TRANSLATABLE_AOI_TYPES:
            raise RuntimeError(
                "AOI name translation CSV may only include EEZ and IHO rows; "
                f"found {aoi_type_short_name!r}."
            )
        expected_checksum = _source_checksum(row["source_name"])
        if row["source_checksum"] != expected_checksum:
            raise RuntimeError(
                "AOI name translation CSV has stale source_checksum for "
                f"{aoi_type_short_name}:{row['mrgid']}."
            )

    return translation_locales, rows


def _load_aoi_id_map():
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            """
            SELECT
                keyed_aoi.aoi_type_short_name,
                keyed_aoi.mrgid,
                MIN(keyed_aoi.aoi_id) AS aoi_id
            FROM (
                SELECT
                    aoi_type.short_name AS aoi_type_short_name,
                    aoi_eez.mrgid::text AS mrgid,
                    aoi.id AS aoi_id
                FROM public.aoi_eez
                JOIN public.aoi
                  ON aoi.id = aoi_eez.aoi_id
                JOIN public.aoi_type
                  ON aoi_type.id = aoi.type
                WHERE aoi_type.short_name = 'EEZ'
                  AND aoi_eez.mrgid IS NOT NULL

                UNION ALL

                SELECT
                    aoi_type.short_name AS aoi_type_short_name,
                    aoi_iho.mrgid::text AS mrgid,
                    aoi.id AS aoi_id
                FROM public.aoi_iho
                JOIN public.aoi
                  ON aoi.id = aoi_iho.aoi_id
                JOIN public.aoi_type
                  ON aoi_type.id = aoi.type
                WHERE aoi_type.short_name = 'IHO'
                  AND aoi_iho.mrgid IS NOT NULL
            ) AS keyed_aoi
            GROUP BY keyed_aoi.aoi_type_short_name, keyed_aoi.mrgid
            """
        )
    ).fetchall()
    return {(row[0], row[1]): row[2] for row in rows}


def _seed_aoi_name_translations() -> None:
    translation_locales, translation_rows = _load_aoi_name_translation_csv()
    aoi_id_map = _load_aoi_id_map()

    skipped_keys = []
    seed_rows = []
    for row in translation_rows:
        key = (row["aoi_type_short_name"], row["mrgid"])
        aoi_id = aoi_id_map.get(key)
        if aoi_id is None:
            skipped_keys.append(key)
            continue

        for locale in translation_locales:
            translated_name = row[locale].strip()
            if not translated_name:
                continue
            seed_rows.append(
                {
                    "aoi_id": aoi_id,
                    "locale": locale,
                    "name": translated_name,
                    "status": "published",
                    "quality": "human",
                    "source_checksum": row["source_checksum"],
                    "updated_by": None,
                }
            )

    if skipped_keys:
        skipped_list = ", ".join(
            f"{aoi_type_short_name}:{mrgid}"
            for aoi_type_short_name, mrgid in sorted(skipped_keys)
        )
        op.get_context().impl.static_output(
            "Skipping AOI name translation seed rows for AOIs not present in "
            f"this database: {skipped_list}"
        )

    if seed_rows:
        op.bulk_insert(_aoi_i18n_seed_table(), seed_rows)


def upgrade() -> None:
    """Add i18n rows for curated AOI names."""
    op.create_table(
        "aoi_i18n",
        sa.Column(
            "aoi_id",
            sa.BigInteger(),
            sa.ForeignKey("aoi.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "locale",
            sa.Text(),
            sa.ForeignKey("supported_locale.code"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
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
        sa.PrimaryKeyConstraint("aoi_id", "locale"),
        sa.CheckConstraint(STATUS_CHECK, name="ck_aoi_i18n_status"),
        sa.CheckConstraint(QUALITY_CHECK, name="ck_aoi_i18n_quality"),
        sa.CheckConstraint("name <> ''", name="ck_aoi_i18n_name_not_empty"),
    )
    op.create_index(
        "idx_aoi_i18n_locale_published",
        "aoi_i18n",
        ["locale"],
        postgresql_where=sa.text("status = 'published'"),
    )
    _seed_aoi_name_translations()


def downgrade() -> None:
    """Remove curated AOI name translations."""
    op.drop_table("aoi_i18n")
