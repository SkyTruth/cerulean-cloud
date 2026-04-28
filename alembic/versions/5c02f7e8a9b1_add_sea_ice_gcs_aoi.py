"""Add SEA_ICE GCS AOI

Revision ID: 5c02f7e8a9b1
Revises: 4d8b6f4e6d2a
Create Date: 2026-04-28 12:30:00.000000

"""

import json

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "5c02f7e8a9b1"
down_revision = "4d8b6f4e6d2a"
branch_labels = None
depends_on = None

SEA_ICE_AOI_CITATION = (
    "NOAA National Ice Center and National Snow and Ice Data Center. Multisensor "
    "Analyzed Sea Ice Extent - Northern Hemisphere (MASIE-NH), Version 1. "
    "Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed "
    "Active Archive Center. https://doi.org/10.7265/N5GT5K3K."
)
SEA_ICE_AOI_PROPERTIES = {
    "fgb_uri": "gs://cerulean-cloud-aoi/sea-ice/masie_4km/%Y/masie_ice_r00_v01_%Y%j_4km.fgb",
    "dataset_version": "NOAA_MASIE_G02186_4km_v01",
    "dataset_version_format": "NOAA_MASIE_G02186_4km_v01_%Y-%m-%d",
    "ext_id_field": "MASK_DATE",
    "display_name_field": "NAME",
    "date_fallback_days": 31,
    "match_buffer_m": 1000,
}
SEA_ICE_AOI_SOURCE_CHECKSUM = "b103f297b46fbcbc106ed36725138c6e"
GCS_PROP_KEYS = [
    "fgb_uri",
    "pmt_uri",
    "dataset_version",
    "dataset_version_format",
    "ext_id_field",
    "display_name_field",
    "date_fallback_days",
    "match_buffer_m",
]
LEGACY_GCS_PROP_KEYS = [
    "fgb_uri",
    "pmt_uri",
    "dataset_version",
    "ext_id_field",
    "display_name_field",
]


def _get_owner_id():
    owner_id = (
        op.get_bind()
        .execute(
            sa.text(
                """
            SELECT id
            FROM public.users
            WHERE email = :email
            ORDER BY id
            LIMIT 1
            """
            ),
            {"email": "dummy@dummy.dummy"},
        )
        .scalar()
    )
    if owner_id is None:
        op.get_context().impl.static_output(
            "No bootstrap user dummy@dummy.dummy found; leaving aoi_type.owner NULL."
        )
    return owner_id


def _set_gcs_prop_keys(prop_keys: list[str]) -> None:
    op.get_bind().execute(
        sa.text(
            """
            UPDATE public.aoi_access_type
            SET prop_keys = CAST(:prop_keys AS text[])
            WHERE short_name = 'GCS'
            """
        ),
        {"prop_keys": prop_keys},
    )


def _upsert_sea_ice_aoi_type() -> None:
    op.get_bind().execute(
        sa.text(
            """
            INSERT INTO public.aoi_type (
                short_name,
                long_name,
                source_url,
                citation,
                filter_toggle,
                owner,
                read_perm,
                access_type,
                properties
            )
            VALUES (
                'SEA_ICE',
                'Sea Ice',
                'https://nsidc.org/data/g02186',
                :citation,
                FALSE,
                :owner_id,
                NULL,
                'GCS',
                CAST(:properties AS jsonb)
            )
            ON CONFLICT (short_name) DO UPDATE
            SET
                long_name = EXCLUDED.long_name,
                source_url = EXCLUDED.source_url,
                citation = EXCLUDED.citation,
                filter_toggle = EXCLUDED.filter_toggle,
                owner = EXCLUDED.owner,
                read_perm = EXCLUDED.read_perm,
                access_type = EXCLUDED.access_type,
                properties = EXCLUDED.properties
            """
        ),
        {
            "citation": SEA_ICE_AOI_CITATION,
            "owner_id": _get_owner_id(),
            "properties": json.dumps(SEA_ICE_AOI_PROPERTIES),
        },
    )


def _seed_sea_ice_aoi_i18n() -> None:
    op.get_bind().execute(
        sa.text(
            """
            WITH sea_ice_aoi_type AS (
                SELECT id
                FROM public.aoi_type
                WHERE short_name = 'SEA_ICE'
            ),
            translations(locale, long_name, citation) AS (
                VALUES
                    ('es', 'Hielo marino', :citation),
                    ('fr', 'Glace de mer', :citation),
                    ('pt', 'Gelo marinho', :citation),
                    ('pt-br', 'Gelo marinho', :citation),
                    ('id', 'Es laut', :citation),
                    ('sw', 'Barafu ya baharini', :citation)
            )
            INSERT INTO public.aoi_type_i18n (
                aoi_type_id,
                locale,
                long_name,
                citation,
                status,
                quality,
                source_checksum
            )
            SELECT
                sea_ice_aoi_type.id,
                translations.locale,
                translations.long_name,
                translations.citation,
                'published',
                'human',
                :source_checksum
            FROM sea_ice_aoi_type
            JOIN translations ON TRUE
            JOIN public.supported_locale
              ON supported_locale.code = translations.locale
            ON CONFLICT (aoi_type_id, locale) DO UPDATE
            SET
                long_name = EXCLUDED.long_name,
                citation = EXCLUDED.citation,
                status = EXCLUDED.status,
                quality = EXCLUDED.quality,
                source_checksum = EXCLUDED.source_checksum,
                updated_at = now()
            """
        ),
        {
            "citation": SEA_ICE_AOI_CITATION,
            "source_checksum": SEA_ICE_AOI_SOURCE_CHECKSUM,
        },
    )


def upgrade() -> None:
    """Seed SEA_ICE as an internal temporal GCS AOI."""
    _set_gcs_prop_keys(GCS_PROP_KEYS)
    _upsert_sea_ice_aoi_type()
    _seed_sea_ice_aoi_i18n()


def downgrade() -> None:
    """Remove the internal SEA_ICE AOI seed."""
    op.execute(
        """
        DELETE FROM public.slick_to_aoi
        WHERE aoi IN (
            SELECT aoi.id
            FROM public.aoi
            JOIN public.aoi_type
              ON aoi_type.id = aoi.type
            WHERE aoi_type.short_name = 'SEA_ICE'
        )
        """
    )
    op.execute(
        """
        DELETE FROM public.aoi
        WHERE type IN (
            SELECT id
            FROM public.aoi_type
            WHERE short_name = 'SEA_ICE'
        )
        """
    )
    op.execute(
        """
        DELETE FROM public.aoi_type_i18n
        WHERE aoi_type_id IN (
            SELECT id
            FROM public.aoi_type
            WHERE short_name = 'SEA_ICE'
        )
        """
    )
    op.execute("DELETE FROM public.aoi_type WHERE short_name = 'SEA_ICE'")
    _set_gcs_prop_keys(LEGACY_GCS_PROP_KEYS)
