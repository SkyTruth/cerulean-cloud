"""Drop legacy AOI geometry storage

Revision ID: 4d8b6f4e6d2a
Revises: 1f70e7d0c5b1
Create Date: 2026-04-23 16:30:00.000000

"""

import sqlalchemy as sa
from geoalchemy2 import Geography, Geometry
from sqlalchemy.types import ARRAY

from alembic import op

# revision identifiers, used by Alembic.
revision = "4d8b6f4e6d2a"
down_revision = "1f70e7d0c5b1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Drop legacy AOI geometry storage."""
    # Preserve user AOI geometry before removing the legacy parent-table copy.
    op.execute(
        """
        UPDATE public.aoi_user au
        SET geometry = a.geometry
        FROM public.aoi a
        WHERE a.id = au.aoi_id
          AND au.geometry IS NULL
        """
    )

    # Preserve any post-backfill sea ice writes in the canonical JSONB field.
    op.execute(
        """
        UPDATE public.orchestrator_run
        SET dataset_versions = jsonb_set(
            COALESCE(dataset_versions, '{}'::jsonb),
            '{sea_ice_date}',
            to_jsonb(sea_ice_date),
            true
        )
        WHERE sea_ice_date IS NOT NULL
        """
    )

    # Flush deferred FK triggers from earlier revisions in the same Alembic
    # transaction before dropping AOI tables.
    op.execute("SET CONSTRAINTS ALL IMMEDIATE")
    op.execute("DROP RULE IF EXISTS bypass_slick_to_aoi_insert ON public.slick_to_aoi")

    op.drop_column("orchestrator_run", "sea_ice_date")
    op.drop_table("aoi_chunks")
    op.drop_table("aoi_mpa")
    op.drop_table("aoi_iho")
    op.drop_table("aoi_eez")
    op.drop_column("aoi", "geometry")


def downgrade() -> None:
    """Restore legacy AOI geometry storage."""
    op.add_column(
        "aoi",
        sa.Column(
            "geometry",
            Geography("MULTIPOLYGON"),
            nullable=False,
            server_default=sa.text("ST_GeogFromText('SRID=4326;MULTIPOLYGON EMPTY')"),
        ),
    )
    op.execute(
        """
        UPDATE public.aoi a
        SET geometry = au.geometry
        FROM public.aoi_user au
        WHERE a.id = au.aoi_id
          AND au.geometry IS NOT NULL
        """
    )
    op.alter_column("aoi", "geometry", server_default=None)

    op.create_table(
        "aoi_eez",
        sa.Column("aoi_id", sa.BigInteger, sa.ForeignKey("aoi.id"), primary_key=True),
        sa.Column("mrgid", sa.Integer),
        sa.Column("sovereigns", ARRAY(sa.Text)),
    )

    op.create_table(
        "aoi_iho",
        sa.Column("aoi_id", sa.BigInteger, sa.ForeignKey("aoi.id"), primary_key=True),
        sa.Column("mrgid", sa.Integer),
    )

    op.create_table(
        "aoi_mpa",
        sa.Column("aoi_id", sa.BigInteger, sa.ForeignKey("aoi.id"), primary_key=True),
        sa.Column("wdpaid", sa.Integer),
        sa.Column("desig", sa.Text),
        sa.Column("desig_type", sa.Text),
        sa.Column("status_yr", sa.Integer),
        sa.Column("mang_auth", sa.Text),
        sa.Column("parent_iso", sa.Text),
    )

    op.create_table(
        "aoi_chunks",
        sa.Column(
            "id",
            sa.BigInteger,
            sa.ForeignKey(
                "aoi.id", ondelete="CASCADE", deferrable=True, initially="DEFERRED"
            ),
        ),
        sa.Column("geometry", Geometry("POLYGON", srid=4326), nullable=False),
    )

    op.add_column("orchestrator_run", sa.Column("sea_ice_date", sa.Date()))
    op.execute(
        """
        UPDATE public.orchestrator_run
        SET sea_ice_date = (dataset_versions ->> 'sea_ice_date')::date
        WHERE dataset_versions ? 'sea_ice_date'
        """
    )

    op.execute(
        """
        CREATE OR REPLACE RULE bypass_slick_to_aoi_insert
        AS ON INSERT TO public.slick_to_aoi DO INSTEAD NOTHING
        """
    )
