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

SLICK_TRIGGER_FUNC_WITH_AOI_CHUNKS = """
    CREATE OR REPLACE FUNCTION slick_before_trigger_func()
    RETURNS trigger
    AS $$
    DECLARE
        timer timestamptz := clock_timestamp();
        _geog geography := NEW.geometry;
        _geom geometry;
        oriented_envelope geometry;
        oe_ring geometry;
        rec record;
    BEGIN
        RAISE NOTICE '---------------------------------------------------------';
        RAISE NOTICE 'In slick_before_trigger_func. %', (clock_timestamp() - timer)::interval;
        _geom := _geog::geometry;
        oriented_envelope := st_orientedenvelope(_geom);
        oe_ring := st_exteriorring(oriented_envelope);
        NEW.geometry_count := st_numgeometries(_geom);
        NEW.largest_area := (
            SELECT MAX(st_area((poly.geom)::geography))
            FROM st_dump(_geom) AS poly
        );
        NEW.median_area := (
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY st_area((poly.geom)::geography))
            FROM st_dump(_geom) AS poly
        );
        NEW.area := st_area(_geog);
        NEW.centroid := st_centroid(_geog);
        NEW.perimeter = st_perimeter(_geog);
        NEW.polsby_popper := 4.0 * pi() * NEW.area / (NEW.perimeter ^ 2.0);
        NEW.fill_factor := NEW.area / st_area(oriented_envelope::geography);
        NEW.length := GREATEST(
            st_distance(
                st_pointn(oe_ring,1)::geography,
                st_pointn(oe_ring,2)::geography
            ),
            st_distance(
                st_pointn(oe_ring,2)::geography,
                st_pointn(oe_ring,3)::geography
            )
        );
        RAISE NOTICE 'Calculated all generated fields. %', (clock_timestamp() - timer)::interval;
        NEW.cls := COALESCE(
            NEW.cls,
            (
                SELECT cls.id
                FROM cls
                JOIN orchestrator_run ON NEW.orchestrator_run = orchestrator_run.id
                JOIN LATERAL json_each_text((SELECT cls_map FROM model WHERE id = orchestrator_run.model))
                    m(key, value)
                    ON key::integer = NEW.inference_idx
                WHERE cls.short_name = CASE
                    WHEN value = 'BACKGROUND' THEN 'NOT_OIL'
                    ELSE value
                END
                LIMIT 1
            )
        );
        RAISE NOTICE 'Calculated NEW.cls. %', (clock_timestamp() - timer)::interval;

        INSERT INTO slick_to_aoi(slick, aoi)
        SELECT DISTINCT NEW.id, aoi_chunks.id
        FROM aoi_chunks
        WHERE st_intersects(_geom, aoi_chunks.geometry);

        RAISE NOTICE 'Insert done to slick_to_aoi. %', (clock_timestamp() - timer)::interval;

        RETURN NEW;
    END;
    $$ LANGUAGE PLPGSQL;
"""

SLICK_TRIGGER_FUNC_WITHOUT_AOI_CHUNKS = """
    CREATE OR REPLACE FUNCTION slick_before_trigger_func()
    RETURNS trigger
    AS $$
    DECLARE
        timer timestamptz := clock_timestamp();
        _geog geography := NEW.geometry;
        _geom geometry;
        oriented_envelope geometry;
        oe_ring geometry;
        rec record;
    BEGIN
        RAISE NOTICE '---------------------------------------------------------';
        RAISE NOTICE 'In slick_before_trigger_func. %', (clock_timestamp() - timer)::interval;
        _geom := _geog::geometry;
        oriented_envelope := st_orientedenvelope(_geom);
        oe_ring := st_exteriorring(oriented_envelope);
        NEW.geometry_count := st_numgeometries(_geom);
        NEW.largest_area := (
            SELECT MAX(st_area((poly.geom)::geography))
            FROM st_dump(_geom) AS poly
        );
        NEW.median_area := (
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY st_area((poly.geom)::geography))
            FROM st_dump(_geom) AS poly
        );
        NEW.area := st_area(_geog);
        NEW.centroid := st_centroid(_geog);
        NEW.perimeter = st_perimeter(_geog);
        NEW.polsby_popper := 4.0 * pi() * NEW.area / (NEW.perimeter ^ 2.0);
        NEW.fill_factor := NEW.area / st_area(oriented_envelope::geography);
        NEW.length := GREATEST(
            st_distance(
                st_pointn(oe_ring,1)::geography,
                st_pointn(oe_ring,2)::geography
            ),
            st_distance(
                st_pointn(oe_ring,2)::geography,
                st_pointn(oe_ring,3)::geography
            )
        );
        RAISE NOTICE 'Calculated all generated fields. %', (clock_timestamp() - timer)::interval;
        NEW.cls := COALESCE(
            NEW.cls,
            (
                SELECT cls.id
                FROM cls
                JOIN orchestrator_run ON NEW.orchestrator_run = orchestrator_run.id
                JOIN LATERAL json_each_text((SELECT cls_map FROM model WHERE id = orchestrator_run.model))
                    m(key, value)
                    ON key::integer = NEW.inference_idx
                WHERE cls.short_name = CASE
                    WHEN value = 'BACKGROUND' THEN 'NOT_OIL'
                    ELSE value
                END
                LIMIT 1
            )
        );
        RAISE NOTICE 'Calculated NEW.cls. %', (clock_timestamp() - timer)::interval;

        RETURN NEW;
    END;
    $$ LANGUAGE PLPGSQL;
"""


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
    op.execute(SLICK_TRIGGER_FUNC_WITHOUT_AOI_CHUNKS)

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

    op.execute(SLICK_TRIGGER_FUNC_WITH_AOI_CHUNKS)
    op.execute(
        """
        CREATE OR REPLACE RULE bypass_slick_to_aoi_insert
        AS ON INSERT TO public.slick_to_aoi DO INSTEAD NOTHING
        """
    )
