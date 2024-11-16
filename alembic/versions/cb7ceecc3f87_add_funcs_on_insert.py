"""Add funcs on insert

Revision ID: cb7ceecc3f87
Revises: 5e03ce584f3c
Create Date: 2022-08-01 16:18:55.163046

"""

from sqlalchemy import text

from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision = "cb7ceecc3f87"
down_revision = "5e03ce584f3c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add funcs on insert"""

    op.execute(
        text(
            """
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
            NEW.cls := (
                SELECT cls.id
                FROM cls
                JOIN orchestrator_run ON NEW.orchestrator_run = orchestrator_run.id
                JOIN LATERAL json_each_text((SELECT cls_map FROM model WHERE id = orchestrator_run.model))
                    m(key, value)
                    ON key::integer = NEW.inference_idx
                    WHERE cls.short_name = value
                LIMIT 1
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

        CREATE TRIGGER slick_before_trigger BEFORE INSERT ON slick FOR EACH ROW EXECUTE FUNCTION  slick_before_trigger_func();

        SET CLIENT_MIN_MESSAGES TO NOTICE;
    """
        )
    )


def downgrade() -> None:
    """Add funcs on insert"""
    op.execute(
        """
        DROP TRIGGER IF EXISTS slick_before_trigger ON slick;
        """
    )
    op.execute(
        """
        DROP FUNCTION IF EXISTS slick_before_trigger_func();
        """
    )
