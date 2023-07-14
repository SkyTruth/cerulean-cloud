"""Rework eez

Revision ID: cb7ceecc3f87
Revises: 9c76187d7a13
Create Date: 2022-08-01 16:18:55.163046

"""
from alembic_utils.pg_view import PGView

from alembic import op

# revision identifiers, used by Alembic.
revision = "cb7ceecc3f87"
down_revision = "9c76187d7a13"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """rework eez"""

    op.execute(
        """
        CREATE OR REPLACE FUNCTION map_slick_to_aoi(slick_id bigint, g geography)
        RETURNS void AS $$
        BEGIN
            INSERT INTO slick_to_aoi (slick, aoi)
            SELECT DISTINCT slick_id, aoi.id FROM aoi
            WHERE ST_Intersects(aoi.geometry, g);
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION map_slick_to_aoi_inter()
        RETURNS TRIGGER AS $$
        BEGIN
            PERFORM map_slick_to_aoi(NEW.id, NEW.geometry);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    op.execute(
        """
        CREATE TRIGGER map_slick_to_aoi_trigger
        AFTER INSERT ON slick
        FOR EACH ROW
        EXECUTE FUNCTION map_slick_to_aoi_inter();
    """
    )

    slick_with_source = PGView(
        schema="public",
        signature="slick_with_source",
        definition="""
    SELECT slick.*, slick_source.slick_source, slick_source.slick_source_human_confidence FROM slick
    LEFT JOIN (
        SELECT slick_to_slick_source_slick_source.slick,
                array_agg(slick_to_slick_source_slick_source.name) AS slick_source,
                array_agg(slick_to_slick_source_slick_source.human_confidence) AS slick_source_human_confidence
        FROM (
            SELECT slick_to_slick_source.slick, slick_source.name, slick_to_slick_source.human_confidence
            FROM slick_to_slick_source
            INNER JOIN slick_source
            ON slick_source.id = slick_to_slick_source.slick_source
        ) AS slick_to_slick_source_slick_source
        GROUP BY slick_to_slick_source_slick_source.slick) AS slick_source
    ON slick_source.slick = slick.id;
    """,
    )
    op.create_entity(slick_with_source)


def downgrade() -> None:
    """rework eez"""

    op.execute(
        """
        DROP TRIGGER IF EXISTS map_slick_to_aoi_trigger ON slick;
        """
    )
    op.execute(
        """
        DROP FUNCTION IF EXISTS map_slick_to_aoi_inter();
        """
    )
    op.execute(
        """
        DROP FUNCTION IF EXISTS map_slick_to_aoi(bigint, geography);
        """
    )

    slick_with_source = PGView(
        schema="public",
        signature="slick_with_source",
        definition="// not needed",
    )
    op.drop_entity(slick_with_source)
