"""Add funcs on insert

Revision ID: cb7ceecc3f87
Revises: 5e03ce584f3c
Create Date: 2022-08-01 16:18:55.163046

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "cb7ceecc3f87"
down_revision = "5e03ce584f3c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add funcs on insert"""

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


def downgrade() -> None:
    """Add funcs on insert"""

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
