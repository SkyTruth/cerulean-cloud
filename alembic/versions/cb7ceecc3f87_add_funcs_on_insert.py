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

    # Create a trigger function that maps slick to aoi upon insert
    op.execute(
        """
        CREATE OR REPLACE FUNCTION map_slick_to_aoi()
        RETURNS TRIGGER AS $$
        BEGIN
            INSERT INTO slick_to_aoi (slick, aoi)
            SELECT DISTINCT NEW.id, aoi.id FROM aoi
            WHERE ST_Intersects(aoi.geometry, NEW.geometry);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # Create a trigger that calls the trigger function after insert
    op.execute(
        """
        CREATE TRIGGER trigger_map_slick_to_aoi
        AFTER INSERT ON slick
        FOR EACH ROW
        EXECUTE FUNCTION map_slick_to_aoi();
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION populate_inferred_cls()
        RETURNS TRIGGER AS
        $$
        BEGIN
            NEW.inferred_cls := (
                SELECT cls.id
                FROM cls
                JOIN orchestrator_run ON NEW.orchestrator_run = orchestrator_run.id
                JOIN LATERAL json_each_text((SELECT class_map FROM model WHERE id = orchestrator_run.model))
                    m(key, value)
                    ON key::integer = NEW.inference_idx
                    WHERE cls.short_name = value
                LIMIT 1
            );
            RETURN NEW;
        END;
        $$
        LANGUAGE plpgsql;
        """
    )

    op.execute(
        """
        CREATE TRIGGER trigger_populate_inferred_cls
        BEFORE INSERT ON slick
        FOR EACH ROW
        EXECUTE FUNCTION populate_inferred_cls();
        """
    )


def downgrade() -> None:
    """Add funcs on insert"""

    op.execute(
        """
        DROP TRIGGER IF EXISTS trigger_map_slick_to_aoi ON slick;
        """
    )
    op.execute(
        """
        DROP FUNCTION IF EXISTS map_slick_to_aoi();
        """
    )
    op.execute(
        """
        DROP TRIGGER IF EXISTS trigger_populate_inferred_cls ON slick;
        """
    )
    op.execute(
        """
        DROP FUNCTION IF EXISTS populate_inferred_cls();
        """
    )
