"""Add exposed funcs

Revision ID: 3736e85bc273
Revises: 9622ae2a4a04
Create Date: 2023-07-18 01:07:58.731501

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "3736e85bc273"
down_revision = "9622ae2a4a04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add funcs"""
    op.execute(
        """
        CREATE OR REPLACE FUNCTION get_slick_subclasses(class_id bigint)
        RETURNS SETOF class AS $$
        BEGIN
            RETURN QUERY (
                WITH RECURSIVE recurse_classes_cte AS (
                    SELECT id FROM class WHERE id = class_id
                    UNION
                    SELECT cls.id FROM class cls
                    JOIN recurse_classes_cte rc_cte ON cls.superclass = rc_cte.id
                )
                SELECT * FROM recurse_classes_cte
            );
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def downgrade() -> None:
    """Add funcs"""
    op.execute(
        """
        DROP FUNCTION IF EXISTS get_slick_subclasses(bigint);
        """
    )
