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
        CREATE OR REPLACE FUNCTION get_slick_subclses(cls_id bigint)
        RETURNS SETOF cls AS $$
        BEGIN
            RETURN QUERY (
                WITH RECURSIVE recurse_clses_cte AS (
                    SELECT * FROM cls WHERE id = cls_id
                    UNION
                    SELECT nextcls.* FROM cls nextcls
                    JOIN recurse_clses_cte rc ON nextcls.supercls = rc.id
                )
                SELECT * FROM recurse_clses_cte
            );
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def downgrade() -> None:
    """Add funcs"""
    op.execute(
        """
        DROP FUNCTION IF EXISTS get_slick_subclses(bigint);
        """
    )
