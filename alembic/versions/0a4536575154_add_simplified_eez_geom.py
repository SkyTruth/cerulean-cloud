"""Add simplified eez geom



Revision ID: 0a4536575154
Revises: 04058775366b
Create Date: 2022-07-11 13:32:18.803010

"""
import sqlalchemy as sa
from geoalchemy2 import Geography

from alembic import op

# revision identifiers, used by Alembic.
revision = "0a4536575154"
down_revision = "04058775366b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add simplified eez geom"""
    op.add_column(
        "eez",
        sa.Column(
            "geometry_005",
            Geography,
            sa.Computed("ST_Simplify(geometry::geometry, 0.05)::geography"),
        ),
    )


def downgrade() -> None:
    """Remove simplified eez geom"""
    op.drop_column("eez", "geometry_005")
