"""Add postgis

Revision ID: 54c42e9e879f
Revises:
Create Date: 2022-06-27 16:34:29.607839

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "54c42e9e879f"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """create postgis extension"""
    op.execute("CREATE EXTENSION postgis;")
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;")
    op.execute(
        "COMMENT ON EXTENSION postgis IS 'PostGIS geometry, geography, and raster spatial types and functions';"
    )


def downgrade() -> None:
    """drop postgis extension"""
    op.execute("DROP EXTENSION postgis;")
