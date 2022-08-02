"""Rework eez

Revision ID: cb7ceecc3f87
Revises: 9c76187d7a13
Create Date: 2022-08-01 16:18:55.163046

"""
import sqlalchemy as sa
from alembic_utils.pg_function import PGFunction
from geoalchemy2 import Geography
from sqlalchemy.types import ARRAY

from alembic import op

# revision identifiers, used by Alembic.
revision = "cb7ceecc3f87"
down_revision = "9c76187d7a13"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """rework eez"""
    op.create_table(
        "eez_parts",
        sa.Column("partid", sa.BigInteger, primary_key=True),
        sa.Column("geoname", sa.Text),
        sa.Column("geometry", Geography("MULTIPOLYGON"), nullable=False),
    )
    op.execute(
        """
    INSERT INTO eez_parts (geoname, geometry)
    SELECT geoname, st_subdivide(st_makevalid((st_dump(geometry_005::geometry)).geom))::geography AS geometry FROM eez;
    """
    )

    eezs = PGFunction(
        schema="public",
        signature="eezs(g geography)",
        definition="""
        RETURNS text[] AS $$
        SELECT array_agg(distinct geoname) FROM eez_parts
        WHERE ST_Intersects(geometry, g);
        $$ LANGUAGE SQL STABLE;
        """,
    )
    op.create_entity(eezs)

    op.add_column(
        "slick",
        sa.Column(
            "eezs",
            ARRAY(sa.Text),
            sa.Computed("eezs(geometry)"),
        ),
    )

    op.drop_table("slick_to_eez")


def downgrade() -> None:
    """rework eez"""
    op.drop_table("eez_parts")

    eezs = PGFunction(
        schema="public",
        signature="eezs(g geography)",
        definition="// not needed",
    )
    op.drop_entity(eezs)

    op.create_table(
        "slick_to_eez",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("slick", sa.BigInteger, sa.ForeignKey("slick.id"), nullable=False),
        sa.Column("eez", sa.BigInteger, sa.ForeignKey("eez.id"), nullable=False),
    )
    op.drop_column("slick", "eezs")
