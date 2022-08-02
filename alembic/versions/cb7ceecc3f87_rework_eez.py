"""Rework eez

Revision ID: cb7ceecc3f87
Revises: 9c76187d7a13
Create Date: 2022-08-01 16:18:55.163046

"""
import sqlalchemy as sa
from alembic_utils.pg_function import PGFunction, PGView
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

    slick_with_eez_and_source = PGView(
        schema="public",
        signature="slick_with_eez_and_source",
        definition="// not needed",
    )

    op.drop_entity(slick_with_eez_and_source)

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

    slick_with_eez_and_source = PGView(
        schema="public",
        signature="slick_with_eez_and_source",
        definition="""
    SELECT slick.*, eez.eez, slick_source.slick_source, slick_source.slick_source_human_confidence FROM slick
    LEFT JOIN (
        SELECT slick_to_eez_eez.slick, array_agg(slick_to_eez_eez.geoname) AS eez
        FROM (
            SELECT slick_to_eez.slick, eez.geoname
            FROM slick_to_eez
            INNER JOIN eez
            ON eez.id = slick_to_eez.eez
        ) AS slick_to_eez_eez
        GROUP BY slick_to_eez_eez.slick) AS eez
    ON eez.slick = slick.id
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
    op.create_entity(slick_with_eez_and_source)

    slick_with_source = PGView(
        schema="public",
        signature="slick_with_source",
        definition="// not needed",
    )
    op.drop_entity(slick_with_source)
