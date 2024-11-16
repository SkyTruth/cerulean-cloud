"""Populate aoi_chunks table

Revision ID: c7c033c1cdb5
Revises: 3736e85bc273
Create Date: 2023-11-17 11:44:59.370910

"""

import sqlalchemy as sa
from sqlalchemy import orm

from alembic import op  # type: ignore

# revision identifiers, used by Alembic.
revision = "c7c033c1cdb5"
down_revision = "3736e85bc273"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Upgrade the database schema.

    This function subdivides the geometries in the 'aoi' table into smaller chunks
    if they have more than 255 points, and inserts these chunks into the 'aoi_chunks' table.
    This is done to manage large geometries more efficiently.
    """
    bind = op.get_bind()
    session = orm.Session(bind=bind)
    with session.begin():
        session.execute(
            sa.text(
                """
                WITH dumped AS (
                    SELECT aoi.id, (st_dump(st_makevalid(st_buffer(geometry::geometry,0)))).geom as dgeom
                    FROM aoi
                ), split as(
                    SELECT id, st_subdivide(dgeom) as dgeom FROM dumped WHERE st_npoints(dgeom)>255
                    UNION ALL
                    SELECT id, dgeom FROM dumped WHERE st_npoints(dgeom)<=255
                )
                INSERT INTO aoi_chunks (id, geometry)
                    SELECT id, dgeom
                    FROM split;
                """
            )
        )


def downgrade() -> None:
    """
    Downgrade the database schema.

    This function clears the 'aoi_chunks' table by truncating it, effectively
    undoing the changes made by the upgrade function.
    """
    bind = op.get_bind()
    session = orm.Session(bind=bind)
    with session.begin():
        session.execute(sa.text("TRUNCATE TABLE aoi_chunks;"))
    pass
