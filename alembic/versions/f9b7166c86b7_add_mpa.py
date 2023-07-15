"""Add MPA

Revision ID: f9b7166c86b7
Revises: c0bd1215a3ca
Create Date: 2023-07-15 01:52:45.298587

"""
import json

# import geojson
# import httpx
from shapely import from_geojson, to_wkt
from sqlalchemy import orm

import cerulean_cloud.database_schema as database_schema
from alembic import op

# revision identifiers, used by Alembic.
revision = "f9b7166c86b7"
down_revision = "c0bd1215a3ca"
branch_labels = None
depends_on = None


def get_mpa_from_url(
    mpa_url="XXX TODO UPLOAD AND ADD HERE",
):
    """Fetch previously saved file from gcp to avoid interacting with (slow) api"""
    # res = geojson.FeatureCollection(**httpx.get(mpa_url).json())
    res = {"features": []}
    return res


def upgrade() -> None:
    """Add mpa"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    mpa = get_mpa_from_url()

    for feat in mpa.get("features"):
        with session.begin():
            aoi_mpa = database_schema.AoiMpa(
                type=3,
                name=feat["properties"]["NAME"],
                geometry=to_wkt(from_geojson(json.dumps(feat["geometry"]))),
                wdpaid=1234,
                desig="desig",
                desig_type="desig_type",
                status_yr=2023,
                mang_auth="mang_auth",
                parent_iso="parent_iso",
            )
            session.add(aoi_mpa)


def downgrade() -> None:
    """remove mpa"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with session.begin():
        session.query(database_schema.AoiMpa).delete()
        session.query(database_schema.Aoi).filter(
            database_schema.Aoi.type == 3
        ).delete()
