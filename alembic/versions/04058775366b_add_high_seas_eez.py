"""Add high seas eez

Revision ID: 04058775366b
Revises: 5e03ce584f3c
Create Date: 2022-07-11 11:25:36.754357

"""
import json

import geojson
import httpx
from shapely import from_geojson, to_wkt
from sqlalchemy import orm
from sqlalchemy import text as _sql_text

import cerulean_cloud.database_schema as database_schema
from alembic import op

# revision identifiers, used by Alembic.
revision = "04058775366b"
down_revision = "5e03ce584f3c"
branch_labels = None
depends_on = None


def get_eez_from_url(
    eez_url="https://storage.googleapis.com/ceruleanml/aux_datasets/highseas_11_7_2022.json",
):
    """Fetch previously saved file from gcp to avoid interacting with (slow) api"""
    res = geojson.FeatureCollection(**httpx.get(eez_url).json())
    return res


def upgrade() -> None:
    """Add high sea eez"""

    bind = op.get_bind()
    session = orm.Session(bind=bind)

    high_seas = get_eez_from_url()

    for feat in high_seas.features:
        with session.begin():
            region = database_schema.Eez(
                mrgid=feat["properties"]["mrgid"],
                geoname=feat["properties"]["name"],
                sovereigns=["None"],
                geometry=to_wkt(from_geojson(json.dumps(feat["geometry"]))),
            )
            sql_string = _sql_text(
                """
                INSERT INTO eez (mrgid, geoname, sovereigns, geometry)
                VALUES (
                    :mrgid,
                    :geoname,
                    :sovereigns,
                    st_geogfromtext(:geometry)
                )
            """
            )

            session.execute(
                sql_string,
                {
                    "mrgid": region.mrgid,
                    "geoname": region.geoname,
                    "sovereigns": region.sovereigns,
                    "geometry": region.geometry,
                },
            )


def downgrade() -> None:
    """Remove high sea eez"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with session.begin():
        session.query(database_schema.Eez).filter_by(geoname="High Seas").delete()
