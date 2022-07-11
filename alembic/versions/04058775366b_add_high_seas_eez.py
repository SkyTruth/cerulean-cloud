"""Add high seas eez

Revision ID: 04058775366b
Revises: 5e03ce584f3c
Create Date: 2022-07-11 11:25:36.754357

"""
import geojson
import httpx
from geoalchemy2.shape import from_shape
from shapely.geometry import shape
from sqlalchemy import orm

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
                geometry=from_shape(shape(feat["geometry"])),
            )
            session.add(region)


def downgrade() -> None:
    """Remove high sea eez"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with session.begin():
        session.query(database_schema.Eez).filter(name="High Seas").delete()
