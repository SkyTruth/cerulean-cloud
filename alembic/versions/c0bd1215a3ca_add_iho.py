"""Add IHO

Revision ID: c0bd1215a3ca
Revises: cb7ceecc3f87
Create Date: 2023-07-15 00:26:04.493750

"""
import json
from typing import Any, Dict

import geojson
import httpx
from shapely import from_geojson, to_wkt
from sqlalchemy import orm

import cerulean_cloud.database_schema as database_schema
from alembic import op

# revision identifiers, used by Alembic.
revision = "c0bd1215a3ca"
down_revision = "cb7ceecc3f87"
branch_labels = None
depends_on = None


def get_iho_from_url(
    iho_url="https://storage.googleapis.com/ceruleanml/aux_datasets/World_Seas_IHO_v3.deleteholes.simplify.repair3.caspian.geojson",
):
    """Fetch previously saved file from gcp to avoid interacting with (slow) api"""
    res = geojson.FeatureCollection(**httpx.get(iho_url).json())
    return res


def upgrade() -> None:
    """Add iho"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    # iho = get_iho_from_url()
    for feat in []:  # type: Dict[str, Any]
        with session.begin():
            aoi_iho = database_schema.AoiIho(
                type=2,
                name=feat["properties"]["NAME"],
                geometry=to_wkt(from_geojson(json.dumps(feat["geometry"]))),
                mrgid=feat["properties"]["MRGID"],
            )
            session.add(aoi_iho)


def downgrade() -> None:
    """remove iho"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with session.begin():
        session.query(database_schema.AoiIho).delete()
        session.query(database_schema.Aoi).filter(
            database_schema.Aoi.type == 2
        ).delete()
