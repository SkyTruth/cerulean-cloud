"""Add eez

Revision ID: 5e03ce584f3c
Revises: c941681a050d
Create Date: 2022-07-08 11:24:31.802462

"""
import json

import geojson
import httpx
from shapely import from_geojson, to_wkt
from sqlalchemy import orm
from sqlalchemy.sql import text as _sql_text

import cerulean_cloud.database_schema as database_schema
from alembic import op

# revision identifiers, used by Alembic.
revision = "5e03ce584f3c"
down_revision = "c941681a050d"
branch_labels = None
depends_on = None


def save_eez_to_file():
    """Auxiliary method to save a geojson from marine regions WFS"""
    url = "https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0&request=GetFeature&typeName=eez&outputFormat=json"
    res = geojson.FeatureCollection(**httpx.get(url).json())
    with open("eez.json", "w") as dst:
        geojson.dump(res, dst)


def get_eez_from_url(
    eez_url="https://storage.googleapis.com/ceruleanml/aux_datasets/EEZ_and_HighSeas_20230410.geojson",
):
    """Fetch previously saved file from gcp to avoid interacting with (slow) api"""
    res = geojson.FeatureCollection(**httpx.get(eez_url).json())
    return res


def upgrade() -> None:
    """Add eez"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    eez = get_eez_from_url()  # geojson.load(open("EEZ_and_HighSeas_20230410.json"))
    for feat in eez.features:
        sovereign_keys = [
            k for k in list(feat["properties"].keys()) if k.startswith("SOVEREIGN")
        ]
        sovereigns = [
            feat["properties"][k]
            for k in sovereign_keys
            if feat["properties"][k] is not None
        ]
        with session.begin():
            region = database_schema.Eez(
                mrgid=feat["properties"]["MRGID"],
                geoname=feat["properties"]["GEONAME"],
                sovereigns=sovereigns,
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
    """remove eez"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with session.begin():
        session.query(database_schema.Eez).delete()
