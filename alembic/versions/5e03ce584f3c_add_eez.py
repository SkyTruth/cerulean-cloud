"""Add eez

Revision ID: 5e03ce584f3c
Revises: c941681a050d
Create Date: 2022-07-08 11:24:31.802462

"""
import json

import geoalchemy2.functions as func
import geojson
import httpx
from geoalchemy2 import Geography, Geometry
from geoalchemy2.shape import from_shape
from shapely import from_geojson, to_wkt
from shapely.geometry import box
from sqlalchemy import orm
from sqlalchemy.sql import cast
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
    eez_url="https://storage.googleapis.com/ceruleanml/aux_datasets/eez_8_7_2022.json",
):
    """Fetch previously saved file from gcp to avoid interacting with (slow) api"""
    res = geojson.FeatureCollection(**httpx.get(eez_url).json())
    return res


def upgrade() -> None:
    """Add eez"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    eez = get_eez_from_url()  # geojson.load(open("eez_8_7_2022.json"))
    for feat in eez.features:
        sovereign_keys = [
            k for k in list(feat["properties"].keys()) if k.startswith("sovereign")
        ]
        sovereigns = [
            feat["properties"][k]
            for k in sovereign_keys
            if feat["properties"][k] is not None
        ]
        with session.begin():
            region = database_schema.Eez(
                mrgid=feat["properties"]["mrgid"],
                geoname=feat["properties"]["geoname"],
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

    # Add inverted EEZ (no sovereign)
    # Too long to compute, so disabled
    if False:
        with session.begin():
            outer_geom = session.query(
                cast(
                    func.ST_Difference(
                        cast(
                            from_shape(box(*[-179, -89, 179, 89])), Geometry(srid=4326)
                        ),
                        func.ST_Union(
                            cast(database_schema.Eez.geometry, Geometry(srid=4326))
                        ),
                    ),
                    Geography(srid=4326),
                )
            )
            outer_region = database_schema.Eez(
                mrgid=10000000,
                geoname="International Waters",
                sovereigns=["None"],
                geometry=outer_geom,
            )
            session.add(outer_region)


def downgrade() -> None:
    """remove eez"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with session.begin():
        session.query(database_schema.Eez).delete()
