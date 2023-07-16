"""Add infra sources

Revision ID: 9622ae2a4a04
Revises: f9b7166c86b7
Create Date: 2023-07-16 00:35:23.124372

"""
import json
import random
import string

import geojson
import httpx
from shapely import from_geojson, to_wkt
from sqlalchemy import orm

import cerulean_cloud.database_schema as database_schema
from alembic import op

# revision identifiers, used by Alembic.
revision = "9622ae2a4a04"
down_revision = "f9b7166c86b7"
branch_labels = None
depends_on = None


def get_infra_from_url(
    infra_url="https://storage.googleapis.com/ceruleanml/aux_datasets/Global%20Coincident%20Infrastructure.geojson",
):
    """Fetch previously saved file from gcp to avoid interacting with (slow) api"""
    res = geojson.FeatureCollection(**httpx.get(infra_url).json())
    return res


def geom_to_st_name(geom):
    """Given the geom from the geojson, generate a unique name"""
    lon, lat = geom["coordinates"]
    random_string = "".join(random.choices(string.ascii_uppercase, k=3))
    return f"{abs(lon):.2f}{'E' if lon > 0 else 'W'}_{abs(lat):.2f}{'N' if lat > 0 else 'S'}_{random_string}"


def upgrade() -> None:
    """Add infra sources"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    infra = get_infra_from_url()

    for feat in infra.get("features"):
        with session.begin():
            source_infra = database_schema.SourceInfra(
                type=2,
                st_name=geom_to_st_name(feat["geometry"]),
                geometry=to_wkt(from_geojson(json.dumps(feat["geometry"]))),
            )
            session.add(source_infra)


def downgrade() -> None:
    """remove infra sources"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with session.begin():
        session.query(database_schema.SourceInfra).delete()
        session.query(database_schema.Source).filter(
            database_schema.Source.type == 2
        ).delete()
