"""Add eez

Revision ID: 5e03ce584f3c
Revises: c941681a050d
Create Date: 2022-07-08 11:24:31.802462

"""

import geojson
import httpx
import sqlalchemy as sa
from shapely.geometry import MultiPolygon, shape

from alembic import op

# revision identifiers, used by Alembic.
revision = "5e03ce584f3c"
down_revision = "c941681a050d"
branch_labels = None
depends_on = None
AOI_TYPE_SHORT_NAME = "EEZ"


def save_eez_to_file():
    """Auxiliary method to save a geojson from marine regions WFS"""
    url = "https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0&request=GetFeature&typeName=eez&outputFormat=json"
    res = geojson.FeatureCollection(**httpx.get(url).json())
    with open("eez.json", "w") as dst:
        geojson.dump(res, dst)


def get_eez_from_url(
    eez_url="https://storage.googleapis.com/ceruleanml/aux_datasets/EEZ_and_HighSeas_20230410_split.geojson",
):
    """Fetch previously saved file from gcp to avoid interacting with (slow) api"""
    res = geojson.FeatureCollection(**httpx.get(eez_url).json())
    return res


def _aoi_type_id(bind) -> int:
    return bind.execute(
        sa.text("SELECT id FROM aoi_type WHERE short_name = :short_name"),
        {"short_name": AOI_TYPE_SHORT_NAME},
    ).scalar_one()


def _multipolygon_wkt(feature_geometry) -> str:
    geometry = shape(feature_geometry).buffer(0)
    if not isinstance(geometry, MultiPolygon):
        geometry = MultiPolygon([geometry])
    return geometry.wkt


def upgrade() -> None:
    """Add eez"""
    bind = op.get_bind()
    aoi_type_id = _aoi_type_id(bind)

    eez = get_eez_from_url()  # geojson.load(open("EEZ_and_HighSeas_20230410.json"))
    for feat in eez.get("features"):
        sovereign_keys = [
            k for k in list(feat["properties"].keys()) if k.startswith("SOVEREIGN")
        ]
        sovereigns = [
            feat["properties"][k]
            for k in sovereign_keys
            if feat["properties"][k] is not None
        ]

        aoi_id = bind.execute(
            sa.text("""
                INSERT INTO aoi (type, name, geometry)
                VALUES (:type, :name, ST_GeogFromText(:geometry))
                RETURNING id
                """),
            {
                "type": aoi_type_id,
                "name": feat["properties"]["GEONAME"],
                "geometry": f"SRID=4326;{_multipolygon_wkt(feat['geometry'])}",
            },
        ).scalar_one()
        bind.execute(
            sa.text("""
                INSERT INTO aoi_eez (aoi_id, mrgid, sovereigns)
                VALUES (:aoi_id, :mrgid, :sovereigns)
                """),
            {
                "aoi_id": aoi_id,
                "mrgid": feat["properties"]["MRGID"],
                "sovereigns": sovereigns,
            },
        )


def downgrade() -> None:
    """remove eez"""
    bind = op.get_bind()
    bind.execute(sa.text("DELETE FROM aoi_eez"))
    bind.execute(
        sa.text("""
            DELETE FROM aoi
            WHERE type = (
                SELECT id
                FROM aoi_type
                WHERE short_name = :short_name
            )
            """),
        {"short_name": AOI_TYPE_SHORT_NAME},
    )
