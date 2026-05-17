"""Add IHO

Revision ID: c0bd1215a3ca
Revises: cb7ceecc3f87
Create Date: 2023-07-15 00:26:04.493750

"""

import geojson
import httpx
import sqlalchemy as sa
from shapely.geometry import MultiPolygon, shape

from alembic import op

# revision identifiers, used by Alembic.
revision = "c0bd1215a3ca"
down_revision = "cb7ceecc3f87"
branch_labels = None
depends_on = None
AOI_TYPE_SHORT_NAME = "IHO"


def get_iho_from_url(
    iho_url="https://storage.googleapis.com/ceruleanml/aux_datasets/World_Seas_IHO_v3.deleteholes.simplify.repair3.caspian.geojson",
):
    """Fetch previously saved file from gcp to avoid interacting with (slow) api"""
    res = geojson.FeatureCollection(**httpx.get(iho_url).json())
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
    """Add iho"""
    bind = op.get_bind()
    aoi_type_id = _aoi_type_id(bind)

    iho = get_iho_from_url()

    for feat in iho.get("features"):
        aoi_id = bind.execute(
            sa.text("""
                INSERT INTO aoi (type, name, geometry)
                VALUES (:type, :name, ST_GeogFromText(:geometry))
                RETURNING id
                """),
            {
                "type": aoi_type_id,
                "name": feat["properties"]["NAME"],
                "geometry": f"SRID=4326;{_multipolygon_wkt(feat['geometry'])}",
            },
        ).scalar_one()
        bind.execute(
            sa.text("""
                INSERT INTO aoi_iho (aoi_id, mrgid)
                VALUES (:aoi_id, :mrgid)
                """),
            {"aoi_id": aoi_id, "mrgid": feat["properties"]["MRGID"]},
        )


def downgrade() -> None:
    """remove iho"""
    bind = op.get_bind()
    bind.execute(sa.text("DELETE FROM aoi_iho"))
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
