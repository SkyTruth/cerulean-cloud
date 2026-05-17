"""Add MPA

Revision ID: f9b7166c86b7
Revises: c0bd1215a3ca
Create Date: 2023-07-15 01:52:45.298587

"""

import geojson
import httpx
import sqlalchemy as sa
from shapely.geometry import MultiPolygon, shape

from alembic import op

# revision identifiers, used by Alembic.
revision = "f9b7166c86b7"
down_revision = "c0bd1215a3ca"
branch_labels = None
depends_on = None
AOI_TYPE_SHORT_NAME = "MPA"


def get_mpa_from_url(
    mpa_url="https://storage.googleapis.com/ceruleanml/aux_datasets/mpa_all_deleteholes_simplify_repair1.geojson",
):
    """Fetch previously saved file from gcp to avoid interacting with (slow) api"""
    res = geojson.FeatureCollection(**httpx.get(mpa_url).json())
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
    """Add mpa"""
    bind = op.get_bind()
    aoi_type_id = _aoi_type_id(bind)

    mpa = get_mpa_from_url()
    for feat in mpa.get("features"):
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
                INSERT INTO aoi_mpa (
                    aoi_id,
                    wdpaid,
                    desig,
                    desig_type,
                    status_yr,
                    mang_auth,
                    parent_iso
                )
                VALUES (
                    :aoi_id,
                    :wdpaid,
                    :desig,
                    :desig_type,
                    :status_yr,
                    :mang_auth,
                    :parent_iso
                )
                """),
            {
                "aoi_id": aoi_id,
                "wdpaid": feat["properties"]["WDPAID"],
                "desig": feat["properties"]["DESIG"],
                "desig_type": feat["properties"]["DESIG_TYPE"],
                "status_yr": feat["properties"]["STATUS_YR"],
                "mang_auth": feat["properties"]["MANG_AUTH"],
                "parent_iso": feat["properties"]["PARENT_ISO"],
            },
        )


def downgrade() -> None:
    """remove mpa"""
    bind = op.get_bind()
    bind.execute(sa.text("DELETE FROM aoi_mpa"))
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
