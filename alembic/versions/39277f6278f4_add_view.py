"""Add view

Revision ID: 39277f6278f4
Revises: 7cd715196b8d
Create Date: 2022-07-01 16:59:54.560440

"""
from alembic_utils.pg_view import PGView

from alembic import op

# revision identifiers, used by Alembic.
revision = "39277f6278f4"
down_revision = "7cd715196b8d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """add views"""
    # Fiddle: https://dbfiddle.uk/?rdbms=postgres_14&fiddle=d63d3e9dbfa5522d65076c4f8863b737
    slick_with_urls = PGView(
        schema="public",
        signature="slick_with_urls",
        definition="""
    SELECT slick.*, orchestrator_run_with_url.sentinel1_grd_url FROM slick
    LEFT JOIN (
        SELECT orchestrator_run.id, sentinel1_grd.url AS sentinel1_grd_url
        FROM orchestrator_run
        LEFT JOIN sentinel1_grd
        ON orchestrator_run.sentinel1_grd = sentinel1_grd.id
    ) AS orchestrator_run_with_url
    ON slick.orchestrator_run = orchestrator_run_with_url.id
    """,
    )
    op.create_entity(slick_with_urls)

    slick_plus = PGView(
        schema="public",
        signature="slick_plus",
        definition="""
    SELECT
        slick.*,
        sentinel1_grd.scene_id AS s1_scene_id,
        sentinel1_grd.geometry AS s1_geometry,
        cls.short_name AS cls_short_name,
        cls.long_name AS cls_long_name,
        aoi_agg.aoi_type_1_ids,
        aoi_agg.aoi_type_2_ids,
        aoi_agg.aoi_type_3_ids
    FROM slick
    JOIN orchestrator_run ON orchestrator_run.id = slick.orchestrator_run
    JOIN sentinel1_grd ON sentinel1_grd.id = orchestrator_run.sentinel1_grd
    JOIN cls ON cls.id = slick.cls
    JOIN (
        SELECT slick_to_aoi.slick,
            array_agg(aoi.id) FILTER (WHERE aoi.type = 1) AS aoi_type_1_ids,
            array_agg(aoi.id) FILTER (WHERE aoi.type = 2) AS aoi_type_2_ids,
            array_agg(aoi.id) FILTER (WHERE aoi.type = 3) AS aoi_type_3_ids
        FROM slick_to_aoi
        JOIN aoi ON slick_to_aoi.aoi = aoi.id
        GROUP BY slick_to_aoi.slick
        ) aoi_agg ON aoi_agg.slick = slick.id;
    """,
    )
    op.create_entity(slick_plus)


def downgrade() -> None:
    """remove views"""
    slick_with_urls = PGView(
        schema="public", signature="slick_with_urls", definition="// not needed"
    )
    op.drop_entity(slick_with_urls)

    slick_plus = PGView(
        schema="public", signature="slick_plus", definition="// not needed"
    )
    op.drop_entity(slick_plus)
