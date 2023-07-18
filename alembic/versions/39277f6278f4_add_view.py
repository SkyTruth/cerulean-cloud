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
    SELECT slick.*, orchestrator_run_with_url.sentinel1_grd_url, orchestrator_run_with_url.infra_distance_url FROM slick
    LEFT JOIN (
        SELECT orchestrator_run.id, sentinel1_grd.url AS sentinel1_grd_url, infra_distance.url AS infra_distance_url
        FROM orchestrator_run
        LEFT JOIN sentinel1_grd
        ON orchestrator_run.sentinel1_grd = sentinel1_grd.id
        LEFT JOIN infra_distance
        ON orchestrator_run.infra_distance = infra_distance.id
    ) AS orchestrator_run_with_url
    ON slick.orchestrator_run = orchestrator_run_with_url.id
    """,
    )
    op.create_entity(slick_with_urls)


def downgrade() -> None:
    """remove views"""
    slick_with_urls = PGView(
        schema="public", signature="slick_with_urls", definition="// not needed"
    )
    op.drop_entity(slick_with_urls)
