"""Add view

Revision ID: 39277f6278f4
Revises: 7cd715196b8d
Create Date: 2022-07-01 16:59:54.560440

"""
from alembic_utils.pg_function import PGFunction
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

    # Fiddle: https://dbfiddle.uk/?rdbms=postgres_14&fiddle=a78602f74ea6c2d87a9fa82f1b3a5868
    get_history_slick = PGFunction(
        schema="public",
        signature="slick_history(_slick_id INT)",
        definition="""
        RETURNS TABLE(id integer, slick integer ARRAY, create_time timestamp, active BOOLEAN) as
        $$
        WITH RECURSIVE ctename AS (
            SELECT id, slick, create_time, active
            FROM slick
            WHERE slick.id = _slick_id
            UNION ALL
            SELECT slick.id, slick.slick, slick.create_time, slick.active
            FROM slick
                JOIN ctename ON slick.id = any(ctename.slick)
        )
        SELECT * FROM ctename;
        $$ language SQL;
        """,
    )
    op.create_entity(get_history_slick)


def downgrade() -> None:
    """remove views"""
    slick_with_urls = PGView(
        schema="public", signature="slick_with_urls", definition="// not needed"
    )
    op.drop_entity(slick_with_urls)

    get_history_slick = PGFunction(
        schema="public",
        signature="slick_history(_slick_id INT)",
        definition="// not needed",
    )
    op.drop_entity(get_history_slick)
