"""Add per-source ASA completion times.

Revision ID: 6c09faf08eee
Revises: 9c73b5c1d2e4
"""

import sqlalchemy as sa
from alembic import op

revision = "6c09faf08eee"
down_revision = "9c73b5c1d2e4"
branch_labels = None
depends_on = None

COLUMNS = ("asa_vessel_time", "asa_infra_time", "asa_dark_time", "asa_natural_time")


def upgrade():
    # The fast default stamps legacy runs without rewriting the table.
    # Reopen runs whose AIS window may still be awaiting publication.
    for column in COLUMNS:
        op.add_column(
            "orchestrator_run",
            sa.Column(
                column,
                sa.DateTime(),
                server_default=sa.text("timezone('UTC', now())"),
            ),
        )
        op.alter_column("orchestrator_run", column, server_default=None)

    op.execute("""
        UPDATE orchestrator_run AS run
        SET asa_vessel_time = NULL, asa_infra_time = NULL,
            asa_dark_time = NULL, asa_natural_time = NULL
        FROM sentinel1_grd AS scene
        WHERE run.sentinel1_grd = scene.id
          AND scene.start_time + interval '6 hours'
              > date_trunc('day', now() AT TIME ZONE 'UTC') - interval '3 days'
    """)


def downgrade():
    for column in reversed(COLUMNS):
        op.drop_column("orchestrator_run", column)
