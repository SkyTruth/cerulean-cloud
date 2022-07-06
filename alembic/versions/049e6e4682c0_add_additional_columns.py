"""Add additional columns

Revision ID: 049e6e4682c0
Revises: 39277f6278f4
Create Date: 2022-07-05 11:05:46.836225

"""
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "049e6e4682c0"
down_revision = "39277f6278f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """add missing columns"""
    op.add_column("orchestrator_run", sa.Column("zoom", sa.Integer))
    op.add_column("orchestrator_run", sa.Column("scale", sa.Integer))
    op.add_column("orchestrator_run", sa.Column("success", sa.Boolean))


def downgrade() -> None:
    """remove missing columns"""
    op.drop_column("orchestrator_run", "zoom")
    op.drop_column("orchestrator_run", "scale")
    op.drop_column("orchestrator_run", "success")
