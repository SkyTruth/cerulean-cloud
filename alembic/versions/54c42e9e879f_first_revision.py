"""First revision

Revision ID: 54c42e9e879f
Revises:
Create Date: 2022-06-27 16:34:29.607839

"""
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "54c42e9e879f"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """create test table"""
    op.create_table(
        "test",
        sa.Column("id", sa.Integer, primary_key=True),
    )


def downgrade() -> None:
    """drop test table"""
    op.drop_table("test")
