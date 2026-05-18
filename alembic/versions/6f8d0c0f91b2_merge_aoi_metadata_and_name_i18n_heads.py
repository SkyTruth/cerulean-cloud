"""Merge AOI metadata and AOI name i18n heads

Revision ID: 6f8d0c0f91b2
Revises: 1f70e7d0c5b1, 9c73b5c1d2e4
Create Date: 2026-05-18 00:00:00.000000

"""

# revision identifiers, used by Alembic.
revision = "6f8d0c0f91b2"
down_revision = ("1f70e7d0c5b1", "9c73b5c1d2e4")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Merge independent AOI migration heads."""


def downgrade() -> None:
    """Unmerge independent AOI migration heads."""
