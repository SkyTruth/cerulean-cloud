"""Add the hot-path index pack with concurrent builds.

Revision ID: b7c9e2f4d1aa
Revises: a6e4f8d91b9c
Create Date: 2026-03-09 11:10:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "b7c9e2f4d1aa"
down_revision = "a6e4f8d91b9c"
branch_labels = None
depends_on = None


INDEX_STATEMENTS = (
    (
        "idx_hitl_slick_slick_update_desc",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hitl_slick_slick_update_desc
        ON public.hitl_slick (slick, update_time DESC);
        """,
    ),
    (
        "idx_hitl_slick_user_slick_update_desc",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hitl_slick_user_slick_update_desc
        ON public.hitl_slick ("user", slick, update_time DESC);
        """,
    ),
    (
        "idx_sts_active_source_rank_slick",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sts_active_source_rank_slick
        ON public.slick_to_source (source, rank, slick)
        WHERE active;
        """,
    ),
    (
        "idx_sts_active_slick_score_desc",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sts_active_slick_score_desc
        ON public.slick_to_source (slick, collated_score DESC)
        WHERE active;
        """,
    ),
    (
        "idx_sts_active_slick_rank",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sts_active_slick_rank
        ON public.slick_to_source (slick, rank)
        WHERE active;
        """,
    ),
    (
        "idx_hitl_request_user_date_desc",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hitl_request_user_date_desc
        ON public.hitl_request ("user", date_requested DESC);
        """,
    ),
    (
        "idx_source_vessel_flag_source",
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_source_vessel_flag_source
        ON public.source_vessel (flag, source_id);
        """,
    ),
)


def _run_non_transactional(statement: str) -> None:
    with op.get_context().autocommit_block():
        op.execute(statement)


def upgrade() -> None:
    """Create the hot-path indexes without blocking writers."""
    for _, statement in INDEX_STATEMENTS:
        _run_non_transactional(statement)


def downgrade() -> None:
    """Drop the hot-path indexes without blocking writers."""
    for index_name, _ in reversed(INDEX_STATEMENTS):
        _run_non_transactional(
            f"DROP INDEX CONCURRENTLY IF EXISTS public.{index_name};"
        )
