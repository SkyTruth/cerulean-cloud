"""Install pg_stat_statements and tighten hot-table stats hygiene.

Revision ID: a6e4f8d91b9c
Revises: 8f0c0f3f1f6d
Create Date: 2026-03-09 09:30:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "a6e4f8d91b9c"
down_revision = "8f0c0f3f1f6d"
branch_labels = None
depends_on = None


HOT_TABLES = (
    "slick",
    "slick_to_source",
    "slick_to_aoi",
    "orchestrator_run",
    "sentinel1_grd",
)

HOT_TABLE_REL_OPTIONS = (
    ("autovacuum_enabled", "true"),
    ("toast.autovacuum_enabled", "true"),
    ("autovacuum_vacuum_scale_factor", "0.02"),
    ("autovacuum_vacuum_threshold", "500"),
    ("autovacuum_vacuum_insert_scale_factor", "0.02"),
    ("autovacuum_vacuum_insert_threshold", "500"),
    ("autovacuum_analyze_scale_factor", "0.01"),
    ("autovacuum_analyze_threshold", "250"),
)

REL_OPTION_NAMES = ", ".join(name for name, _ in HOT_TABLE_REL_OPTIONS)


def _set_hot_table_reloptions(table_name: str) -> None:
    option_sql = ", ".join(f"{name} = {value}" for name, value in HOT_TABLE_REL_OPTIONS)
    op.execute(f"ALTER TABLE public.{table_name} SET ({option_sql});")


def _reset_hot_table_reloptions(table_name: str) -> None:
    op.execute(f"ALTER TABLE public.{table_name} RESET ({REL_OPTION_NAMES});")


def upgrade() -> None:
    """Install pg_stat_statements and keep planner stats fresher on hot tables."""
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements;")

    for table_name in HOT_TABLES:
        _set_hot_table_reloptions(table_name)
        op.execute(f"ANALYZE public.{table_name};")


def downgrade() -> None:
    """Remove hot-table reloptions and pg_stat_statements."""
    for table_name in HOT_TABLES:
        _reset_hot_table_reloptions(table_name)

    op.execute("DROP EXTENSION IF EXISTS pg_stat_statements;")
