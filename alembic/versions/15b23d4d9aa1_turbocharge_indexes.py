"""Update indexes for performance

Revision ID: 15b23d4d9aa1
Revises: c7c033c1cdb5
Create Date: 2025-06-01 00:00:00.000000

"""

from alembic import op

# Alembic versions vary across deployments and older versions do not support
# if_exists/if_not_exists flags on create_index or drop_index.  Using raw SQL
# keeps the migration robust regardless of the library version.

revision = "15b23d4d9aa1"
down_revision = "c7c033c1cdb5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Optimize indexes for frontend performance"""
    # Drop unused indexes if they exist. Using raw SQL avoids errors when an
    # index is already absent.
    for idx in [
        "idx_model_name",
        "idx_trigger_trigger_time",
        "idx_orchestrator_run_time",
        "idx_orchestrator_run_git_tag",
        "idx_orchestrator_run_git_hash",
        "idx_filter_hash",
        "idx_slick_hitl",
        "idx_slick_length",
        "idx_slick_polsby_popper",
        "idx_slick_fill_factor",
        "idx_slick_cls",
        "idx_source_name",
    ]:
        op.execute(f"DROP INDEX IF EXISTS {idx}")

    # New indexes on frequently queried fields
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_ext_id_type ON source (ext_id, type)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_slick_to_source_slick ON slick_to_source (slick)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_slick_to_source_source ON slick_to_source (source)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_slick_to_source_rank ON slick_to_source (rank)"
    )

    # Spatial indexes
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_slick_geometry ON slick USING gist (geometry)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_slick_to_source_geometry ON slick_to_source USING gist (geometry)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sentinel1_grd_geometry ON sentinel1_grd USING gist (geometry)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_orchestrator_run_geometry ON orchestrator_run USING gist (geometry)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_aoi_geometry ON aoi USING gist (geometry)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_aoi_chunks_geometry ON aoi_chunks USING gist (geometry)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_infra_geometry ON source_infra USING gist (geometry)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_dark_geometry ON source_dark USING gist (geometry)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_natural_geometry ON source_natural USING gist (geometry)"
    )


def downgrade() -> None:
    """Revert index changes"""
    for idx in [
        "idx_source_natural_geometry",
        "idx_source_dark_geometry",
        "idx_source_infra_geometry",
        "idx_aoi_chunks_geometry",
        "idx_aoi_geometry",
        "idx_orchestrator_run_geometry",
        "idx_sentinel1_grd_geometry",
        "idx_slick_to_source_geometry",
        "idx_slick_geometry",
        "idx_slick_to_source_rank",
        "idx_slick_to_source_source",
        "idx_slick_to_source_slick",
        "idx_source_ext_id_type",
    ]:
        op.execute(f"DROP INDEX IF EXISTS {idx}")

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_name ON source (st_name, type)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_slick_cls ON slick (cls)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_slick_fill_factor ON slick (fill_factor)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_slick_polsby_popper ON slick (polsby_popper)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_slick_length ON slick (length)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_slick_hitl ON slick (hitl_cls)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_filter_hash ON filter (hash)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_orchestrator_run_git_hash ON orchestrator_run (git_hash)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_orchestrator_run_git_tag ON orchestrator_run (git_tag)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_orchestrator_run_time ON orchestrator_run (inference_start_time, inference_end_time)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_trigger_trigger_time ON trigger (trigger_time)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON model (name)")
