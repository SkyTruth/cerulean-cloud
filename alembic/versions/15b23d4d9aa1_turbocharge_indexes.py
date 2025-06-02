"""Update indexes for performance

Revision ID: 15b23d4d9aa1
Revises: c7c033c1cdb5
Create Date: 2025-06-01 00:00:00.000000

"""

from alembic import op

revision = "15b23d4d9aa1"
down_revision = "c7c033c1cdb5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Optimize indexes for frontend performance"""
    # Drop unused indexes
    op.drop_index("idx_model_name", table_name="model")
    op.drop_index("idx_trigger_trigger_time", table_name="trigger")
    op.drop_index("idx_orchestrator_run_time", table_name="orchestrator_run")
    op.drop_index("idx_orchestrator_run_git_tag", table_name="orchestrator_run")
    op.drop_index("idx_orchestrator_run_git_hash", table_name="orchestrator_run")
    op.drop_index("idx_filter_hash", table_name="filter")
    op.drop_index("idx_slick_hitl", table_name="slick")
    op.drop_index("idx_slick_length", table_name="slick")
    op.drop_index("idx_slick_polsby_popper", table_name="slick")
    op.drop_index("idx_slick_fill_factor", table_name="slick")
    op.drop_index("idx_slick_cls", table_name="slick")
    op.drop_index("idx_slick_to_source_collated_score", table_name="slick_to_source")
    op.drop_index("idx_source_name", table_name="source")

    # New indexes on frequently queried fields
    op.create_index("idx_source_ext_id_type", "source", ["ext_id", "type"])
    op.create_index("idx_slick_to_source_slick", "slick_to_source", ["slick"])
    op.create_index("idx_slick_to_source_source", "slick_to_source", ["source"])
    op.create_index("idx_slick_to_source_rank", "slick_to_source", ["rank"])

    # Spatial indexes
    op.create_index(
        "idx_slick_geometry",
        "slick",
        ["geometry"],
        postgresql_using="gist",
    )
    op.create_index(
        "idx_slick_to_source_geometry",
        "slick_to_source",
        ["geometry"],
        postgresql_using="gist",
    )
    op.create_index(
        "idx_sentinel1_grd_geometry",
        "sentinel1_grd",
        ["geometry"],
        postgresql_using="gist",
    )
    op.create_index(
        "idx_orchestrator_run_geometry",
        "orchestrator_run",
        ["geometry"],
        postgresql_using="gist",
    )
    op.create_index(
        "idx_aoi_geometry",
        "aoi",
        ["geometry"],
        postgresql_using="gist",
    )
    op.create_index(
        "idx_aoi_chunks_geometry",
        "aoi_chunks",
        ["geometry"],
        postgresql_using="gist",
    )
    op.create_index(
        "idx_source_infra_geometry",
        "source_infra",
        ["geometry"],
        postgresql_using="gist",
    )
    op.create_index(
        "idx_source_dark_geometry",
        "source_dark",
        ["geometry"],
        postgresql_using="gist",
    )
    op.create_index(
        "idx_source_natural_geometry",
        "source_natural",
        ["geometry"],
        postgresql_using="gist",
    )


def downgrade() -> None:
    """Revert index changes"""
    op.drop_index("idx_source_natural_geometry", table_name="source_natural")
    op.drop_index("idx_source_dark_geometry", table_name="source_dark")
    op.drop_index("idx_source_infra_geometry", table_name="source_infra")
    op.drop_index("idx_aoi_chunks_geometry", table_name="aoi_chunks")
    op.drop_index("idx_aoi_geometry", table_name="aoi")
    op.drop_index("idx_orchestrator_run_geometry", table_name="orchestrator_run")
    op.drop_index("idx_sentinel1_grd_geometry", table_name="sentinel1_grd")
    op.drop_index("idx_slick_to_source_geometry", table_name="slick_to_source")
    op.drop_index("idx_slick_geometry", table_name="slick")

    op.drop_index("idx_slick_to_source_rank", table_name="slick_to_source")
    op.drop_index("idx_slick_to_source_source", table_name="slick_to_source")
    op.drop_index("idx_slick_to_source_slick", table_name="slick_to_source")
    op.drop_index("idx_source_ext_id_type", table_name="source")

    op.create_index("idx_source_name", "source", ["st_name", "type"])
    op.create_index("idx_slick_to_source_collated_score", "slick_to_source", ["collated_score"])
    op.create_index("idx_slick_cls", "slick", ["cls"])
    op.create_index("idx_slick_fill_factor", "slick", ["fill_factor"])
    op.create_index("idx_slick_polsby_popper", "slick", ["polsby_popper"])
    op.create_index("idx_slick_length", "slick", ["length"])
    op.create_index("idx_slick_hitl", "slick", ["hitl_cls"])
    op.create_index("idx_filter_hash", "filter", ["hash"])
    op.create_index("idx_orchestrator_run_git_hash", "orchestrator_run", ["git_hash"])
    op.create_index("idx_orchestrator_run_git_tag", "orchestrator_run", ["git_tag"])
    op.create_index(
        "idx_orchestrator_run_time",
        "orchestrator_run",
        ["inference_start_time", "inference_end_time"],
    )
    op.create_index("idx_trigger_trigger_time", "trigger", ["trigger_time"])
    op.create_index("idx_model_name", "model", ["name"])
