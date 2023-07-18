"""Add index

Revision ID: 7cd715196b8d
Revises: 3c4693517ef6
Create Date: 2022-07-01 14:03:52.485218

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "7cd715196b8d"
down_revision = "3c4693517ef6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """add indices"""
    op.create_index("idx_model_name", "model", ["name"])
    op.create_index("idx_model_file_path", "model", ["file_path"])

    op.create_index(
        "idx_sentinel1_grd_scene_id", "sentinel1_grd", ["scene_id"], unique=True
    )

    op.create_index("idx_vessel_density_name", "vessel_density", ["name"])

    op.create_index("idx_infra_distance_name", "infra_distance", ["name"])

    op.create_index("idx_trigger_trigger_time", "trigger", ["trigger_time"])

    op.create_index(
        "idx_orchestrator_run_time",
        "orchestrator_run",
        ["inference_start_time", "inference_end_time"],
    )
    op.create_index("idx_orchestrator_run_git_tag", "orchestrator_run", ["git_tag"])
    op.create_index("idx_orchestrator_run_git_hash", "orchestrator_run", ["git_hash"])

    op.create_index("idx_class_map_model", "class_map", ["model"])
    op.create_index("idx_class_map_inference_idx", "class_map", ["inference_idx"])
    op.create_index("idx_class_map_class", "class_map", ["class"])

    op.create_index("idx_source_name", "source", ["st_name", "type"])

    op.create_index("idx_filter_hash", "filter", ["hash"])

    op.create_index("idx_slick_hitl", "slick", ["hitl_class"])
    op.create_index("idx_slick_confidence", "slick", ["machine_confidence"])
    op.create_index("idx_slick_polsby_popper", "slick", ["polsby_popper"])
    op.create_index("idx_slick_fill_factor", "slick", ["fill_factor"])


def downgrade() -> None:
    """drop indices"""
    op.drop_index("idx_model_name", "model")
    op.drop_index("idx_model_file_path", "model")

    op.drop_index("idx_sentinel1_grd_scene_id", "sentinel1_grd")

    op.drop_index("idx_vessel_density_name", "vessel_density")

    op.drop_index("idx_infra_distance_name", "infra_distance")

    op.drop_index("idx_trigger_trigger_time", "trigger")

    op.drop_index("idx_orchestrator_run_time", "orchestrator_run")
    op.drop_index("idx_orchestrator_run_git_tag", "orchestrator_run")
    op.drop_index("idx_orchestrator_run_git_hash", "orchestrator_run")

    op.drop_index("idx_class_map_model", "class_map")
    op.drop_index("idx_class_map_inference_idx", "class_map")
    op.drop_index("idx_class_map_class", "class_map")

    op.drop_index("idx_source_name", "source")

    op.drop_index("idx_filter_hash", "filter")

    op.drop_index("idx_slick_hitl", "slick")
    op.drop_index("idx_slick_confidence", "slick")
    op.drop_index("idx_slick_polsby_popper", "slick")
    op.drop_index("idx_slick_fill_factor", "slick")
