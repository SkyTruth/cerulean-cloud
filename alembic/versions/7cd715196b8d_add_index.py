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

    op.create_index("idx_layer_short_name", "layer", ["short_name"])

    op.create_index(
        "idx_sentinel1_grd_scene_id", "sentinel1_grd", ["scene_id"], unique=True
    )

    op.create_index("idx_trigger_trigger_time", "trigger", ["trigger_time"])

    op.create_index(
        "idx_orchestrator_run_time",
        "orchestrator_run",
        ["inference_start_time", "inference_end_time"],
    )
    op.create_index("idx_orchestrator_run_git_tag", "orchestrator_run", ["git_tag"])
    op.create_index("idx_orchestrator_run_git_hash", "orchestrator_run", ["git_hash"])
    op.create_index(
        "idx_orchestrator_run_sentinel1_grd", "orchestrator_run", ["sentinel1_grd"]
    )

    op.create_index("idx_source_name", "source", ["st_name", "type"])

    op.create_index("idx_slick_to_aoi_slick", "slick_to_aoi", ["slick"])
    op.create_index("idx_slick_to_aoi_aoi", "slick_to_aoi", ["aoi"])

    op.create_index("idx_filter_hash", "filter", ["hash"])

    op.create_index("idx_slick_hitl", "slick", ["hitl_cls"])
    op.create_index("idx_slick_confidence", "slick", ["machine_confidence"])
    op.create_index("idx_slick_length", "slick", ["length"])
    op.create_index("idx_slick_polsby_popper", "slick", ["polsby_popper"])
    op.create_index("idx_slick_fill_factor", "slick", ["fill_factor"])
    op.create_index("idx_slick_orchestrator_run", "slick", ["orchestrator_run"])
    op.create_index("idx_slick_cls", "slick", ["cls"])


def downgrade() -> None:
    """drop indices"""
    op.drop_index("idx_model_name", "model")
    op.drop_index("idx_model_file_path", "model")

    op.drop_index("idx_layer_short_name", "layer")

    op.drop_index("idx_sentinel1_grd_scene_id", "sentinel1_grd")

    op.drop_index("idx_trigger_trigger_time", "trigger")

    op.drop_index("idx_orchestrator_run_time", "orchestrator_run")
    op.drop_index("idx_orchestrator_run_git_tag", "orchestrator_run")
    op.drop_index("idx_orchestrator_run_git_hash", "orchestrator_run")
    op.drop_index("idx_orchestrator_run_sentinel1_grd", "orchestrator_run")

    op.drop_index("idx_slick_to_aoi_aoi", "slick_to_aoi")
    op.drop_index("idx_slick_to_aoi_slick", "slick_to_aoi")

    op.drop_index("idx_source_name", "source")

    op.drop_index("idx_filter_hash", "filter")

    op.drop_index("idx_slick_hitl", "slick")
    op.drop_index("idx_slick_confidence", "slick")
    op.drop_index("idx_slick_length", "slick")
    op.drop_index("idx_slick_polsby_popper", "slick")
    op.drop_index("idx_slick_fill_factor", "slick")
    op.drop_index("idx_slick_orchestrator_run", "slick")
    op.drop_index("idx_slick_cls", "slick")
