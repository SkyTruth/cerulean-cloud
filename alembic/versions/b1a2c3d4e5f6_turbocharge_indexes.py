"""Turbo-charge indexes, constraints, and JSONB conversion.

Revision ID: b1a2c3d4e5f6
Revises: 7cd715196b8d
Create Date: 2025-05-31 12:34:56
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "b1a2c3d4e5f6"
down_revision = "7cd715196b8d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- 1. DROP redundant / unused B-tree indexes -------------------------
    for idx in (
        "idx_sentinel1_grd_scene_id",
        "idx_slick_to_aoi_slick",
        "idx_source_to_tag_source",
        "idx_model_name",
        "idx_slick_hitl",
        "idx_slick_confidence",
        "idx_slick_length",
        "idx_slick_polsby_popper",
        "idx_slick_fill_factor",
        "idx_slick_orchestrator_run",
        "idx_slick_cls",
        "idx_slick_to_source_collated_score",
    ):
        op.drop_index(idx, if_exists=True)

    # --- 2. Convert JSON ➜ JSONB -------------------------------------------
    op.execute("ALTER TABLE filter ALTER COLUMN json TYPE JSONB USING json::jsonb")

    # --- 3. Add missing UNIQUE constraints ---------------------------------
    op.create_unique_constraint("users_email_key", "users", ["email"])
    op.create_unique_constraint(
        "accounts_provider_provideraccountid_key",
        "accounts",
        ["provider", "providerAccountId"],
    )

    # --- 4. Spatial, BRIN, composite & partial indexes ----------------------
    op.create_index(
        "idx_slick_geometry_gist",
        "slick",
        ["geometry"],
        postgresql_using="gist",
        postgresql_ops={"geometry": "gist_geography_ops"},
        if_not_exists=True,
    )
    op.create_index(
        "idx_slick_timestamp_brin",
        "slick",
        ["slick_timestamp"],
        postgresql_using="brin",
        if_not_exists=True,
    )
    op.create_index(
        "idx_orchestrator_run_start_brin",
        "orchestrator_run",
        ["inference_start_time"],
        postgresql_using="brin",
        if_not_exists=True,
    )
    op.create_index(
        "idx_slick_orun_cls",
        "slick",
        ["orchestrator_run", "cls"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_slick_to_source_slick_score",
        "slick_to_source",
        ["slick", sa.text("collated_score DESC")],
        if_not_exists=True,
    )
    op.create_index(
        "idx_source_type_name",
        "source",
        ["type", "st_name"],
        if_not_exists=True,
    )
    op.create_index(
        "idx_slick_active_recent",
        "slick",
        ["orchestrator_run", sa.text("slick_timestamp DESC")],
        postgresql_where=sa.text("active IS TRUE"),
        if_not_exists=True,
    )
    op.create_index(
        "idx_sts_active_highconf",
        "slick_to_source",
        ["source", sa.text("collated_score DESC")],
        postgresql_where=sa.text("active IS TRUE AND collated_score > 0.8"),
        if_not_exists=True,
    )
    op.create_index(
        "idx_filter_json_gin",
        "filter",
        ["json"],
        postgresql_using="gin",
        postgresql_ops={"json": "jsonb_path_ops"},
        if_not_exists=True,
    )


def downgrade() -> None:
    # --- drop new objects ---------------------------------------------------
    for idx in (
        "idx_slick_geometry_gist",
        "idx_slick_timestamp_brin",
        "idx_orchestrator_run_start_brin",
        "idx_slick_orun_cls",
        "idx_slick_to_source_slick_score",
        "idx_source_type_name",
        "idx_slick_active_recent",
        "idx_sts_active_highconf",
        "idx_filter_json_gin",
    ):
        op.drop_index(idx, if_exists=True)

    op.drop_constraint("users_email_key", "users", type_="unique")
    op.drop_constraint(
        "accounts_provider_provideraccountid_key",
        "accounts",
        type_="unique",
    )

    op.execute("ALTER TABLE filter ALTER COLUMN json TYPE JSON USING json::json")

    # --- restore original indexes ------------------------------------------
    op.create_index(
        "idx_sentinel1_grd_scene_id", "sentinel1_grd", ["scene_id"], unique=True
    )
    op.create_index("idx_slick_to_aoi_slick", "slick_to_aoi", ["slick"])
    op.create_index("idx_source_to_tag_source", "source_to_tag", ["source"])
    op.create_index("idx_model_name", "model", ["name"])
    op.create_index("idx_slick_hitl", "slick", ["hitl_cls"])
    op.create_index("idx_slick_confidence", "slick", ["machine_confidence"])
    op.create_index("idx_slick_length", "slick", ["length"])
    op.create_index("idx_slick_polsby_popper", "slick", ["polsby_popper"])
    op.create_index("idx_slick_fill_factor", "slick", ["fill_factor"])
    op.create_index("idx_slick_orchestrator_run", "slick", ["orchestrator_run"])
    op.create_index("idx_slick_cls", "slick", ["cls"])
    op.create_index(
        "idx_slick_to_source_collated_score", "slick_to_source", ["collated_score"]
    )
