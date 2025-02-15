"""Add tables

Revision ID: 3c4693517ef6
Revises: 54c42e9e879f
Create Date: 2022-06-30 11:45:00.359562

"""

import sqlalchemy as sa
from geoalchemy2 import Geography, Geometry
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import ARRAY

from alembic import op

# revision identifiers, used by Alembic.
revision = "3c4693517ef6"
down_revision = "54c42e9e879f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """add tables"""
    # EditTheDatabase

    op.create_table(
        "layer",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("short_name", sa.Text, nullable=False, unique=True),
        sa.Column("long_name", sa.Text),
        sa.Column("citation", sa.Text),
        sa.Column("source_url", sa.Text),
        sa.Column("notes", sa.Text),
        sa.Column("start_time", sa.DateTime),
        sa.Column("end_time", sa.DateTime),
        sa.Column("json", sa.JSON),
        sa.Column("update_time", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "model",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("layers", ARRAY(sa.Text), nullable=False),
        sa.Column("cls_map", sa.JSON, nullable=False),
        sa.Column("name", sa.Text),
        sa.Column("tile_width_m", sa.Integer, nullable=False),
        sa.Column("tile_width_px", sa.Integer, nullable=False),
        sa.Column(
            "zoom_level",
            sa.Integer,
            sa.Computed("ROUND(LOG(2, 40075000.0 / tile_width_m)) - 1"),
            # 40075000 = Earth Circumference in meters
            # '- 1' comes from using WorldCRS84Quad which has the zoom level "off-by-one" compared to WebMercatorQuad
        ),
        sa.Column("scale", sa.Integer, sa.Computed("ROUND(tile_width_px / 256.0)")),
        sa.Column("epochs", sa.Integer),
        sa.Column("thresholds", sa.JSON, nullable=False),
        sa.Column("backbone_size", sa.Integer),
        sa.Column("pixel_f1", sa.Float),
        sa.Column("instance_f1", sa.Float),
        sa.Column(
            "update_time", sa.DateTime, nullable=False, server_default=sa.func.now()
        ),
    )

    # Layers
    op.create_table(
        "sentinel1_grd",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("scene_id", sa.String(200), nullable=False, unique=True),
        sa.Column("absolute_orbit_number", sa.Integer),
        sa.Column("mode", sa.String(200)),
        sa.Column("polarization", sa.String(200)),
        sa.Column("scihub_ingestion_time", sa.DateTime),
        sa.Column("start_time", sa.DateTime, nullable=False),
        sa.Column("end_time", sa.DateTime, nullable=False),
        sa.Column("meta", JSONB),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("geometry", Geography("POLYGON"), nullable=False),
    )

    op.create_table(
        "trigger",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "trigger_time", sa.DateTime, nullable=False, server_default=sa.func.now()
        ),
        sa.Column("scene_count", sa.Integer),
        sa.Column("filtered_scene_count", sa.Integer),
        sa.Column("trigger_logs", sa.Text, nullable=False),
        sa.Column("trigger_type", sa.String(200), nullable=False),
    )

    op.create_table(
        "orchestrator_run",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("inference_start_time", sa.DateTime, nullable=False),
        sa.Column("inference_end_time", sa.DateTime, nullable=False),
        sa.Column("base_tiles", sa.Integer),
        sa.Column("offset_tiles", sa.Integer),
        sa.Column("git_hash", sa.Text),
        sa.Column("git_tag", sa.String(200)),
        sa.Column("zoom", sa.Integer),
        sa.Column("scale", sa.Integer),
        sa.Column("success", sa.Boolean),
        sa.Column("inference_run_logs", sa.Text, nullable=False),
        sa.Column("geometry", Geography("POLYGON"), nullable=False),
        sa.Column(
            "trigger", sa.BigInteger, sa.ForeignKey("trigger.id"), nullable=False
        ),
        sa.Column("model", sa.Integer, sa.ForeignKey("model.id"), nullable=False),
        sa.Column("sentinel1_grd", sa.BigInteger, sa.ForeignKey("sentinel1_grd.id")),
    )

    op.create_table(
        "cls",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("short_name", sa.Text, unique=True),
        sa.Column("long_name", sa.Text),
        sa.Column("supercls", sa.BigInteger, sa.ForeignKey("cls.id")),
    )

    op.create_table(
        "slick",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("slick_timestamp", sa.DateTime, nullable=False),
        sa.Column("geometry", Geography("MULTIPOLYGON"), nullable=False),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.Column(
            "orchestrator_run",
            sa.BigInteger,
            sa.ForeignKey("orchestrator_run.id"),
            nullable=False,
        ),
        sa.Column(
            "create_time", sa.DateTime, nullable=False, server_default=sa.func.now()
        ),
        sa.Column("inference_idx", sa.Integer, nullable=False),
        sa.Column("cls", sa.Integer),
        sa.Column(
            "hitl_cls",
            sa.BigInteger,
            sa.ForeignKey("cls.id"),
        ),
        sa.Column("machine_confidence", sa.Float),
        sa.Column("precursor_slicks", ARRAY(sa.BigInteger)),
        sa.Column("notes", sa.Text),
        sa.Column("splines", sa.JSON),
        sa.Column("aspect_ratio_factor", sa.Float),
        sa.Column("length", sa.Float),
        sa.Column("area", sa.Float),
        sa.Column("perimeter", sa.Float),
        sa.Column("centroid", Geography("POINT")),
        sa.Column("polsby_popper", sa.Float),
        sa.Column("fill_factor", sa.Float),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("name", sa.Text),
        sa.Column("email", sa.Text),
        sa.Column("emailVerified", sa.DateTime),
        sa.Column("image", sa.Text),
        sa.Column("role", sa.Text),
    )

    op.create_table(
        "filter",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("json", sa.JSON, nullable=False),
        sa.Column("hash", sa.Text),
    )

    op.create_table(
        "frequency",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("short_name", sa.Text, nullable=False, unique=True),
        sa.Column("long_name", sa.Text),
    )

    op.create_table(
        "subscription",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("user", sa.BigInteger, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("filter", sa.BigInteger, sa.ForeignKey("filter.id"), nullable=False),
        sa.Column(
            "frequency", sa.Integer, sa.ForeignKey("frequency.id"), nullable=False
        ),
        sa.Column("active", sa.Boolean),
        sa.Column("create_time", sa.DateTime, server_default=sa.func.now()),
        sa.Column("update_time", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "verification_token",
        sa.Column("identifier", sa.Text, nullable=False),
        sa.Column("expires", sa.DateTime, nullable=False),
        sa.Column("token", sa.Text, nullable=False),
        sa.PrimaryKeyConstraint("identifier", "token"),
    )

    op.create_table(
        "accounts",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("userId", sa.BigInteger, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("provider", sa.Text, nullable=False),
        sa.Column("providerAccountId", sa.Text, nullable=False),
        sa.Column("refresh_token", sa.Text),
        sa.Column("access_token", sa.Text),
        sa.Column("expires_at", sa.BigInteger),
        sa.Column("id_token", sa.Text),
        sa.Column("scope", sa.Text),
        sa.Column("session_state", sa.Text),
        sa.Column("token_type", sa.Text),
    )

    op.create_table(
        "sessions",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("userId", sa.BigInteger, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("expires", sa.DateTime, nullable=False),
        sa.Column("sessionToken", sa.Text, nullable=False),
    )

    op.create_table(
        "aoi_type",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("table_name", sa.Text, nullable=False),
        sa.Column("long_name", sa.Text),
        sa.Column("short_name", sa.Text),
        sa.Column("source_url", sa.Text),
        sa.Column("citation", sa.Text),
        sa.Column("update_time", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "aoi",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("type", sa.BigInteger, sa.ForeignKey("aoi_type.id"), nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("geometry", Geography("MULTIPOLYGON"), nullable=False),
    )

    op.create_table(
        "aoi_eez",
        sa.Column("aoi_id", sa.BigInteger, sa.ForeignKey("aoi.id"), primary_key=True),
        sa.Column("mrgid", sa.Integer),
        sa.Column("sovereigns", ARRAY(sa.Text)),
    )

    op.create_table(
        "aoi_iho",
        sa.Column("aoi_id", sa.BigInteger, sa.ForeignKey("aoi.id"), primary_key=True),
        sa.Column("mrgid", sa.Integer),
    )

    op.create_table(
        "aoi_mpa",
        sa.Column("aoi_id", sa.BigInteger, sa.ForeignKey("aoi.id"), primary_key=True),
        sa.Column("wdpaid", sa.Integer),
        sa.Column("desig", sa.Text),
        sa.Column("desig_type", sa.Text),
        sa.Column("status_yr", sa.Integer),
        sa.Column("mang_auth", sa.Text),
        sa.Column("parent_iso", sa.Text),
    )

    op.create_table(
        "aoi_user",
        sa.Column("aoi_id", sa.BigInteger, sa.ForeignKey("aoi.id"), primary_key=True),
        sa.Column("user", sa.BigInteger, sa.ForeignKey("users.id")),
        sa.Column("create_time", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "aoi_chunks",
        sa.Column(
            "id",
            sa.BigInteger,
            sa.ForeignKey(
                "aoi.id", ondelete="CASCADE", deferrable=True, initially="DEFERRED"
            ),
        ),
        sa.Column("geometry", Geometry("POLYGON", srid=4326), nullable=False),
    )

    op.create_table(
        "slick_to_aoi",
        sa.Column(
            "slick",
            sa.BigInteger,
            sa.ForeignKey(
                "slick.id", ondelete="CASCADE", deferrable=True, initially="DEFERRED"
            ),
            primary_key=True,
        ),
        sa.Column(
            "aoi",
            sa.BigInteger,
            sa.ForeignKey(
                "aoi.id", ondelete="CASCADE", deferrable=True, initially="DEFERRED"
            ),
            primary_key=True,
        ),
    )

    op.create_table(
        "source_type",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("table_name", sa.Text),
        sa.Column("long_name", sa.Text),
        sa.Column("short_name", sa.Text),
        sa.Column("citation", sa.Text),
        sa.Column("ext_id_name", sa.Text),
    )

    op.create_table(
        "source",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "type", sa.BigInteger, sa.ForeignKey("source_type.id"), nullable=False
        ),
        sa.Column("st_name", sa.Text, nullable=False),
        sa.Column("ext_id", sa.Text, nullable=False),
    )

    op.create_table(
        "source_vessel",
        sa.Column(
            "source_id", sa.BigInteger, sa.ForeignKey("source.id"), primary_key=True
        ),
        sa.Column("ext_name", sa.Text),
        sa.Column("ext_shiptype", sa.Text),
        sa.Column("flag", sa.Text),
    )

    op.create_table(
        "source_infra",
        sa.Column(
            "source_id", sa.BigInteger, sa.ForeignKey("source.id"), primary_key=True
        ),
        sa.Column("geometry", Geography("POINT"), nullable=False),
        sa.Column("ext_name", sa.Text),
        sa.Column("operator", sa.Text),
        sa.Column("sovereign", sa.Text),
        sa.Column("orig_yr", sa.DateTime),
        sa.Column("last_known_status", sa.Text),
    )

    op.create_table(
        "source_dark",
        sa.Column(
            "source_id", sa.BigInteger, sa.ForeignKey("source.id"), primary_key=True
        ),
        sa.Column("geometry", Geography("POINT"), nullable=False),
        sa.Column("scene_id", sa.Text),
        sa.Column("length_m", sa.Float),
        sa.Column("detection_probability", sa.Float),
    )

    op.create_table(
        "source_natural",
        sa.Column(
            "source_id", sa.BigInteger, sa.ForeignKey("source.id"), primary_key=True
        ),
        sa.Column("geometry", Geography("POINT"), nullable=False),
    )

    op.create_table(
        "slick_to_source",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("slick", sa.BigInteger, sa.ForeignKey("slick.id"), nullable=False),
        sa.Column("source", sa.BigInteger, sa.ForeignKey("source.id"), nullable=False),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.Column("git_hash", sa.Text),
        sa.Column("git_tag", sa.Text),
        sa.Column("coincidence_score", sa.Float),
        sa.Column("collated_score", sa.Float),
        sa.Column("rank", sa.BigInteger),
        sa.Column("geojson_fc", sa.JSON, nullable=False),
        sa.Column("geometry", Geography("GEOMETRY"), nullable=False),
        sa.Column(
            "create_time", sa.DateTime, nullable=False, server_default=sa.func.now()
        ),
        sa.Column("hitl_verification", sa.Boolean),
        sa.Column("hitl_confidence", sa.Float),
        sa.Column("hitl_user", sa.BigInteger, sa.ForeignKey("users.id")),
        sa.Column("hitl_time", sa.DateTime),
        sa.Column("hitl_notes", sa.Text),
    )

    op.create_table(
        "tag",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("short_name", sa.Text, nullable=False, unique=True),
        sa.Column("long_name", sa.Text, nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("citation", sa.Text),
    )

    op.create_table(
        "source_to_tag",
        sa.Column("source", sa.BigInteger, sa.ForeignKey("source.id"), nullable=False),
        sa.Column("tag", sa.BigInteger, sa.ForeignKey("tag.id"), nullable=False),
        sa.Column(
            "create_time", sa.DateTime, nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("source", "tag"),
    )

    op.create_table(
        "hitl_slick",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("slick", sa.BigInteger, sa.ForeignKey("slick.id"), nullable=False),
        sa.Column("user", sa.BigInteger, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("cls", sa.BigInteger, sa.ForeignKey("cls.id"), nullable=False),
        sa.Column("confidence", sa.Float),
        sa.Column(
            "update_time", sa.DateTime, nullable=False, server_default=sa.func.now()
        ),
        sa.Column("is_duplicate", sa.Boolean),
    )


def downgrade() -> None:
    """drop tables"""
    op.drop_table("hitl_slick")
    op.drop_table("source_to_tag")
    op.drop_table("tag")
    op.drop_table("slick_to_source")
    op.drop_table("source_dark")
    op.drop_table("source_infra")
    op.drop_table("source_vessel")
    op.drop_table("source")
    op.drop_table("source_type")
    op.drop_table("slick_to_aoi")
    op.drop_table("aoi_chunks")
    op.drop_table("aoi_user")
    op.drop_table("aoi_mpa")
    op.drop_table("aoi_iho")
    op.drop_table("aoi_eez")
    op.drop_table("aoi")
    op.drop_table("aoi_type")
    op.drop_table("verification_token")
    op.drop_table("sessions")
    op.drop_table("accounts")
    op.drop_table("subscription")
    op.drop_table("frequency")
    op.drop_table("filter")
    op.drop_table("users")
    op.drop_table("slick")
    op.drop_table("cls")
    op.drop_table("orchestrator_run")
    op.drop_table("trigger")
    op.drop_table("sentinel1_grd")
    op.drop_table("model")
    op.drop_table("layer")

    # EditTheDatabase
