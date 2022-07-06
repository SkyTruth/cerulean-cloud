"""Add tables

Revision ID: 3c4693517ef6
Revises: 54c42e9e879f
Create Date: 2022-06-30 11:45:00.359562

"""
import sqlalchemy as sa
from geoalchemy2 import Geography
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
    op.create_table(
        "model",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("thresholds", sa.Integer),
        sa.Column("fine_pkl_idx", sa.Integer),
        sa.Column("chip_size_orig", sa.Integer),
        sa.Column("chip_size_reduced", sa.Integer),
        sa.Column("overhang", sa.Boolean),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column(
            "updated_time", sa.DateTime, nullable=False, server_default=sa.func.now()
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
        "vessel_density",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("start_time", sa.DateTime, nullable=False),
        sa.Column("end_time", sa.DateTime, nullable=False),
        sa.Column("meta", JSONB),
        sa.Column("geometry", Geography("POLYGON"), nullable=False),
    )
    op.create_table(
        "infra_distance",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("start_time", sa.DateTime, nullable=False),
        sa.Column("end_time", sa.DateTime, nullable=False),
        sa.Column("meta", JSONB),
        sa.Column("geometry", Geography("POLYGON"), nullable=False),
        sa.Column("url", sa.Text, nullable=False),
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
        sa.Column("vessel_density", sa.Integer, sa.ForeignKey("vessel_density.id")),
        sa.Column("infra_distance", sa.Integer, sa.ForeignKey("infra_distance.id")),
    )

    op.create_table(
        "eez",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("mrgid", sa.Integer),
        sa.Column("geoname", sa.Text),
        sa.Column("sovereigns", ARRAY(sa.Text)),
        sa.Column("geometry", Geography("MULTIPOLYGON"), nullable=False),
    )

    op.create_table(
        "slick_class",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("value", sa.Integer),
        sa.Column("name", sa.String(200)),
        sa.Column("notes", sa.Text),
        sa.Column("slick_class", ARRAY(sa.Integer)),
        sa.Column(
            "create_time", sa.DateTime, nullable=False, server_default=sa.func.now()
        ),
        sa.Column("active", sa.Boolean, nullable=False),
    )

    op.create_table(
        "slick_source",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("name", sa.String(200)),
        sa.Column("notes", sa.Text),
        sa.Column("slick_source", ARRAY(sa.BigInteger)),
        sa.Column(
            "create_time", sa.DateTime, nullable=False, server_default=sa.func.now()
        ),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.Column("geometry", Geography("geometry")),
    )

    op.create_table(
        "slick",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("slick_timestamp", sa.DateTime, nullable=False),
        sa.Column("geometry", Geography("MULTIPOLYGON"), nullable=False),
        sa.Column("machine_confidence", sa.Float),
        sa.Column("human_confidence", sa.Float),
        sa.Column("area", sa.Float, sa.Computed("ST_Area(geometry)")),
        sa.Column("perimeter", sa.Float, sa.Computed("ST_Perimeter(geometry)")),
        sa.Column("centroid", Geography("POINT"), sa.Computed("ST_Centroid(geometry)")),
        sa.Column(
            "polsby_popper",
            sa.Float,
            sa.Computed(
                "(ST_Perimeter(geometry) * ST_Perimeter(geometry)) / ST_Area(geometry)"
            ),
        ),
        sa.Column(
            "fill_factor",
            sa.Float,
            sa.Computed(
                "ST_Area(geometry) / ST_Area(ST_OrientedEnvelope(geometry::geometry)::geography)"
            ),
        ),
        sa.Column(
            "create_time", sa.DateTime, nullable=False, server_default=sa.func.now()
        ),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.Column("validated", sa.Boolean, nullable=False),
        sa.Column("slick", ARRAY(sa.BigInteger)),
        sa.Column("notes", sa.Text),
        sa.Column("meta", JSONB),
        sa.Column(
            "orchestrator_run",
            sa.BigInteger,
            sa.ForeignKey("orchestrator_run.id"),
            nullable=False,
        ),
        sa.Column(
            "slick_class",
            sa.BigInteger,
            sa.ForeignKey("slick_class.id"),
            nullable=False,
        ),
    )

    op.create_table(
        "slick_to_slick_source",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("slick", sa.BigInteger, sa.ForeignKey("slick.id"), nullable=False),
        sa.Column(
            "slick_source",
            sa.BigInteger,
            sa.ForeignKey("slick_source.id"),
            nullable=False,
        ),
        sa.Column("human_confidence", sa.Float),
        sa.Column("machine_confidence", sa.Float),
    )

    op.create_table(
        "slick_to_eez",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("slick", sa.BigInteger, sa.ForeignKey("slick.id"), nullable=False),
        sa.Column("eez", sa.BigInteger, sa.ForeignKey("eez.id"), nullable=False),
    )


def downgrade() -> None:
    """drop tables"""
    op.drop_table("slick_to_slick_source")
    op.drop_table("slick_to_eez")
    op.drop_table("slick")
    op.drop_table("slick_class")
    op.drop_table("orchestrator_run")
    op.drop_table("eez")
    op.drop_table("slick_source")
    op.drop_table("trigger")
    op.drop_table("model")
    op.drop_table("sentinel1_grd")
    op.drop_table("vessel_density")
    op.drop_table("infra_distance")
