"""Prepare DB for Cv3

Revision ID: 96cd87e159a9
Revises: c7c033c1cdb5
Create Date: 2024-08-21 14:38:03.735682

"""

import sqlalchemy as sa
from sqlalchemy import orm

import cerulean_cloud.database_schema as database_schema
from alembic import op

# revision identifiers, used by Alembic.
revision = "96cd87e159a9"
down_revision = "c7c033c1cdb5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Performs the database schema upgrade needed for the Cv3 release.

    This function handles multiple schema changes including:
    - Dropping old foreign key constraints.
    - Renaming the 'user' table to 'users' and altering its structure by adding new columns.
    - Modifying the 'slick_to_source' table by removing an existing column and adding new ones related to 'hitl' verification.
    - Inserting a new entry in the 'cls' table.
    - Updating threshold settings in the 'model' table.
    - Replacing the 'magic_link' table with 'verification_token' and adjusting its structure.
    - Creating new 'accounts' and 'sessions' tables.
    - Adding a new 'hitl_slick' table for handling specific 'hitl' related interactions.

    Each step involves careful handling of foreign keys and data integrity.
    """

    # Drop the old foreign key constraints
    op.drop_constraint("subscription_user_fkey", "subscription", type_="foreignkey")
    op.drop_constraint("aoi_user_user_fkey", "aoi_user", type_="foreignkey")
    op.drop_constraint("magic_link_user_fkey", "magic_link", type_="foreignkey")

    # Alter user table and create new structure as 'users'
    op.rename_table("user", "users")
    op.add_column("users", sa.Column("name", sa.Text))
    op.add_column("users", sa.Column("emailVerified", sa.DateTime))
    op.add_column("users", sa.Column("image", sa.Text))
    op.add_column("users", sa.Column("role", sa.Text))
    op.alter_column("users", "email", nullable=True)
    op.drop_column("users", "create_time")

    op.create_foreign_key(None, "subscription", "users", ["user"], ["id"])
    op.create_foreign_key(None, "aoi_user", "users", ["user"], ["id"])
    op.create_foreign_key(None, "magic_link", "users", ["user"], ["id"])

    # Remove hitl_confirmed column
    op.drop_column("slick_to_source", "hitl_confirmed")

    # Add hitl columns to slick_to_source
    op.add_column("slick_to_source", sa.Column("hitl_verification", sa.Boolean))
    op.add_column("slick_to_source", sa.Column("hitl_confidence", sa.Float))
    op.add_column(
        "slick_to_source",
        sa.Column("hitl_user", sa.BigInteger, sa.ForeignKey("users.id")),
    )
    op.add_column("slick_to_source", sa.Column("hitl_time", sa.DateTime))
    op.add_column("slick_to_source", sa.Column("hitl_notes", sa.Text))

    # Insert new cls entry for AMBIGUOUS
    op.execute(
        "INSERT INTO cls (short_name, long_name) VALUES ('AMBIGUOUS', 'Ambiguous')"
    )

    # Modify specific model thresholds
    op.execute(
        """
    UPDATE model SET thresholds = '{"poly_nms_thresh": 0.2, "pixel_nms_thresh": 0.4, "bbox_score_thresh": 0.3, "poly_score_thresh": 0.1, "pixel_score_thresh": 0.5, "groundtruth_dice_thresh": 0.0}' WHERE id = 1
    """
    )

    # Replace 'magic_link' with 'verification_token'
    op.rename_table("magic_link", "verification_token")
    op.add_column(
        "verification_token", sa.Column("identifier", sa.Text, nullable=False)
    )
    op.add_column(
        "verification_token", sa.Column("expires", sa.DateTime, nullable=False)
    )
    op.drop_constraint("magic_link_pkey", "verification_token", type_="primary")
    op.create_primary_key(
        "verification_token_pk", "verification_token", ["identifier", "token"]
    )
    op.drop_column("verification_token", "id")
    op.drop_column("verification_token", "user")
    op.drop_column("verification_token", "expiration_time")
    op.drop_column("verification_token", "is_used")
    op.drop_column("verification_token", "create_time")
    op.drop_column("verification_token", "update_time")

    # Create accounts and sessions tables
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

    # Add hitl_slick table
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
    )

    bind = op.get_bind()
    session = orm.Session(bind=bind)
    with session.begin():
        models = [
            database_schema.Model(
                type="FASTAIUNET",
                file_path="experiments/2024_08_18_06_27_25_4cls_resnet34_pr512_px1024_500epochs_unet/tracing_cpu_model.pt",
                layers=["VV"],
                cls_map={
                    0: "BACKGROUND",
                    1: "INFRA",
                    2: "NATURAL",
                    3: "VESSEL",
                },  # inference_idx maps to class table
                name="ResNet34 41%",
                tile_width_m=40844,  # Used to calculate zoom
                tile_width_px=512,  # Used to calculate scale
                epochs=500,
                thresholds={
                    "poly_nms_thresh": 0.2,
                    "pixel_nms_thresh": 0.0,  # NOT USED IN UNETS
                    "bbox_score_thresh": 0.01,
                    "poly_score_thresh": 0.3,
                    "pixel_score_thresh": 0.8,
                    "groundtruth_dice_thresh": 0.0,
                },
                backbone_size=34,
                pixel_f1=0.536,
                # instance_f1=0.0, # TODO CALCULATE
            ),
        ]
        session.add_all(models)

        layers = [
            database_schema.Layer(
                short_name="ALL_255",  # TODO Rename to something liketo avoid conflict with PIXEL CLASS
                long_name="All Pixels Value=255",
                citation="",
                source_url="",
                notes="Can be used for ablation or to replace unwanted layers.",
            ),
            database_schema.Layer(
                short_name="ALL_ZEROS",  # TODO Rename to something liketo avoid conflict with PIXEL CLASS
                long_name="All Pixels Value=0",
                citation="",
                source_url="",
                notes="Can be used for ablation or to replace unwanted layers.",
            ),
        ]
        session.add_all(layers)


def downgrade() -> None:
    """
    Reverts the database schema changes made by the upgrade function for the Cv3 release.

    This function systematically undoes the changes including:
    - Dropping newly created tables such as 'sessions', 'accounts', and 'hitl_slick'.
    - Restoring the 'verification_token' table to 'magic_link', including the reconstruction of dropped columns.
    - Removing the new entries and columns added in the upgrade to the 'slick_to_source' and 'cls' tables.
    - Reverting changes to the 'model' table's thresholds.
    - Restoring the 'users' table back to 'user' and undoing all structural modifications such as dropped and renamed columns.
    - Recreating the original foreign key constraints that were dropped during the upgrade.

    Each operation is designed to maintain database integrity and ensure that data loss is minimized during the rollback.
    """

    # Restore tables and drop newly created tables
    op.drop_table("sessions")
    op.drop_table("accounts")
    op.drop_table("hitl_slick")

    # Restore the 'verification_token' to 'magic_link'
    op.rename_table("verification_token", "magic_link")
    op.drop_column("magic_link", "identifier")
    op.drop_column("magic_link", "expires")
    op.add_column("magic_link", sa.Column("id", sa.BigInteger, primary_key=True))
    op.add_column(
        "magic_link",
        sa.Column("user", sa.BigInteger, sa.ForeignKey("users.id"), nullable=False),
    )
    op.add_column(
        "magic_link", sa.Column("expiration_time", sa.DateTime, nullable=False)
    )
    op.add_column("magic_link", sa.Column("is_used", sa.Boolean, nullable=False))
    op.add_column("magic_link", sa.Column("create_time", sa.DateTime, nullable=False))
    op.add_column("magic_link", sa.Column("update_time", sa.DateTime, nullable=False))
    op.drop_constraint("verification_token_pk", "magic_link", type_="primary")

    # Revert the model table updates
    op.execute(
        """
        UPDATE model SET thresholds = NULL WHERE id = 1
        """
    )

    # Delete cls entry for AMBIGUOUS
    op.execute("DELETE FROM cls WHERE short_name = 'AMBIGUOUS'")

    # Drop columns added to 'slick_to_source'
    op.drop_column("slick_to_source", "hitl_verification")
    op.drop_column("slick_to_source", "hitl_confidence")
    op.drop_column("slick_to_source", "hitl_user")
    op.drop_column("slick_to_source", "hitl_time")
    op.drop_column("slick_to_source", "hitl_notes")

    # Recreate hitl_confirmed column in 'slick_to_source'
    op.add_column("slick_to_source", sa.Column("hitl_confirmed", sa.Boolean))

    # Revert changes in 'users' table
    op.add_column("users", sa.Column("create_time", sa.DateTime))
    op.drop_column("users", "name")
    op.drop_column("users", "emailVerified")
    op.drop_column("users", "image")
    op.drop_column("users", "role")
    op.alter_column("users", "email", nullable=False)
    op.rename_table("users", "user")

    # Recreate foreign key constraints
    op.create_foreign_key(
        "fk_subscription_user_id", "subscription", "user", ["user_id"], ["id"]
    )
    op.create_foreign_key(
        "fk_aoi_user_user_id", "aoi_user", "user", ["user_id"], ["id"]
    )
