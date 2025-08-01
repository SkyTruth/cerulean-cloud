"""Updates to Users, Accounts and Sessions tables

Revision ID: 0b8874f9ce2e
Revises: c7c033c1cdb5
Create Date: 2025-07-09 16:04:36.756440

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "0b8874f9ce2e"
down_revision = "c7c033c1cdb5"
branch_labels = None
depends_on = None


def upgrade():
    # ----- USERS -----

    op.alter_column(
        "users",
        "emailVerified",
        existing_type=postgresql.TIMESTAMP(),
        type_=sa.Boolean(),
        postgresql_using='CASE WHEN "emailVerified" IS NULL THEN FALSE ELSE TRUE END',
    )
    op.alter_column(
        "users",
        "email",
        nullable=False,
    )
    op.add_column("users", sa.Column("firstName", sa.String()))
    op.add_column("users", sa.Column("lastName", sa.String()))
    op.execute("""
        UPDATE users SET "firstName" = "name"
               """)
    op.execute(
        """
        ALTER TABLE users
        ALTER COLUMN "name" DROP DEFAULT,
        ALTER COLUMN "name" SET GENERATED ALWAYS AS (
            NULLIF(concat_ws(' ', "firstName", "lastName"), '')
        ) STORED
        ALTER COLUMN "name" SET NOT NULL
        """
    )

    op.add_column("users", sa.Column("organization", sa.String()))
    op.add_column("users", sa.Column("organizationType", postgresql.ARRAY(sa.String)))
    op.add_column("users", sa.Column("location", sa.String()))
    op.add_column("users", sa.Column("emailConsent", sa.Boolean(), default=False))
    op.add_column("users", sa.Column("banned", sa.Boolean(), default=False))
    op.add_column("users", sa.Column("banReason", sa.String()))
    op.add_column("users", sa.Column("banExpires", sa.Date()))

    # ----- SESSIONS -----

    op.alter_column("sessions", "expires", new_column_name="expiresAt")
    op.alter_column("sessions", "sessionToken", new_column_name="token")
    op.add_column(
        "sessions",
        sa.Column("createdAt", sa.DateTime(), server_default=sa.text("now()")),
    )
    op.add_column(
        "sessions",
        sa.Column("updatedAt", sa.DateTime(), server_default=sa.text("now()")),
    )
    op.add_column("sessions", sa.Column("impersonatedBy", sa.String()))

    # ----- ACCOUNTS -----

    op.alter_column("accounts", "provider", new_column_name="providerId")
    op.alter_column("accounts", "providerAccountId", new_column_name="accountId")
    op.alter_column("accounts", "refresh_token", new_column_name="refreshToken")
    op.alter_column("accounts", "access_token", new_column_name="accessToken")
    op.alter_column("accounts", "id_token", new_column_name="idToken")

    # Convert expires_at (int) to accessTokenExpiresAt (timestamp)
    op.alter_column(
        "accounts",
        "expires_at",
        new_column_name="accessTokenExpiresAt",
        type_=sa.DateTime(),
        postgresql_using='to_timestamp("expires_at")',
    )

    op.add_column(
        "accounts",
        sa.Column("createdAt", sa.DateTime(), server_default=sa.text("now()")),
    )
    op.add_column(
        "accounts",
        sa.Column("updatedAt", sa.DateTime(), server_default=sa.text("now()")),
    )


def downgrade():
    # ----- ACCOUNTS -----
    # Remove audit columns first to avoid dependencies
    op.drop_column("accounts", "updatedAt")
    op.drop_column("accounts", "createdAt")

    # Revert accessTokenExpiresAt back to integer expires_at
    op.alter_column(
        "accounts",
        "accessTokenExpiresAt",
        new_column_name="expires_at",
        type_=sa.Integer(),
        postgresql_using='extract(epoch from "accessTokenExpiresAt")::int',
    )

    # Revert column names
    op.alter_column("accounts", "idToken", new_column_name="id_token")
    op.alter_column("accounts", "accessToken", new_column_name="access_token")
    op.alter_column("accounts", "refreshToken", new_column_name="refresh_token")
    op.alter_column("accounts", "accountId", new_column_name="providerAccountId")
    op.alter_column("accounts", "providerId", new_column_name="provider")

    # ----- SESSIONS -----
    op.drop_column("sessions", "impersonatedBy")
    op.drop_column("sessions", "updatedAt")
    op.drop_column("sessions", "createdAt")

    op.alter_column("sessions", "token", new_column_name="sessionToken")
    op.alter_column("sessions", "expiresAt", new_column_name="expires")

    # ----- USERS -----
    op.drop_column("users", "banExpires")
    op.drop_column("users", "banReason")
    op.drop_column("users", "banned")
    op.drop_column("users", "emailConsent")
    op.drop_column("users", "location")
    op.drop_column("users", "organizationType")
    op.drop_column("users", "organization")
    op.drop_column("users", "lastName")

    # Convert emailVerified boolean back to TIMESTAMP
    op.alter_column(
        "users",
        "emailVerified",
        existing_type=sa.Boolean(),
        type_=postgresql.TIMESTAMP(),
        postgresql_using='CASE WHEN "emailVerified" THEN now() ELSE NULL END',
    )

    # Revert renamed and constraint changes
    op.alter_column("users", "firstName", new_column_name="name", nullable=True)
    op.alter_column("users", "email", nullable=True)
