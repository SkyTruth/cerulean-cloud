"""First migration

Revision ID: 7ff505f31b77
Revises:
Create Date: 2022-06-24 12:52:00.165406

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "7ff505f31b77"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """upgrade"""
    with open("alembic/initial_migration.sql") as file_:
        for statement in file_.read().split(";\n"):
            if not statement.startswith("\n--"):
                print(statement)
                op.execute(statement)


def downgrade() -> None:
    """downgrade"""
    pass
