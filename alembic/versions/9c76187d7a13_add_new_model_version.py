"""Add new model version

Revision ID: 9c76187d7a13
Revises: 0a4536575154
Create Date: 2022-07-13 10:59:24.267373

"""
from sqlalchemy import orm

import cerulean_cloud.database_schema as database_schema
from alembic import op

# revision identifiers, used by Alembic.
revision = "9c76187d7a13"
down_revision = "0a4536575154"
branch_labels = None
depends_on = None

MODEL_PATH = "experiments/cv2/29_Jun_2022_06_36_38_fastai_unet/tracing_cpu_224_120__512_36__4_34_0.0003_0.436.pt"


def upgrade() -> None:
    """add new model"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with session.begin():
        model = database_schema.Model(
            name=MODEL_PATH,
            file_path=MODEL_PATH,
        )
        session.add(model)


def downgrade() -> None:
    """remove new model"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with session.begin():
        model = (
            session.query(database_schema.Model)
            .filter_by(file_path=MODEL_PATH)
            .one_or_none()
        )
        session.delete(model)
