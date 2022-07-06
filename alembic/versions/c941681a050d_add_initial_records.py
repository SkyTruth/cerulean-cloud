"""Add initial records

Revision ID: c941681a050d
Revises: 39277f6278f4
Create Date: 2022-07-06 12:49:46.037868

"""
from datetime import datetime

from geoalchemy2.shape import from_shape
from shapely.geometry import box
from sqlalchemy import orm

import cerulean_cloud.database_schema as database_schema
from alembic import op

# revision identifiers, used by Alembic.
revision = "c941681a050d"
down_revision = "39277f6278f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """add initial rows"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with session.begin():
        infra_distance = database_schema.InfraDistance(
            name="Infrastructure Distance",
            source="URL",
            start_time=datetime.now(),
            end_time=datetime.now(),
            url="https://storage.googleapis.com/ceruleanml/aux_datasets/infra_locations_01_cogeo.tiff",
            geometry=from_shape(box(*[-179, -89, 179, 89])),
        )
        model = database_schema.Model(
            name="a_model",
            file_path="experiments/cv2/24_May_2022_01_49_56_fastai_unet/tracing_cpu_test_1batch_18_512_0.082.pt",
        )
        vessel_density = database_schema.VesselDensity(
            name="Vessel Density",
            geometry=from_shape(box(*[-179, -89, 179, 89])),
            source="a source",
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        session.add(infra_distance)
        session.add(model)
        session.add(vessel_density)


def downgrade() -> None:
    """drop initial rows"""
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with session.begin():
        infra_distance = (
            session.query(database_schema.InfraDistance)
            .filter_by(
                url="https://storage.googleapis.com/ceruleanml/aux_datasets/infra_locations_01_cogeo.tiff"
            )
            .one_or_none()
        )
        model = (
            session.query(database_schema.Model)
            .filter_by(
                file_path="experiments/cv2/24_May_2022_01_49_56_fastai_unet/tracing_cpu_test_1batch_18_512_0.082.pt"
            )
            .one_or_none()
        )
        vessel_density = (
            session.query(database_schema.VesselDensity)
            .filter_by(name="Vessel Density")
            .one_or_none()
        )

        session.delete(infra_distance)
        session.delete(model)
        session.delete(vessel_density)
