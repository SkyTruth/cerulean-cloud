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
            name="experiments/cv2/24_May_2022_01_49_56_fastai_unet/tracing_cpu_test_1batch_18_512_0.082.pt",
            file_path="experiments/cv2/24_May_2022_01_49_56_fastai_unet/tracing_cpu_test_1batch_18_512_0.082.pt",
        )
        vessel_density = database_schema.VesselDensity(
            name="Vessel Density",
            geometry=from_shape(box(*[-179, -89, 179, 89])),
            source="a source",
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        aoi_types = [
            database_schema.AoiType(
                table_name="aoi_eez",
                long_name="Exclusive Economic Zone",
                short_name="EEZ",
                source_url="https://www.marineregions.org/eez.php",
                citation="Flanders Marine Institute (2019). Maritime Boundaries Geodatabase, version 11. Available online at https://www.marineregions.org/. https://doi.org/10.14284/382.",
                update_time=datetime.now(),
            ),
            database_schema.AoiType(
                table_name="aoi_iho",
                long_name="IHO Sea Areas",
                short_name="IHO",
                source_url="https://www.marineregions.org/sources.php#iho",
                citation="Flanders Marine Institute (2018). IHO Sea Areas, version 3. Available online at https://www.marineregions.org/. https://doi.org/10.14284/323.",
                update_time=datetime.now(),
            ),
            database_schema.AoiType(
                table_name="aoi_mpa",
                long_name="Marine Protected Area",
                short_name="MPA",
                source_url="https://www.protectedplanet.net/en/thematic-areas/marine-protected-areas",
                citation="UNEP-WCMC and IUCN (2023), Protected Planet: The World Database on Protected Areas (WDPA) and World Database on Other Effective Area-based Conservation Measures (WD-OECM) [Online], July 2023, Cambridge, UK: UNEP-WCMC and IUCN. Available at: www.protectedplanet.net.",
                update_time=datetime.now(),
            ),
            database_schema.AoiType(
                table_name="aoi_user",
                long_name="User-generated",
                short_name="USER",
                update_time=datetime.now(),
            ),
        ]

        source_types = [
            database_schema.SourceType(
                table_name="source_vessel",
                long_name="Vessel Source",
                short_name="VESSEL",
                citation="AIS from GFW",
            ),
            database_schema.SourceType(
                table_name="source_infra",
                long_name="Infrastructure Source",
                short_name="INFRA",
                citation="SkyTruth",
            ),
        ]

        session.add(infra_distance)
        session.add(model)
        session.add(vessel_density)
        session.add_all(aoi_types)
        session.add_all(source_types)


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
        aoi_types = (
            session.query(database_schema.AoiType)
            .filter(
                database_schema.AoiType.short_name.in_(["EEZ", "IHO", "MPA", "USER"])
            )
            .all()
        )
        source_types = (
            session.query(database_schema.SourceType)
            .filter(database_schema.SourceType.short_name.in_(["VESSEL", "INFRA"]))
            .all()
        )

        session.delete(infra_distance)
        session.delete(model)
        session.delete(vessel_density)
        for aoi_type in aoi_types:
            session.delete(aoi_type)
        for source_type in source_types:
            session.delete(source_type)
