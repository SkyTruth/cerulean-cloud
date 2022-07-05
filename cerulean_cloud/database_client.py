"""Client code to interact with the database"""
import os
from datetime import datetime

from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, box
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import cerulean_cloud.database_schema as database_schema


def get_engine(db_url: str = os.getenv("DB_URL")):
    """get database engine"""
    return create_engine(db_url)


def existing_or_new(sess, kls, **kwargs):
    """Check if instance exists, creates it if not"""
    inst = sess.query(kls).filter_by(**kwargs).one_or_none()
    if not inst:
        inst = kls(**kwargs)
    return inst


def a_function():
    """test"""
    engine = get_engine()
    with Session(engine) as session:
        try:
            trigger = existing_or_new(session, database_schema.Trigger, id=8)

            model = existing_or_new(session, database_schema.Model, id=9)

            sentinel1_grd = existing_or_new(
                session,
                database_schema.Sentinel1Grd,
                scene_id="S1A_IW_GRDH_1SDV_20200729T034859_20200729T034924_033664_03E6D3_93EF",
            )

            vessel_density = existing_or_new(
                session, database_schema.VesselDensity, id=4
            )

            infra_distance = existing_or_new(
                session, database_schema.InfraDistance, id=7
            )

            slick_class = existing_or_new(
                session, database_schema.SlickClass, name="test", active=True
            )

            orchestrator_run = database_schema.OrchestratorRun(
                inference_start_time=datetime.now(),
                inference_end_time=datetime.now(),
                base_tiles=60,
                offset_tiles=100,
                git_hash="abc",
                inference_run_logs="",
                geometry=from_shape(box(*[32.989094, 43.338009, 36.540836, 45.235191])),
                trigger1=trigger,
                model1=model,
                sentinel1_grd1=sentinel1_grd,
                vessel_density1=vessel_density,
                infra_distance1=infra_distance,
            )

            slick = database_schema.Slick(
                slick_timestamp=datetime.now(),
                geometry=from_shape(
                    MultiPolygon(polygons=[box(*[33, 44, 33.540836, 44.235191])])
                ),
                active=True,
                validated=False,
                slick_class1=slick_class,
                orchestrator_run1=orchestrator_run,
            )
            slick2 = database_schema.Slick(
                slick_timestamp=datetime.now(),
                geometry=from_shape(
                    MultiPolygon(polygons=[box(*[33, 44, 33.540836, 44.235191])])
                ),
                active=True,
                validated=False,
                slick_class1=slick_class,
                orchestrator_run1=orchestrator_run,
            )
            session.add(slick)
            session.add(slick2)
        except:  # noqa: E722
            session.rollback()
            raise
        else:
            session.commit()
