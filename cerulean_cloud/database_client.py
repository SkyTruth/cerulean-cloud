"""Client code to interact with the database"""
import json
import os
from datetime import datetime
from typing import Optional

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


class DatabaseClient:
    """the database client"""

    def __init__(self, engine):
        """init"""
        self.engine = engine

    def __enter__(self):
        """open session"""
        self.session = Session(self.engine)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """close session"""
        self.session.close()

    def get_trigger(self, trigger: Optional[int] = None):
        """get trigger from id"""
        if trigger:
            return existing_or_new(self, database_schema.Trigger, id=trigger)
        else:
            return existing_or_new(
                self.session,
                database_schema.Trigger,
                name="MANUAL",
                trigger_logs="",
                trigger_type="MANUAL",
            )

    def get_model(self, model_path: str):
        """get model from path"""
        return existing_or_new(
            self.session, database_schema.Model, file_path=model_path, name=model_path
        )

    def get_sentinel1_grd(self, sceneid: str, scene_info: dict, titiler_url: str):
        """get sentinel1 record"""
        return existing_or_new(
            self.session,
            database_schema.Sentinel1Grd,
            scene_id=sceneid,
            absolute_orbit_number=scene_info["absoluteOrbitNumber"],
            mode=scene_info["mode"],
            polarization=scene_info["polarization"],
            scihub_ingestion_time=scene_info["sciHubIngestion"],
            start_time=scene_info["startTime"],
            end_time=scene_info["end_time"],
            meta=json.dumps(scene_info),
            url=titiler_url,
            geometry=scene_info["footprint"],
        )

    def get_vessel_density(self, vessel_density: str):
        """get vessel density"""
        return existing_or_new(
            self.session, database_schema.VesselDensity, name=vessel_density
        )

    def get_infra_distance(self, infra_distance_url: str):
        """get infra distance"""
        return existing_or_new(
            self.session, database_schema.InfraDistance, url=infra_distance_url
        )

    def get_slick_class(self, slick_class: str):
        """get slick class"""
        return existing_or_new(
            self.session, database_schema.SlickClass, name=slick_class, active=True
        )

    def add_orchestrator(
        self,
        inference_start_time,
        inference_end_time,
        base_tiles,
        offset_tiles,
        git_hash,
        git_tag,
        zoom,
        scale,
        bounds,
        trigger,
        model,
        sentinel1_grd,
        vessel_density,
        infra_distance,
    ):
        """add a new orchestrator"""
        orchestrator_run = database_schema.OrchestratorRun(
            inference_start_time=inference_start_time,
            inference_end_time=inference_end_time,
            base_tiles=base_tiles,
            offset_tiles=offset_tiles,
            git_hash=git_hash,
            git_tag=git_tag,
            inference_run_logs="",
            zoom=zoom,
            scale=scale,
            success=False,
            geometry=from_shape(box(*bounds)),
            trigger1=trigger,
            model1=model,
            sentinel1_grd1=sentinel1_grd,
            vessel_density1=vessel_density,
            infra_distance1=infra_distance,
        )
        with self.session.begin():
            self.session.add(orchestrator_run)
        return orchestrator_run

    def add_slick(self, orchestrator_run, slick_timestamp, slick_shape, slick_class):
        """add a slick"""
        slick = database_schema.Slick(
            slick_timestamp=datetime.now(),
            geometry=from_shape(slick_shape),
            active=True,
            validated=False,
            slick_class1=slick_class,
            orchestrator_run1=orchestrator_run,
        )
        with self.session.begin():
            self.session.add(slick)
        return slick


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

            session.add(slick)
        except:  # noqa: E722
            session.rollback()
            raise
        else:
            session.commit()
