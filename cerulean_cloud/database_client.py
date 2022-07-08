"""Client code to interact with the database"""
import os
from typing import Optional

import geoalchemy2.functions as func
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, Polygon, box, shape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

import cerulean_cloud.database_schema as database_schema


def get_engine(db_url: str = os.getenv("DB_URL")):
    """get database engine"""
    return create_async_engine(db_url, echo=True)


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

    async def __aenter__(self):
        """open session"""
        self.session = AsyncSession(self.engine)
        return self

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
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
        s1_grd = (
            self.session.query(database_schema.Sentinel1Grd)
            .filter_by(scene_id=sceneid)
            .one_or_none()
        )
        if not s1_grd:
            shape_s1 = shape(scene_info["footprint"])
            if isinstance(Polygon, shape_s1):
                geom = from_shape(shape_s1)
            elif isinstance(MultiPolygon, shape_s1):
                geom = from_shape(shape_s1.geoms[0])
            s1_grd = existing_or_new(
                self.session,
                database_schema.Sentinel1Grd,
                scene_id=sceneid,
                absolute_orbit_number=scene_info["absoluteOrbitNumber"],
                mode=scene_info["mode"],
                polarization=scene_info["polarization"],
                scihub_ingestion_time=scene_info["sciHubIngestion"],
                start_time=scene_info["startTime"],
                end_time=scene_info["stopTime"],
                meta=scene_info,
                url=titiler_url,
                geometry=geom,
            )
        return s1_grd

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
            self.session,
            database_schema.SlickClass,
            value=int(slick_class),
            name=str(int(slick_class)),
            active=True,
        )

    def add_orchestrator(
        self,
        inference_start_time,
        inference_end_time,
        base_tiles,
        offset_tiles,
        git_hash,
        git_tag,
        inference_run_logs,
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
            inference_run_logs=inference_run_logs,
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
        return orchestrator_run

    def add_slick(self, orchestrator_run, slick_timestamp, slick_shape, slick_class):
        """add a slick"""
        s = shape(slick_shape)
        if not isinstance(s, MultiPolygon):
            s = MultiPolygon([s])
        slick = database_schema.Slick(
            slick_timestamp=slick_timestamp,
            geometry=from_shape(s),
            active=True,
            validated=False,
            slick_class1=slick_class,
            orchestrator_run1=orchestrator_run,
        )
        return slick

    async def add_eez_to_slick(self, slick):
        """add a slick"""
        eez = await self.session.execute(
            select(database_schema.Eez).where(
                func.ST_Intersects(slick.geometry, database_schema.Eez.geometry)
            )
        )
        for e in eez.scalars().all():
            print(f"Adding to {slick} eez {e}")
            eez_to_slick = database_schema.SlickToEez(slick1=slick, eez1=e)
            self.session.add(eez_to_slick)
