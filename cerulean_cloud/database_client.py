"""Client code to interact with the database"""
import os
from typing import Optional

from dateutil.parser import parse
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, Polygon, box, shape
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

import cerulean_cloud.database_schema as db


class InstanceNotFoundError(Exception):
    """Raised when an instance is not found in the database."""

    pass


def get_engine(db_url: str = os.getenv("DB_URL")):
    """get database engine"""
    return create_async_engine(db_url, echo=False)


async def get(sess, kls, error_if_absent=True, **kwargs):
    """Return instance if exists else None"""
    res = await sess.execute(select(kls).filter_by(**kwargs))
    res = res.scalars().first()
    if not res and error_if_absent:
        raise InstanceNotFoundError(
            f"Instance of {kls} not found with parameters {kwargs}"
        )
    return res


def insert(sess, kls, **kwargs):
    """Create an instance"""
    res = kls(**kwargs)
    sess.add(res)
    return res


async def get_or_insert(sess, kls, **kwargs):
    """Check if instance exists, creates it if not"""
    return (await get(sess, kls, False, **kwargs)) or insert(sess, kls, **kwargs)


class DatabaseClient:
    """the database client"""

    def __init__(self, engine):
        """init"""
        self.engine = engine

    async def __aenter__(self):
        """open session"""
        self.session = AsyncSession(self.engine, expire_on_commit=False)
        return self

    async def __aexit__(self, exc_type, exc_value, exc_traceback):
        """close session"""
        await self.session.close()

    async def get_trigger(self, trigger: Optional[int] = None):
        """get trigger from id"""
        if trigger:
            return await get_or_insert(self.session, db.Trigger, id=trigger)
        else:
            return await get_or_insert(
                self.session,
                db.Trigger,
                trigger_logs="",
                trigger_type="MANUAL",
            )

    async def get_model(self, model_path: str):
        """get model from path"""
        return await get(self.session, db.Model, file_path=model_path)

    async def get_layer(self, short_name: str):
        """get layer from short_name"""
        return await get(self.session, db.Layer, short_name=short_name)

    async def get_sentinel1_grd(self, sceneid: str, scene_info: dict, titiler_url: str):
        """get sentinel1 record"""
        shape_s1 = shape(scene_info["footprint"])
        if isinstance(shape_s1, Polygon):
            geom = from_shape(shape_s1)
        elif isinstance(shape_s1, MultiPolygon):
            geom = from_shape(shape_s1.geoms[0])

        s1_grd = await get_or_insert(
            self.session,
            db.Sentinel1Grd,
            scene_id=sceneid,
            absolute_orbit_number=scene_info["absoluteOrbitNumber"],
            mode=scene_info["mode"],
            polarization=scene_info["polarization"],
            scihub_ingestion_time=parse(scene_info["sciHubIngestion"], ignoretz=True),
            start_time=parse(scene_info["startTime"]),
            end_time=parse(scene_info["stopTime"]),
            meta=scene_info,
            url=titiler_url,
            geometry=geom,
        )
        return s1_grd

    async def add_orchestrator(
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
    ):
        """add a new orchestrator"""
        orchestrator_run = await get_or_insert(
            self.session,
            db.OrchestratorRun,
            inference_start_time=inference_start_time,
            inference_end_time=inference_end_time,
            base_tiles=base_tiles,
            offset_tiles=offset_tiles,
            git_hash=git_hash,
            git_tag=git_tag,
            inference_run_logs=inference_run_logs,
            zoom=zoom,
            scale=scale,
            geometry=from_shape(box(*bounds)),
            trigger1=trigger,
            model1=model,
            sentinel1_grd1=sentinel1_grd,
        )
        return orchestrator_run

    async def add_slick(
        self,
        orchestrator_run,
        slick_timestamp,
        slick_shape,
        inference_idx,
        machine_confidence,
    ):
        """add a slick"""
        s = shape(slick_shape)
        if not isinstance(s, MultiPolygon):
            s = MultiPolygon([s])

        slick = insert(
            self.session,
            db.Slick,
            slick_timestamp=slick_timestamp,
            geometry=from_shape(s),
            inference_idx=inference_idx,
            active=True,
            orchestrator_run1=orchestrator_run,
            machine_confidence=machine_confidence,
        )
        return slick

    async def get_slicks_without_sources_from_scene_id(self, scene_id, active=True):
        """
        Asynchronously queries the database to fetch slicks without associated sources for a given scene ID.

        Args:
            scene_id (str): The ID of the scene for which slicks are needed.
            active (bool): Flag to filter slicks based on their active status. Default is True.

        Returns:
            list: A list of Slick objects that do not have associated sources and belong to the specified scene.

        Notes:
            - The function uses SQLAlchemy for database queries.
            - It joins multiple tables: `db.Slick`, `db.SlickToSource`, `db.OrchestratorRun`, and `db.Sentinel1Grd`.
            - The query uses an outer join to filter out slicks that have associated sources.
        """

        query = (
            select(db.Slick)
            .outerjoin(db.SlickToSource, db.Slick.id == db.SlickToSource.slick)
            .join(db.OrchestratorRun)
            .join(db.Sentinel1Grd)
            .where(
                and_(
                    db.SlickToSource.slick is None,
                    db.Sentinel1Grd.scene_id == scene_id,
                    db.Slick.active == active,
                )
            )
        )
        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_scene_from_id(self, scene_id):
        """
        Asynchronously fetches a scene object from the database based on its ID.

        Args:
            scene_id (str): The ID of the scene to fetch.

        Returns:
            Scene Object: The fetched Sentinel1Grd scene object, or `None` if not found.

        Notes:
            - The function delegates the actual database fetch operation to a generic `get()` function.
            - It looks for a scene in the `db.Sentinel1Grd` table.
        """
        return await get(self.session, db.Sentinel1Grd, False, scene_id=scene_id)
