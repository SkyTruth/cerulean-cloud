"""Client code to interact with the database"""

import os
from typing import Optional

import pandas as pd
from dateutil.parser import parse
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, Polygon, base, box, shape
from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

import cerulean_cloud.database_schema as db


class InstanceNotFoundError(Exception):
    """Raised when an instance is not found in the database."""

    pass


def get_engine(db_url: str = os.getenv("DB_URL")):
    """get database engine"""
    # Connect args ref: https://docs.sqlalchemy.org/en/20/core/engines.html#use-the-connect-args-dictionary-parameter
    # Note: statement timeout is assumed to be in MILIseconds if no unit is
    # specified (as is the case here)
    # Ref: https://www.postgresql.org/docs/current/runtime-config-client.html#GUC-STATEMENT-TIMEOUT
    # Note: specifying a 1 minute timeout per statement, since each orchestrator
    # run may attempt to execute many statements
    return create_async_engine(
        db_url,
        echo=False,
        # connect_args={"options": f"-c statement_timeout={1000 * 60}"},
        connect_args={"command_timeout": 60},
        pool_size=1,  # Default pool size
        max_overflow=0,  # Default max overflow
        pool_timeout=300,  # Default pool timeout
        pool_recycle=600,  # Default pool recycle
    )


async def get(sess, kls, error_if_absent=True, **kwargs):
    """Return instance if exists else None"""
    res = await sess.execute(select(kls).filter_by(**kwargs))
    res = res.scalars().first()
    if not res and error_if_absent:
        raise InstanceNotFoundError(
            f"Instance of {kls} not found with parameters {kwargs}"
        )
    return res


async def insert(sess, kls, **kwargs):
    """Create an instance"""
    res = kls(**kwargs)
    sess.add(res)
    return res


async def get_or_insert(sess, kls, **kwargs):
    """Check if instance exists, creates it if not"""
    return (await get(sess, kls, False, **kwargs)) or (
        await insert(sess, kls, **kwargs)
    )


async def update_object(
    sess, kls, filter_kwargs: dict, update_kwargs: dict, flush: bool = False
):
    """
    Generic method to update a record in the database.

    Args:
        kls: The SQLAlchemy model class.
        filter_kwargs (dict): Key-value pairs to filter the record to update.
        update_kwargs (dict): Key-value pairs of fields to update.

    Returns:
        The updated instance.
    """
    instance = await get(sess, kls, True, **filter_kwargs)
    for key, value in update_kwargs.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
        else:
            raise AttributeError(f"{kls.__name__} has no attribute '{key}'")
    sess.add(instance)

    if flush:
        await sess.flush()

    return instance


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

    async def get_db_model(self, model_path: str):
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
        # use buffer(0) to attempt to fix any invalid geometries
        s = shape(slick_shape).buffer(0)
        if not isinstance(s, MultiPolygon):
            s = MultiPolygon([s])

        slick = await insert(
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

    async def get_source(self, source_type, ext_id, error_if_absent=False):
        """get existing source"""
        return await get(
            self.session, db.Source, error_if_absent, type=source_type, ext_id=ext_id
        )

    async def get_or_insert_ranked_source(self, source_row):
        """add a new source"""
        existing_source = await self.get_source(
            source_row["type"], source_row["ext_id"]
        )
        if existing_source:
            return existing_source

        for k, v in source_row.items():
            if isinstance(v, base.BaseGeometry):
                source_row[k] = str(v)

        # Create a mapping from table names (stored in SourceType) to ORM classes
        tablename_to_class = {
            subclass.__tablename__: subclass for subclass in db.Source.__subclasses__()
        }

        # Define insertion columns, based on schema source type
        common_cols = [c.name for c in db.Source.__table__.columns]
        insert_cols = {
            1: [c.name for c in db.SourceVessel.__table__.columns],  # Vessels
            2: [c.name for c in db.SourceInfra.__table__.columns],  # Infrastructure
        }
        insert_dict = {
            k: v
            for k, v in source_row.items()
            if not pd.isna(v) and k in (common_cols + insert_cols[source_row["type"]])
        }

        source_type_obj = await get(self.session, db.SourceType, id=source_row["type"])

        source = await insert(
            self.session, tablename_to_class[source_type_obj.table_name], **insert_dict
        )
        await self.session.flush()
        return source

    async def insert_slick_to_source(self, **kwargs):
        """add a new slick_to_source"""
        return await insert(self.session, db.SlickToSource, **kwargs)

    async def get_slicks_from_scene_id(
        self,
        scene_id,
        min_conf=0.0,
        with_sources=True,
        without_sources=True,
        active=True,
    ):
        """
        Asynchronously queries the database to fetch slicks for a given scene ID based on source associations.

        Args:
            scene_id (str): The ID of the scene for which slicks are needed.
            min_conf (float): Minimum machine confidence for slicks to return. Default is 0.0.
            with_sources (bool): If True, fetch slicks with associated sources. Default is True.
            without_sources (bool): If True, fetch slicks without associated sources. Default is True.
            active (bool): Flag to filter slicks based on their active status. Default is True.

        Returns:
            list: A list of Slick objects based on the specified filters.

        Raises:
            ValueError: If both `with_sources` and `without_sources` are set to False.

        Notes:
            - The function uses SQLAlchemy for database queries.
            - It joins multiple tables: `db.Slick`, `db.SlickToSource`, `db.OrchestratorRun`, and `db.Sentinel1Grd`.
            - It conditionally filters slicks based on the `with_sources` and `without_sources` flags.
        """
        query = (
            select(db.Slick)
            .distinct()
            .outerjoin(db.SlickToSource, db.Slick.id == db.SlickToSource.slick)
            .join(db.OrchestratorRun)
            .join(db.Sentinel1Grd)
        )

        conditions = [
            db.Sentinel1Grd.scene_id == scene_id,
            db.Slick.active == active,
            db.Slick.machine_confidence > min_conf,
        ]

        source_conditions = []
        if not with_sources and not without_sources:
            return []
        if with_sources:
            # Slicks that have at least one associated source
            source_conditions.append(db.SlickToSource.slick != None)  # noqa
        if without_sources:
            # Slicks that have no associated sources
            source_conditions.append(db.SlickToSource.slick == None)  # noqa

        if source_conditions:
            # Combine source conditions with OR logic
            conditions.append(or_(*source_conditions))

        query = query.where(and_(*conditions))
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

    async def deactivate_stale_slicks_from_scene_id(self, scene_id):
        """
        Asynchronously queries the database to fetch slicks without associated sources for a given scene ID.

        Args:
            scene_id (str): The ID of the scene for which slicks are needed.

        Returns:
            (integer): count of slicks updated

        Notes:
            - The function uses SQLAlchemy for database queries.
            - It joins multiple tables: `db.Slick`, `db.OrchestratorRun`, and `db.Sentinel1Grd`.
        """
        # Create an update query object
        update_query = (
            update(db.Slick)
            .where(
                db.Slick.id.in_(
                    select(db.Slick.id)
                    .join(db.OrchestratorRun)
                    .join(db.Sentinel1Grd)
                    .where(
                        and_(
                            db.Sentinel1Grd.scene_id == scene_id,
                            db.Slick.active,
                        )
                    )
                )
            )
            .values(active=False)
        )

        # Execute the update query and get the result
        result = await self.session.execute(update_query)

        # Return the number of rows updated
        return result.rowcount

    async def deactivate_sources_for_slick(self, slick_id):
        """deactivate sources for slick"""
        await self.session.execute(
            update(db.SlickToSource)
            .where(db.SlickToSource.slick == slick_id)
            .values(active=False)
        )

    async def get_previous_asa(self, slick_id):
        """Return a list of ASA types that have been run for a slick."""
        return (
            (
                await self.session.execute(
                    select(db.Source.type)
                    .join(db.SlickToSource.source1)
                    .where(
                        and_(
                            db.SlickToSource.slick == slick_id,
                            db.SlickToSource.active,
                        )
                    )
                )
            )
            .scalars()
            .all()
        )

    async def get_id_collated_score_pairs(self, slick_id):
        """
        Return a list of (id, collated_score) pairs for a given slick.

        :param slick_id: The ID of the slick to query.
        :return: List of tuples containing (id, collated_score).
        """
        query = select(db.SlickToSource.id, db.SlickToSource.collated_score).where(
            and_(
                db.SlickToSource.slick == slick_id,
                db.SlickToSource.active.is_(True),
            )
        )
        result = await self.session.execute(query)
        return result.all()

    async def update_slick_to_source(self, filter_kwargs: dict, update_kwargs: dict):
        """
        Update a SlickToSource record
        """
        return await update_object(
            self.session, db.SlickToSource, filter_kwargs, update_kwargs
        )

    # EditTheDatabase
