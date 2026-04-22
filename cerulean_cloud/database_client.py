"""Client code to interact with the database"""

import os
from typing import Optional, Sequence

import pandas as pd
from dateutil.parser import parse
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, Polygon, base, box, shape
from sqlalchemy import and_, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
    id_temp = kwargs.get("id")
    if id_temp is not None and type(id_temp) is not int:
        try:
            kwargs["id"] = int(id_temp)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid id value: {id_temp!r}") from exc
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

    async def get_cls(self, short_name: str, error_if_absent: bool = True):
        """get classification row from short_name"""
        return await get(
            self.session,
            db.Cls,
            error_if_absent=error_if_absent,
            short_name=short_name,
        )

    async def get_cls_subtree_ids(self, root_short_name: str):
        """Return a mapping of class short names to ids for a root class and descendants."""
        cls_tree = (
            select(
                db.Cls.id.label("id"),
                db.Cls.short_name.label("short_name"),
            )
            .where(db.Cls.short_name == root_short_name)
            .cte(name="cls_tree", recursive=True)
        )
        cls_tree = cls_tree.union_all(
            select(db.Cls.id, db.Cls.short_name).join(
                cls_tree, db.Cls.supercls == cls_tree.c.id
            )
        )
        result = await self.session.execute(
            select(cls_tree.c.short_name, cls_tree.c.id)
        )
        cls_ids = {short_name: cls_id for short_name, cls_id in result.all()}
        if not cls_ids:
            raise InstanceNotFoundError(
                f"Cls subtree not found for root_short_name={root_short_name!r}"
            )
        return cls_ids

    async def get_aoi_access_configs(
        self, short_names: Optional[Sequence[str]] = None
    ) -> list[dict]:
        """
        Return AOI access configuration rows needed by AOIJoiner.

        This uses a focused SQL query instead of the ORM model because AOI access
        config columns may evolve independently of the long-lived schema model.
        """
        if short_names:
            short_names = list(short_names)
            query = text(
                """
                SELECT
                    short_name,
                    geometry_source_uri,
                    pmtiles_uri,
                    dataset_version,
                    filter_toggle,
                    read_perm
                FROM aoi_type
                WHERE short_name = ANY(:short_names)
                ORDER BY id
                """
            )
            result = await self.session.execute(
                query, {"short_names": short_names}
            )
        else:
            query = text(
                """
                SELECT
                    short_name,
                    geometry_source_uri,
                    pmtiles_uri,
                    dataset_version,
                    filter_toggle,
                    read_perm
                FROM aoi_type
                ORDER BY id
                """
            )
            result = await self.session.execute(query)

        return [dict(row._mapping) for row in result.fetchall()]

    async def get_or_insert_sentinel1_grd(
        self, scene_id: str, scene_info: dict, titiler_url: str
    ):
        """get or insert sentinel1 record"""
        existing = await get(
            self.session, db.Sentinel1Grd, error_if_absent=False, scene_id=scene_id
        )
        if existing:
            return existing
        shape_s1 = shape(scene_info["footprint"])
        if isinstance(shape_s1, Polygon):
            geom = from_shape(shape_s1)
        elif isinstance(shape_s1, MultiPolygon):
            geom = from_shape(shape_s1.geoms[0])
        s1_grd = await insert(
            self.session,
            db.Sentinel1Grd,
            scene_id=scene_id,
            absolute_orbit_number=scene_info["absoluteOrbitNumber"],
            mode=scene_info["mode"],
            polarization=scene_info["polarization"],
            scihub_ingestion_time=parse(scene_info["sciHubIngestion"], ignoretz=True),
            start_time=parse(scene_info["startTime"]),
            end_time=parse(scene_info["stopTime"]),
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
        centerlines,
        aspect_ratio_factor,
        cls_id=None,
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
            centerlines=centerlines,
            aspect_ratio_factor=aspect_ratio_factor,
            cls=cls_id,
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

        # Create a mapping from table names (stored in SourceType) to ORM classes
        tablename_to_class = {
            subclass.__tablename__: subclass for subclass in db.Source.__subclasses__()
        }

        # Define insertion columns, based on schema source type
        common_cols = [c.name for c in db.Source.__table__.columns]
        insert_cols = {
            1: [c.name for c in db.SourceVessel.__table__.columns],  # Vessels
            2: [c.name for c in db.SourceInfra.__table__.columns],  # Infrastructure
            3: [c.name for c in db.SourceDark.__table__.columns],  # Dark Vessels
            4: [c.name for c in db.SourceNatural.__table__.columns],  # Natural Seeps
        }
        allowed_cols = set(common_cols + insert_cols[int(source_row["type"])])
        insert_dict = {}
        for k, v in source_row.items():
            if k not in allowed_cols:
                continue
            if isinstance(v, base.BaseGeometry):
                v = str(v)
            if pd.isna(v):
                continue
            insert_dict[k] = v

        source_type_obj = await get(self.session, db.SourceType, id=source_row["type"])

        source = await insert(
            self.session, tablename_to_class[source_type_obj.table_name], **insert_dict
        )
        await self.session.flush()
        return source

    async def insert_slick_to_source(self, **kwargs):
        """Insert a new slick_to_source, or update it if it already exists."""
        insert_stmt = pg_insert(db.SlickToSource.__table__).values(**kwargs)
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["slick", "source"],
            set_={
                "active": insert_stmt.excluded.active,
                "git_hash": insert_stmt.excluded.git_hash,
                "git_tag": insert_stmt.excluded.git_tag,
                "coincidence_score": insert_stmt.excluded.coincidence_score,
                "collated_score": insert_stmt.excluded.collated_score,
                "rank": insert_stmt.excluded.rank,
                "geojson_fc": insert_stmt.excluded.geojson_fc,
                "geometry": insert_stmt.excluded.geometry,
            },
        )
        return await self.session.execute(upsert_stmt)

    async def get_slicks_from_scene_id(
        self,
        scene_id,
        min_conf=0.0,
        with_sources=True,
        without_sources=True,
        active=True,
        exclude_not_oil=False,
    ):
        """
        Asynchronously queries the database to fetch slicks for a given scene ID based on source associations.

        Args:
            scene_id (str): The ID of the scene for which slicks are needed.
            min_conf (float): Minimum machine confidence for slicks to return. Default is 0.0.
            with_sources (bool): If True, fetch slicks with associated sources. Default is True.
            without_sources (bool): If True, fetch slicks without associated sources. Default is True.
            active (bool): Flag to filter slicks based on their active status. Default is True.
            exclude_not_oil (bool): If True, omit slicks classified as not-oil (subclasses such as land and sea ice).

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
            .distinct(db.Slick.id)  # distinct on the primary key
            .outerjoin(db.SlickToSource, db.Slick.id == db.SlickToSource.slick)
            .join(db.OrchestratorRun)
            .join(db.Sentinel1Grd)
        )

        conditions = [
            db.Sentinel1Grd.scene_id == scene_id,
            db.Slick.active == active,
            db.Slick.machine_confidence > min_conf,
        ]
        if exclude_not_oil:
            not_oil_clses = (
                select(db.Cls.id.label("id"))
                .where(db.Cls.short_name == "NOT_OIL")
                .cte(name="not_oil_clses", recursive=True)
            )
            not_oil_clses = not_oil_clses.union_all(
                select(db.Cls.id).join(
                    not_oil_clses, db.Cls.supercls == not_oil_clses.c.id
                )
            )
            conditions.append(~db.Slick.cls.in_(select(not_oil_clses.c.id)))

        source_conditions = []
        if not with_sources and not without_sources:
            return []
        if with_sources:
            # Slicks that have at least one associated source
            source_conditions.append(db.SlickToSource.slick is not None)
        if without_sources:
            # Slicks that have no associated sources
            source_conditions.append(db.SlickToSource.slick is None)

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
            .execution_options(synchronize_session=False)
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
            .execution_options(synchronize_session=False)
            .where(db.SlickToSource.slick == slick_id)
            .values(active=False)
        )

    async def lock_slick(self, slick_id):
        """Serialize source-association writes for a slick."""
        await self.session.execute(
            select(db.Slick.id).where(db.Slick.id == slick_id).with_for_update()
        )

    async def deactivate_sources_for_slick_by_source_type(
        self, slick_id, source_type_short_names
    ):
        """Deactivate slick_to_source rows for a slick limited to source types."""
        if not source_type_short_names:
            return

        await self.session.execute(
            update(db.SlickToSource)
            .execution_options(synchronize_session=False)
            .where(
                and_(
                    db.SlickToSource.slick == slick_id,
                    db.SlickToSource.source.in_(
                        select(db.Source.id)
                        .join(db.Source.source_type)
                        .where(db.SourceType.short_name.in_(source_type_short_names))
                    ),
                )
            )
            .values(active=False)
        )

    async def get_id_collated_score_pairs(self, slick_id):
        """
        Return active slick_to_source ranking rows for a given slick.

        :param slick_id: The ID of the slick to query.
        :return: List of tuples containing
            (slick_to_source_id, collated_score, source_type_short_name).
        """
        query = (
            select(
                db.SlickToSource.id,
                db.SlickToSource.collated_score,
                db.SourceType.short_name,
            )
            .select_from(db.SlickToSource)
            .join(db.SlickToSource.source1)
            .join(db.Source.source_type)
            .where(
                and_(
                    db.SlickToSource.slick == slick_id,
                    db.SlickToSource.active.is_(True),
                )
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
