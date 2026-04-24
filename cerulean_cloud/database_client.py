"""Client code to interact with the database"""

import os
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from dateutil.parser import parse
from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, Polygon, base, box, shape
from sqlalchemy import and_, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

import cerulean_cloud.database_schema as db


USER_AOI_TYPE_SHORT_NAME = "USER"


def _coerce_aoi_geometry(geometry):
    """Return a valid MultiPolygon for storage in public.aoi.geometry."""
    geom = geometry if isinstance(geometry, base.BaseGeometry) else shape(geometry)
    geom = geom.buffer(0)
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    raise ValueError(
        f"AOI geometry must be a Polygon or MultiPolygon, got {geom.geom_type!r}"
    )


def _is_empty_aoi_value(value: Any) -> bool:
    """Return True for null scalar AOI payload values."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _iter_aoi_match_payload(aoi_payload: Mapping[str, Any]):
    """
    Yield normalized AOI matches from either legacy ext-id lists or rich match dicts.
    """
    for aoi_type_short_name, matches in aoi_payload.items():
        if matches is None:
            continue
        if isinstance(matches, Mapping) and "ext_id" in matches:
            matches = [matches]
        elif not isinstance(matches, (list, tuple, set)):
            matches = [matches]

        for match in matches:
            if isinstance(match, Mapping):
                ext_id = match.get("ext_id")
                name = match.get("name")
                geometry = match.get("geometry")
                user_id = match.get("user_id")
            else:
                ext_id = match
                name = None
                geometry = None
                user_id = None

            if _is_empty_aoi_value(ext_id):
                continue

            ext_id = str(ext_id)
            yield {
                "aoi_type_short_name": aoi_type_short_name,
                "ext_id": ext_id,
                "name": str(name) if not _is_empty_aoi_value(name) else ext_id,
                "geometry": geometry,
                "user_id": user_id,
            }


class InstanceNotFoundError(Exception):
    """Raised when an instance is not found in the database."""

    pass


class AmbiguousAOIError(Exception):
    """Raised when an AOI lookup matches multiple rows."""

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
        Return AOIJoiner-compatible AOI access configuration rows.

        The checked-in schema stores AOI access metadata in `aoi_type.access_type`
        plus `aoi_type.properties`. At the moment AOIJoiner can consume only
        GCS-backed AOI layers, so DB-backed AOI types are skipped unless
        explicitly requested, in which case a NotImplementedError is raised.
        """
        if short_names:
            short_names = list(short_names)
            query = text(
                """
                SELECT
                    short_name,
                    access_type,
                    properties,
                    filter_toggle,
                    read_perm
                FROM aoi_type
                WHERE short_name = ANY(:short_names)
                ORDER BY id
                """
            )
            result = await self.session.execute(query, {"short_names": short_names})
        else:
            query = text(
                """
                SELECT
                    short_name,
                    access_type,
                    properties,
                    filter_toggle,
                    read_perm
                FROM aoi_type
                ORDER BY id
                """
            )
            result = await self.session.execute(query)

        rows = []
        unsupported = []
        requested_short_names = set(short_names or [])

        for row in result.mappings():
            short_name = row["short_name"]
            access_type = row["access_type"]
            properties = row["properties"] or {}
            if not isinstance(properties, dict):
                raise ValueError(
                    f"AOI type {short_name!r} properties must be a JSON object"
                )

            if access_type != "GCS":
                if short_name in requested_short_names:
                    unsupported.append(f"{short_name} ({access_type})")
                continue

            geometry_source_uri = properties.get(
                "geometry_source_uri"
            ) or properties.get("fgb_uri")
            if not geometry_source_uri:
                raise ValueError(
                    f"GCS AOI type {short_name!r} is missing properties['fgb_uri']"
                )

            ext_id_field = properties.get("ext_id_field") or properties.get(
                "ext_id_col"
            )
            if not ext_id_field:
                raise ValueError(
                    f"GCS AOI type {short_name!r} is missing properties['ext_id_field']"
                )

            name_field = (
                properties.get("name_field")
                or properties.get("display_name_field")
                or properties.get("name_col")
            )

            rows.append(
                {
                    "short_name": short_name,
                    "key": short_name,
                    "access_type": access_type,
                    "properties": properties,
                    "geometry_source_uri": geometry_source_uri,
                    "ext_id_field": ext_id_field,
                    "name_field": name_field,
                    "pmtiles_uri": properties.get("pmt_uri"),
                    "dataset_version": properties.get("dataset_version"),
                    "filter_toggle": row["filter_toggle"],
                    "read_perm": row["read_perm"],
                }
            )

        if unsupported:
            raise NotImplementedError(
                "AOIJoiner currently supports only GCS-backed AOI types; "
                f"unsupported requested AOI type(s): {', '.join(unsupported)}"
            )

        return rows

    async def get_aoi_type_ids(
        self, short_names: Optional[Sequence[str]] = None
    ) -> dict[str, int]:
        """Return a mapping of AOI type short names to ids."""
        query = select(db.AoiType.short_name, db.AoiType.id)
        if short_names:
            query = query.where(db.AoiType.short_name.in_(list(short_names)))
        result = await self.session.execute(query)
        return {short_name: aoi_type_id for short_name, aoi_type_id in result.all()}

    async def get_aoi_rows(
        self,
        aoi_type: int,
        ext_id: str,
    ) -> list[dict]:
        """Return all AOI rows matching an internal type id and external id."""
        query = text(
            """
            SELECT id, type, name, ext_id
            FROM public.aoi
            WHERE type = :aoi_type
              AND ext_id = :ext_id
            ORDER BY id
            """
        )
        result = await self.session.execute(
            query, {"aoi_type": aoi_type, "ext_id": str(ext_id)}
        )
        return [dict(row) for row in result.mappings().all()]

    async def get_aoi(
        self,
        aoi_type: int,
        ext_id: str,
        error_if_absent: bool = False,
    ):
        """Get a single AOI row by internal type id and external id."""
        rows = await self.get_aoi_rows(aoi_type, ext_id)
        if not rows:
            if error_if_absent:
                raise InstanceNotFoundError(
                    f"AOI not found with type={aoi_type!r}, ext_id={ext_id!r}"
                )
            return None
        if len(rows) > 1:
            raise AmbiguousAOIError(
                f"Multiple AOIs found with type={aoi_type!r}, ext_id={ext_id!r}"
            )
        return rows[0]

    async def resolve_single_aoi_id(
        self,
        aoi_type_short_name: str,
        ext_id: str,
        error_if_absent: bool = True,
    ) -> Optional[int]:
        """Resolve a single AOI id for a curated AOI type and external id."""
        aoi_type_ids = await self.get_aoi_type_ids([aoi_type_short_name])
        aoi_type_id = aoi_type_ids.get(aoi_type_short_name)
        if aoi_type_id is None:
            raise InstanceNotFoundError(
                f"AOI type not found for short_name={aoi_type_short_name!r}"
            )

        aoi = await self.get_aoi(aoi_type_id, ext_id, error_if_absent=error_if_absent)
        if aoi is None:
            return None
        return int(aoi["id"])

    async def create_user_aoi(
        self,
        user_id: int,
        name: str,
        geometry,
        ext_id: Optional[str] = None,
    ):
        """
        Create a USER AOI and its child-table geometry row.
        """
        aoi_type_ids = await self.get_aoi_type_ids([USER_AOI_TYPE_SHORT_NAME])
        aoi_type_id = aoi_type_ids.get(USER_AOI_TYPE_SHORT_NAME)
        if aoi_type_id is None:
            raise InstanceNotFoundError(
                f"AOI type not found for short_name={USER_AOI_TYPE_SHORT_NAME!r}"
            )

        geom = _coerce_aoi_geometry(geometry)

        aoi_user = await insert(
            self.session,
            db.AoiUser,
            type=aoi_type_id,
            name=name,
            ext_id=str(ext_id) if ext_id is not None else None,
            geometry=from_shape(geom),
            user=user_id,
            aoi_user_geometry=from_shape(geom),
        )
        await self.session.flush()
        return aoi_user

    async def get_or_insert_aoi(
        self,
        aoi_type_short_name: str,
        ext_id: str,
        name: str,
        geometry=None,
        user_id: Optional[int] = None,
    ):
        """
        Insert or resolve an AOI by `(type, ext_id)`.

        New rows require `geometry` because `public.aoi.geometry` is non-null.
        USER AOIs also require `user_id` for the `public.aoi_user` child row.
        Identity is `(type, ext_id)` and relies on the matching database
        uniqueness constraint.
        """
        aoi_type_ids = await self.get_aoi_type_ids([aoi_type_short_name])
        aoi_type_id = aoi_type_ids.get(aoi_type_short_name)
        if aoi_type_id is None:
            raise InstanceNotFoundError(
                f"AOI type not found for short_name={aoi_type_short_name!r}"
            )

        if geometry is None:
            existing_aoi = await self.get_aoi(aoi_type_id, ext_id)
            if existing_aoi:
                return existing_aoi
            raise ValueError(
                "geometry is required to insert a new AOI because "
                "public.aoi.geometry is non-null"
            )

        if aoi_type_short_name == USER_AOI_TYPE_SHORT_NAME and user_id is None:
            existing_aoi = await self.get_aoi(aoi_type_id, ext_id)
            if existing_aoi:
                return existing_aoi
            raise ValueError("user_id is required to insert a new USER AOI")

        geom = _coerce_aoi_geometry(geometry)
        insert_stmt = pg_insert(db.Aoi).values(
            type=aoi_type_id,
            name=name,
            ext_id=str(ext_id),
            geometry=from_shape(geom),
        )
        insert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["type", "ext_id"],
            set_={
                "name": insert_stmt.excluded.name,
                "geometry": insert_stmt.excluded.geometry,
            },
        ).returning(db.Aoi.id, db.Aoi.type, db.Aoi.name, db.Aoi.ext_id)

        result = await self.session.execute(insert_stmt)
        aoi = dict(result.mappings().one())

        if aoi_type_short_name == USER_AOI_TYPE_SHORT_NAME:
            child_insert = pg_insert(db.AoiUser.__table__).values(
                {
                    db.AoiUser.__table__.c.aoi_id: aoi["id"],
                    db.AoiUser.__table__.c.user: user_id,
                    db.AoiUser.__table__.c["geometry"]: from_shape(geom),
                }
            )
            child_insert = child_insert.on_conflict_do_nothing(
                index_elements=[db.AoiUser.__table__.c.aoi_id]
            )
            await self.session.execute(child_insert)

        await self.session.flush()
        return aoi

    async def insert_slick_to_aoi_from_dataframe(
        self, slick_aoi_df: pd.DataFrame
    ) -> int:
        """
        Insert public.slick_to_aoi rows from a dataframe with:
        - `slick_id`
        - `aoi_matches` dict keyed by AOI type short_name, containing rich AOI
          match dicts with `ext_id`, `name`, and `geometry`; or
        - legacy `aoi_ext_ids` dict keyed by AOI type short_name

        Example `aoi_ext_ids` payload:
        `{\"EEZ\": [\"123\"], \"IHO\": [\"456\"], \"MPA\": [\"789\"]}`
        """
        if "slick_id" not in slick_aoi_df.columns:
            raise ValueError("slick_aoi_df is missing required column: 'slick_id'")
        if not {"aoi_matches", "aoi_ext_ids"} & set(slick_aoi_df.columns):
            raise ValueError(
                "slick_aoi_df must include either 'aoi_matches' or 'aoi_ext_ids'"
            )

        aoi_pairs: set[tuple[str, str]] = set()
        flattened_rows: list[tuple[int, str, str]] = []
        upsert_candidates: dict[tuple[str, str], dict] = {}

        for row in slick_aoi_df.itertuples(index=False):
            slick_id = int(row.slick_id)
            aoi_payload = getattr(row, "aoi_matches", None)
            if _is_empty_aoi_value(aoi_payload):
                aoi_payload = getattr(row, "aoi_ext_ids", None)
            if _is_empty_aoi_value(aoi_payload):
                aoi_payload = {}
            if not isinstance(aoi_payload, Mapping):
                raise ValueError(
                    "aoi_matches/aoi_ext_ids must be a dict for "
                    f"slick_id={slick_id}, got {type(aoi_payload)!r}"
                )

            for match in _iter_aoi_match_payload(aoi_payload):
                aoi_type_short_name = match["aoi_type_short_name"]
                ext_id = match["ext_id"]
                aoi_pairs.add((aoi_type_short_name, ext_id))
                flattened_rows.append((slick_id, aoi_type_short_name, ext_id))

                if match["geometry"] is not None:
                    upsert_candidates.setdefault(
                        (aoi_type_short_name, ext_id),
                        {
                            "aoi_type_short_name": aoi_type_short_name,
                            "ext_id": ext_id,
                            "name": match["name"],
                            "geometry": match["geometry"],
                            "user_id": match["user_id"],
                        },
                    )

        if not flattened_rows:
            return 0

        for candidate in upsert_candidates.values():
            await self.get_or_insert_aoi(
                candidate["aoi_type_short_name"],
                candidate["ext_id"],
                candidate["name"],
                geometry=candidate["geometry"],
                user_id=candidate["user_id"],
            )

        aoi_type_ids = await self.get_aoi_type_ids(
            sorted({short_name for short_name, _ in aoi_pairs})
        )
        missing_types = sorted(
            {short_name for short_name, _ in aoi_pairs} - set(aoi_type_ids)
        )
        if missing_types:
            raise InstanceNotFoundError(
                f"AOI type(s) not found for short_name values: {missing_types}"
            )

        ext_ids = sorted({ext_id for _, ext_id in aoi_pairs})
        result = await self.session.execute(
            text(
                """
                SELECT MIN(id) AS id, type, ext_id
                FROM public.aoi
                WHERE type = ANY(:aoi_type_ids)
                  AND ext_id = ANY(:ext_ids)
                GROUP BY type, ext_id
                """
            ),
            {
                "aoi_type_ids": list(aoi_type_ids.values()),
                "ext_ids": ext_ids,
            },
        )
        aoi_lookup = {}
        for aoi_id, aoi_type_id, ext_id in result.fetchall():
            aoi_lookup[(aoi_type_id, str(ext_id))] = aoi_id

        missing_pairs = sorted(
            [
                (short_name, ext_id)
                for short_name, ext_id in aoi_pairs
                if (aoi_type_ids[short_name], ext_id) not in aoi_lookup
            ]
        )
        if missing_pairs:
            raise InstanceNotFoundError(
                "AOI ext_id values do not yet exist in public.aoi for: "
                + ", ".join(
                    f"{short_name}:{ext_id}" for short_name, ext_id in missing_pairs
                )
            )

        insert_rows = []
        seen_pairs = set()
        for slick_id, aoi_type_short_name, ext_id in flattened_rows:
            pair = (
                slick_id,
                aoi_lookup[(aoi_type_ids[aoi_type_short_name], ext_id)],
            )
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            insert_rows.append({"slick": pair[0], "aoi": pair[1]})

        if not insert_rows:
            return 0

        insert_stmt = pg_insert(db.t_slick_to_aoi).values(insert_rows)
        insert_stmt = insert_stmt.on_conflict_do_nothing(
            index_elements=["slick", "aoi"]
        )
        await self.session.execute(insert_stmt)
        return len(insert_rows)

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
