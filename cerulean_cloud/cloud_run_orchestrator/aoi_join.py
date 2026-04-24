"""Utilities for orchestrator-side AOI joins."""

import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
)

import geopandas as gpd
import google.auth
import sqlalchemy as sa
from google.cloud import storage
from shapely import wkb
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


BoundsLike = Union[Sequence[float], BaseGeometry, gpd.GeoSeries, gpd.GeoDataFrame]
GCS_READONLY_SCOPE = ("https://www.googleapis.com/auth/devstorage.read_only",)
SUPPORTED_AOI_ACCESS_TYPES = {"GCS", "DB_LOCAL", "DB_REMOTE"}
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
REMOTE_AOI_ENGINE_CACHE: Dict[str, AsyncEngine] = {}


@dataclass(frozen=True)
class AOIAccessConfig:
    """Configuration for accessing a single AOI dataset."""

    key: str
    access_type: str = "GCS"
    fgb_uri: Optional[str] = None
    ext_id_field: Optional[str] = None
    name_field: Optional[str] = None
    pmtiles_uri: Optional[str] = None
    dataset_version: Optional[str] = None
    filter_toggle: Optional[bool] = None
    read_perm: Optional[int] = None
    table_name: Optional[str] = None
    geometry_column: Optional[str] = None
    ext_id_column: Optional[str] = None
    db_conn_str: Optional[str] = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "AOIAccessConfig":
        """
        Build an AOI access config from an `aoi_type`-style row.

        The authoritative field mapping lives in `aoi_type.properties`; direct
        keys are accepted for compatibility with existing DatabaseClient return
        values while that interface is being updated.
        """
        properties = row.get("properties") or {}
        if not isinstance(properties, Mapping):
            raise ValueError("AOI access config properties must be a mapping")

        key = row.get("key") or row.get("short_name")
        if not key:
            raise ValueError("AOI access config requires `key` or `short_name`")

        access_type = row.get("access_type", "GCS")
        if access_type not in SUPPORTED_AOI_ACCESS_TYPES:
            raise NotImplementedError(
                f"Unsupported AOI access_type={access_type!r} for AOI type {key!r}"
            )

        common_kwargs = {
            "key": str(key),
            "access_type": str(access_type),
            "filter_toggle": row.get("filter_toggle"),
            "read_perm": row.get("read_perm"),
        }
        name_field = _first_present(
            row,
            properties,
            "name_field",
            "display_name_field",
            "name_col",
            "name_column",
        )

        if access_type == "GCS":
            fgb_uri = _first_present(row, properties, "fgb_uri")
            ext_id_field = _first_present(row, properties, "ext_id_field", "ext_id_col")
            missing = [
                name
                for name, value in (
                    ("fgb_uri", fgb_uri),
                    ("ext_id_field", ext_id_field),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f"GCS AOI type {key!r} is missing required properties: "
                    + ", ".join(missing)
                )
            return cls(
                **common_kwargs,
                fgb_uri=str(fgb_uri),
                ext_id_field=str(ext_id_field),
                name_field=str(name_field) if name_field else None,
                pmtiles_uri=row.get("pmtiles_uri") or properties.get("pmt_uri"),
                dataset_version=row.get("dataset_version")
                or properties.get("dataset_version"),
            )

        table_name = row.get("table_name") or properties.get("table_name")
        geometry_column = _first_present(row, properties, "geometry_column", "geog_col")
        ext_id_column = _first_present(row, properties, "ext_id_column", "ext_id_col")
        db_conn_str = row.get("db_conn_str") or properties.get("db_conn_str")
        missing = [
            name
            for name, value in (
                ("table_name", table_name),
                ("geog_col", geometry_column),
                ("ext_id_col", ext_id_column),
            )
            if not value
        ]
        if access_type == "DB_REMOTE" and not db_conn_str:
            missing.append("db_conn_str")
        if missing:
            raise ValueError(
                f"{access_type} AOI type {key!r} is missing required properties: "
                + ", ".join(missing)
            )

        return cls(
            **common_kwargs,
            name_field=str(name_field) if name_field else None,
            pmtiles_uri=row.get("pmtiles_uri") or properties.get("pmt_uri"),
            dataset_version=row.get("dataset_version")
            or properties.get("dataset_version"),
            table_name=str(table_name),
            geometry_column=str(geometry_column),
            ext_id_column=str(ext_id_column),
            db_conn_str=db_conn_str,
        )


class AOIAccessor(Protocol):
    """Interface for AOI candidate loading by access pattern."""

    config: AOIAccessConfig

    async def load_candidates(self, scene_bounds: BoundsLike) -> gpd.GeoDataFrame:
        """Load AOI candidates intersecting the scene bounds."""

    async def candidates_for_scene(self, scene_bounds: BoundsLike) -> gpd.GeoDataFrame:
        """Load or return cached AOI candidates for the scene bounds."""

    async def compute_matches(
        self, slick_gdf: gpd.GeoDataFrame, scene_bounds: BoundsLike
    ) -> List[Dict[str, List[Dict[str, Any]]]]:
        """Return rich AOI match payloads for each slick."""


def _first_present(row: Mapping[str, Any], properties: Mapping[str, Any], *keys: str):
    for key in keys:
        value = row.get(key)
        if value:
            return value
        value = properties.get(key)
        if value:
            return value
    return None


def normalize_bounds(scene_bounds: BoundsLike) -> BaseGeometry:
    """Normalize supported bounds inputs into a shapely polygon."""
    if isinstance(scene_bounds, BaseGeometry):
        return scene_bounds
    if isinstance(scene_bounds, gpd.GeoDataFrame):
        return scene_bounds.union_all()
    if isinstance(scene_bounds, gpd.GeoSeries):
        return scene_bounds.union_all()
    if len(scene_bounds) != 4:
        raise ValueError("scene_bounds must be a geometry or a 4-value bounds tuple")
    minx, miny, maxx, maxy = scene_bounds
    return box(minx, miny, maxx, maxy)


def empty_candidate_gdf() -> gpd.GeoDataFrame:
    """Return an empty normalized AOI candidate GeoDataFrame."""
    return gpd.GeoDataFrame(
        {"ext_id": [], "name": [], "geometry": []},
        geometry="geometry",
        crs="EPSG:4326",
    )


def normalize_candidate_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    aoi_key: str,
    ext_id_field: str = "ext_id",
    name_field: Optional[str] = "name",
) -> gpd.GeoDataFrame:
    """Normalize AOI candidate columns and CRS for downstream matching."""
    if gdf.empty:
        return empty_candidate_gdf()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    rename_map = {ext_id_field: "ext_id"}
    if name_field and name_field in gdf.columns:
        rename_map[name_field] = "name"
    gdf = gdf.rename(columns=rename_map)

    if "ext_id" not in gdf.columns:
        raise ValueError(
            f"AOI dataset '{aoi_key}' is missing expected ext id field '{ext_id_field}'"
        )
    if "geometry" not in gdf.columns:
        raise ValueError(f"AOI dataset '{aoi_key}' is missing geometry")

    keep_cols = [col for col in ("ext_id", "name", "geometry") if col in gdf.columns]
    gdf = gdf[keep_cols].copy()
    gdf = gdf[gdf["geometry"].notna()]
    gdf = gdf[~gdf["geometry"].is_empty]
    gdf["ext_id"] = gdf["ext_id"].astype(str)
    if "name" not in gdf.columns:
        gdf["name"] = gdf["ext_id"]
    else:
        gdf["name"] = gdf["name"].fillna(gdf["ext_id"])
    return gdf


def normalize_slick_gdf(slick_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return slick geometries in EPSG:4326 with a dense integer index."""
    if slick_gdf.empty:
        return slick_gdf
    slicks = slick_gdf.copy()
    if slicks.crs is None:
        slicks = slicks.set_crs("EPSG:4326")
    else:
        slicks = slicks.to_crs("EPSG:4326")
    return slicks.reset_index(drop=True)


def compute_matches_for_candidates(
    aoi_key: str, slick_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame
) -> List[Dict[str, List[Dict[str, Any]]]]:
    """Compute rich AOI matches for one normalized AOI candidate layer."""
    slicks = normalize_slick_gdf(slick_gdf)
    if slicks.empty:
        return []

    results: List[Dict[str, List[Dict[str, Any]]]] = [
        {aoi_key: []} for _ in range(len(slicks))
    ]
    if aoi_gdf.empty:
        return results

    joined = gpd.sjoin(
        slicks[["geometry"]],
        aoi_gdf[["ext_id", "name", "geometry"]],
        how="left",
        predicate="intersects",
    )
    for slick_idx, group in joined.groupby(level=0):
        matched = group.dropna(subset=["ext_id", "index_right"])
        if matched.empty:
            continue

        matches: List[Dict[str, Any]] = []
        for ext_id, ext_id_group in matched.groupby("ext_id", sort=True):
            right_indices = ext_id_group["index_right"].dropna().tolist()
            geometries = aoi_gdf.loc[right_indices, "geometry"].tolist()
            names = ext_id_group["name"].dropna().tolist()
            matches.append(
                {
                    "ext_id": str(ext_id),
                    "name": str(names[0]) if names else str(ext_id),
                    "geometry": unary_union(geometries),
                }
            )
        results[int(slick_idx)][aoi_key] = matches

    return results


def quote_identifier(identifier: str) -> str:
    """Quote a single SQL identifier after validating a conservative pattern."""
    if not IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Unsafe SQL identifier in AOI config: {identifier!r}")
    return f'"{identifier}"'


def quote_table_name(table_name: str) -> str:
    """Quote a possibly schema-qualified table name."""
    return ".".join(quote_identifier(part) for part in table_name.split("."))


def db_url_for_asyncpg(db_conn_str: str) -> str:
    """Return a SQLAlchemy asyncpg URL for Postgres connection strings."""
    if db_conn_str.startswith("postgresql://"):
        return db_conn_str.replace("postgresql://", "postgresql+asyncpg://", 1)
    if db_conn_str.startswith("postgres://"):
        return db_conn_str.replace("postgres://", "postgresql+asyncpg://", 1)
    if db_conn_str.startswith("postgresql+psycopg2://"):
        return db_conn_str.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    return db_conn_str


def get_remote_aoi_engine(db_conn_str: str) -> AsyncEngine:
    """Return a cached async engine for remote AOI reads."""
    if db_conn_str not in REMOTE_AOI_ENGINE_CACHE:
        REMOTE_AOI_ENGINE_CACHE[db_conn_str] = create_async_engine(
            db_url_for_asyncpg(db_conn_str),
            echo=False,
            connect_args={"command_timeout": 60},
            pool_size=1,
            max_overflow=0,
            pool_timeout=300,
            pool_recycle=600,
        )
    return REMOTE_AOI_ENGINE_CACHE[db_conn_str]


def loads_ewkb(value):
    """Decode PostGIS ST_AsEWKB output into a shapely geometry."""
    if value is None:
        return None
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, str):
        return wkb.loads(value, hex=True)
    return wkb.loads(bytes(value))


class BaseAoiAccessor:
    """Shared AOI accessor behavior."""

    def __init__(self, config: AOIAccessConfig) -> None:
        self.config = config
        self._candidate_bbox: Optional[tuple] = None
        self._candidate_gdf: Optional[gpd.GeoDataFrame] = None

    async def load_candidates(self, scene_bounds: BoundsLike) -> gpd.GeoDataFrame:
        raise NotImplementedError

    async def candidates_for_scene(self, scene_bounds: BoundsLike) -> gpd.GeoDataFrame:
        scene_bbox = tuple(normalize_bounds(scene_bounds).bounds)
        if self._candidate_bbox == scene_bbox and self._candidate_gdf is not None:
            return self._candidate_gdf
        self._candidate_gdf = await self.load_candidates(scene_bounds)
        self._candidate_bbox = scene_bbox
        return self._candidate_gdf

    async def compute_matches(
        self, slick_gdf: gpd.GeoDataFrame, scene_bounds: BoundsLike
    ) -> List[Dict[str, List[Dict[str, Any]]]]:
        aoi_gdf = await self.candidates_for_scene(scene_bounds)
        return compute_matches_for_candidates(self.config.key, slick_gdf, aoi_gdf)


class GCSAoiAccessor(BaseAoiAccessor):
    """AOI accessor for FlatGeobuf assets stored locally or in GCS."""

    def __init__(self, config: AOIAccessConfig) -> None:
        super().__init__(config)
        if not config.fgb_uri or not config.ext_id_field:
            raise ValueError(
                f"GCS AOI type {config.key!r} requires fgb_uri and ext_id_field"
            )
        self.cache_dir = Path(tempfile.gettempdir()) / "cerulean_aoi_cache"
        self.gcp_project: Optional[str] = None

    def _get_gcs_credentials(self):
        """Resolve application default credentials for AOI downloads."""
        credentials, project = google.auth.default(scopes=GCS_READONLY_SCOPE)
        self.gcp_project = project
        return credentials

    def _download_aoi_dataset(self) -> str:
        """Resolve `gs://` AOI dataset paths into local cached files."""
        fgb_uri = self.config.fgb_uri
        if not fgb_uri.startswith("gs://"):
            return fgb_uri

        bucket_and_path = fgb_uri[len("gs://") :]
        bucket_name, _, object_name = bucket_and_path.partition("/")
        if not bucket_name or not object_name:
            raise ValueError(f"Invalid gs:// AOI dataset URL: {fgb_uri}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_digest = hashlib.sha256(
            f"{fgb_uri}|{self.config.dataset_version or ''}".encode("utf-8")
        ).hexdigest()[:12]
        local_name = f"{bucket_name}__{object_name.replace('/', '__')}__{cache_digest}"
        local_path = self.cache_dir / local_name
        if local_path.exists() and local_path.stat().st_size > 0:
            return str(local_path)

        credentials = self._get_gcs_credentials()
        client = storage.Client(project=self.gcp_project, credentials=credentials)
        client.bucket(bucket_name).blob(object_name).download_to_filename(local_path)
        return str(local_path)

    async def load_candidates(self, scene_bounds: BoundsLike) -> gpd.GeoDataFrame:
        scene_bbox = tuple(normalize_bounds(scene_bounds).bounds)
        gdf = gpd.read_file(self._download_aoi_dataset(), bbox=scene_bbox)
        return normalize_candidate_gdf(
            gdf,
            aoi_key=self.config.key,
            ext_id_field=self.config.ext_id_field,
            name_field=self.config.name_field,
        )


class DbBaseAoiAccessor(BaseAoiAccessor):
    """Base AOI accessor for PostGIS-backed AOI sources."""

    def __init__(self, config: AOIAccessConfig) -> None:
        super().__init__(config)
        missing = [
            name
            for name, value in (
                ("table_name", config.table_name),
                ("geog_col", config.geometry_column),
                ("ext_id_col", config.ext_id_column),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"{config.access_type} AOI type {config.key!r} is missing required "
                f"properties: {', '.join(missing)}"
            )

    def _candidate_sql(self) -> str:
        table_name = quote_table_name(self.config.table_name)
        geometry_column = quote_identifier(self.config.geometry_column)
        ext_id_column = quote_identifier(self.config.ext_id_column)
        name_expr = (
            f"{quote_identifier(self.config.name_field)}::text"
            if self.config.name_field
            else f"{ext_id_column}::text"
        )
        return f"""
            WITH candidates AS (
                SELECT
                    {ext_id_column}::text AS ext_id,
                    {name_expr} AS name,
                    {geometry_column}::geometry AS geom
                FROM {table_name}
                WHERE {ext_id_column} IS NOT NULL
            ),
            normalized AS (
                SELECT
                    ext_id,
                    COALESCE(name, ext_id) AS name,
                    CASE
                        WHEN ST_SRID(geom) = 0 THEN ST_SetSRID(geom, 4326)
                        WHEN ST_SRID(geom) = 4326 THEN geom
                        ELSE ST_Transform(geom, 4326)
                    END AS geom
                FROM candidates
                WHERE geom IS NOT NULL
                  AND NOT ST_IsEmpty(geom)
            )
            SELECT
                ext_id,
                name,
                ST_AsEWKB(geom) AS geometry
            FROM normalized
            WHERE ST_Intersects(
                geom,
                ST_MakeEnvelope(:minx, :miny, :maxx, :maxy, 4326)
            )
        """

    async def _load_candidates_from_engine(
        self, engine: AsyncEngine, scene_bounds: BoundsLike
    ) -> gpd.GeoDataFrame:
        minx, miny, maxx, maxy = tuple(normalize_bounds(scene_bounds).bounds)
        async with engine.connect() as conn:
            result = await conn.execute(
                sa.text(self._candidate_sql()),
                {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy},
            )
            rows = result.mappings().all()

        if not rows:
            return empty_candidate_gdf()

        records = [
            {
                "ext_id": str(row["ext_id"]),
                "name": row["name"] or str(row["ext_id"]),
                "geometry": loads_ewkb(row["geometry"]),
            }
            for row in rows
        ]
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        return normalize_candidate_gdf(gdf, aoi_key=self.config.key)


class DbLocalAoiAccessor(DbBaseAoiAccessor):
    """AOI accessor for local PostGIS tables or views."""

    def __init__(self, config: AOIAccessConfig, local_engine: AsyncEngine) -> None:
        super().__init__(config)
        if local_engine is None:
            raise ValueError(f"DB_LOCAL AOI type {config.key!r} requires local_engine")
        self.local_engine = local_engine

    async def load_candidates(self, scene_bounds: BoundsLike) -> gpd.GeoDataFrame:
        return await self._load_candidates_from_engine(self.local_engine, scene_bounds)


class DbRemoteAoiAccessor(DbBaseAoiAccessor):
    """AOI accessor for remote PostGIS tables or views."""

    def __init__(self, config: AOIAccessConfig) -> None:
        super().__init__(config)
        if not config.db_conn_str:
            raise ValueError(f"DB_REMOTE AOI type {config.key!r} requires db_conn_str")

    async def load_candidates(self, scene_bounds: BoundsLike) -> gpd.GeoDataFrame:
        engine = get_remote_aoi_engine(self.config.db_conn_str)
        return await self._load_candidates_from_engine(engine, scene_bounds)


def build_aoi_accessor(
    config: Union[AOIAccessConfig, Mapping[str, Any]],
    *,
    local_engine: Optional[AsyncEngine] = None,
) -> AOIAccessor:
    """Build an AOI accessor from normalized AOI access config."""
    access_config = (
        config
        if isinstance(config, AOIAccessConfig)
        else AOIAccessConfig.from_mapping(config)
    )
    if access_config.access_type == "GCS":
        return GCSAoiAccessor(access_config)
    if access_config.access_type == "DB_LOCAL":
        return DbLocalAoiAccessor(access_config, local_engine)
    if access_config.access_type == "DB_REMOTE":
        return DbRemoteAoiAccessor(access_config)
    raise NotImplementedError(
        f"Unsupported AOI access_type={access_config.access_type!r} "
        f"for AOI type {access_config.key!r}"
    )


class AOIJoiner:
    """Coordinate AOI accessors and merge their per-slick match payloads."""

    def __init__(
        self,
        scene_bounds: BoundsLike,
        aoi_access_configs: Optional[
            Iterable[Union[AOIAccessConfig, Mapping[str, Any]]]
        ] = None,
        *,
        accessors: Optional[Iterable[AOIAccessor]] = None,
        local_engine: Optional[AsyncEngine] = None,
    ) -> None:
        self.scene_bounds = normalize_bounds(scene_bounds)
        if accessors is not None:
            self.accessors = tuple(accessors)
        else:
            self.accessors = tuple(
                build_aoi_accessor(access_config, local_engine=local_engine)
                for access_config in aoi_access_configs or []
            )
        if not self.accessors:
            raise ValueError("AOIJoiner requires at least one AOI accessor")

    async def compute_aoi_matches(
        self, slick_gdf: gpd.GeoDataFrame
    ) -> List[Dict[str, List[Dict[str, Any]]]]:
        """Return rich AOI matches for each slick across all accessors."""
        slicks = normalize_slick_gdf(slick_gdf)
        if slicks.empty:
            return []

        results: List[Dict[str, List[Dict[str, Any]]]] = [
            {accessor.config.key: [] for accessor in self.accessors}
            for _ in range(len(slicks))
        ]
        for accessor in self.accessors:
            accessor_results = await accessor.compute_matches(slicks, self.scene_bounds)
            for idx, accessor_result in enumerate(accessor_results):
                results[idx].update(accessor_result)
        return results

    async def compute_aoi_intersect(
        self, slick_gdf: gpd.GeoDataFrame
    ) -> List[Dict[str, List[str]]]:
        """Return AOI external IDs per slick, shaped like the legacy `aoi_ext_ids`."""
        return [
            {
                aoi_type: [match["ext_id"] for match in matches]
                for aoi_type, matches in slick_matches.items()
            }
            for slick_matches in await self.compute_aoi_matches(slick_gdf)
        ]

    async def compute_single_slick_intersect(
        self, slick_gdf: gpd.GeoDataFrame
    ) -> Dict[str, List[str]]:
        """Convenience wrapper for the common single-slick case."""
        if len(slick_gdf) != 1:
            raise ValueError(
                "compute_single_slick_intersect expects exactly one slick row"
            )
        return (await self.compute_aoi_intersect(slick_gdf))[0]

    async def compute_single_slick_matches(
        self, slick_gdf: gpd.GeoDataFrame
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Convenience wrapper for single-slick AOI match metadata."""
        if len(slick_gdf) != 1:
            raise ValueError(
                "compute_single_slick_matches expects exactly one slick row"
            )
        return (await self.compute_aoi_matches(slick_gdf))[0]

    async def get_aoi_gdf(self, aoi_type: str) -> gpd.GeoDataFrame:
        """Return the cached candidate GeoDataFrame for a single AOI type."""
        for accessor in self.accessors:
            if accessor.config.key == aoi_type:
                return await accessor.candidates_for_scene(self.scene_bounds)
        raise KeyError(aoi_type)

    async def as_dict(self) -> Mapping[str, gpd.GeoDataFrame]:
        """Expose cached candidate GeoDataFrames."""
        return {
            accessor.config.key: await accessor.candidates_for_scene(self.scene_bounds)
            for accessor in self.accessors
        }
