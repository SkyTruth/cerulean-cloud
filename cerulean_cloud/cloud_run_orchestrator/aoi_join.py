"""Utilities for orchestrator-side AOI joins."""

import hashlib
import re
import tempfile
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
)

import geopandas as gpd
import google.auth
import sqlalchemy as sa
from google.cloud import storage
from shapely import wkb
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


GCS_READONLY_SCOPE = ("https://www.googleapis.com/auth/devstorage.read_only",)
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def empty_candidate_gdf() -> gpd.GeoDataFrame:
    """Return an empty normalized AOI candidate GeoDataFrame."""
    return gpd.GeoDataFrame(
        {"ext_id": [], "name": [], "geometry": []},
        geometry="geometry",
        crs="EPSG:4326",
    )


def quote_identifier(identifier: str) -> str:
    """Quote a single SQL identifier after validating a conservative pattern."""
    if not IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Unsafe SQL identifier in AOI config: {identifier!r}")
    return f'"{identifier}"'


def quote_table_name(table_name: str) -> str:
    """Quote a possibly schema-qualified table name."""
    return ".".join(quote_identifier(part) for part in table_name.split("."))


class BaseAoiAccessor:
    """
    Shared AOI accessor behavior.

    Concrete accessors normalize candidates into EPSG:4326 GeoDataFrames with
    `ext_id`, `name`, and `geometry`. `ext_id` and `geometry` are non-null,
    geometry is non-empty, and `name` falls back to `ext_id`.
    """

    def __init__(self, row: Mapping[str, Any]) -> None:
        self.short_name = row["short_name"]
        self.properties = row.get("properties") or {}
        self.dataset_version = self.properties.get("dataset_version")
        self._candidate_bbox: Optional[tuple] = None
        self._candidate_gdf: Optional[gpd.GeoDataFrame] = None

    async def load_candidates(self, scene_bounds: Sequence[float]) -> gpd.GeoDataFrame:
        raise NotImplementedError

    async def candidates_for_scene(
        self, scene_bounds: Sequence[float]
    ) -> gpd.GeoDataFrame:
        scene_bbox = tuple(scene_bounds)
        if self._candidate_bbox == scene_bbox and self._candidate_gdf is not None:
            return self._candidate_gdf
        self._candidate_gdf = await self.load_candidates(scene_bounds)
        self._candidate_bbox = scene_bbox
        return self._candidate_gdf

    async def matches_for_scene(
        self, scene_bounds: Sequence[float], slick_gdf: gpd.GeoDataFrame
    ) -> List[Dict[str, List[Dict[str, str]]]]:
        if slick_gdf.empty:
            return []

        aoi_gdf = await self.candidates_for_scene(scene_bounds)
        results: List[Dict[str, List[Dict[str, str]]]] = [
            {self.short_name: []} for _ in range(len(slick_gdf))
        ]
        if aoi_gdf.empty:
            return results

        joined = gpd.sjoin(
            slick_gdf[["geometry"]],
            aoi_gdf[["ext_id", "name", "geometry"]],
            how="left",
            predicate="intersects",
        )
        for slick_idx, group in joined.groupby(level=0):
            matched = group.dropna(subset=["ext_id", "index_right"])
            if matched.empty:
                continue

            matches: List[Dict[str, str]] = []
            for ext_id, ext_id_group in matched.groupby("ext_id", sort=True):
                names = ext_id_group["name"].dropna().tolist()
                matches.append(
                    {
                        "ext_id": str(ext_id),
                        "name": str(names[0]) if names else str(ext_id),
                    }
                )
            results[int(slick_idx)][self.short_name] = matches

        return results


class GCSAoiAccessor(BaseAoiAccessor):
    """AOI accessor for FlatGeobuf assets stored locally or in GCS."""

    def __init__(self, row: Mapping[str, Any]) -> None:
        super().__init__(row)
        self.fgb_uri = self.properties["fgb_uri"]
        self.ext_id_field = self.properties["ext_id_field"]
        self.display_name_field = self.properties.get("display_name_field")
        self.cache_dir = Path(tempfile.gettempdir()) / "cerulean_aoi_cache"
        self.gcp_project: Optional[str] = None

    def _get_gcs_credentials(self):
        """Resolve application default credentials for AOI downloads."""
        credentials, project = google.auth.default(scopes=GCS_READONLY_SCOPE)
        self.gcp_project = project
        return credentials

    def _download_aoi_dataset(self) -> str:
        """Resolve `gs://` AOI dataset paths into local cached files."""
        bucket_and_path = self.fgb_uri[len("gs://") :]
        bucket_name, _, object_name = bucket_and_path.partition("/")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_digest = hashlib.sha256(
            f"{self.fgb_uri}|{self.dataset_version or ''}".encode("utf-8")
        ).hexdigest()[:12]
        local_name = f"{bucket_name}__{object_name.replace('/', '__')}__{cache_digest}"
        local_path = self.cache_dir / local_name
        if local_path.exists() and local_path.stat().st_size > 0:
            return str(local_path)

        credentials = self._get_gcs_credentials()
        client = storage.Client(project=self.gcp_project, credentials=credentials)
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=self.cache_dir,
            prefix=f"{local_name}.",
            suffix=".tmp",
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            client.bucket(bucket_name).blob(object_name).download_to_filename(tmp_path)
            if tmp_path.stat().st_size <= 0:
                raise ValueError(f"Downloaded empty AOI dataset: {self.fgb_uri}")
            tmp_path.replace(local_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return str(local_path)

    async def load_candidates(self, scene_bounds: Sequence[float]) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(self._download_aoi_dataset(), bbox=tuple(scene_bounds))
        if gdf.empty:
            return empty_candidate_gdf()

        gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
        rename_map = {self.ext_id_field: "ext_id"}
        if self.display_name_field:
            rename_map[self.display_name_field] = "name"
        gdf = gdf.rename(columns=rename_map)
        if "name" not in gdf.columns:
            gdf["name"] = gdf["ext_id"]

        gdf = gdf[["ext_id", "name", "geometry"]].copy()
        gdf = gdf[gdf["geometry"].notna()]
        gdf = gdf[~gdf["geometry"].is_empty]
        gdf = gdf[gdf["ext_id"].notna()]
        if gdf.empty:
            return empty_candidate_gdf()
        gdf["ext_id"] = gdf["ext_id"].astype(str)
        gdf["name"] = gdf["name"].fillna(gdf["ext_id"])
        return gdf


class DbBaseAoiAccessor(BaseAoiAccessor):
    """Base AOI accessor for PostGIS-backed AOI sources."""

    def __init__(self, row: Mapping[str, Any]) -> None:
        super().__init__(row)
        self.table_name = self.properties["table_name"]
        self.geog_col = self.properties["geog_col"]
        self.ext_id_col = self.properties["ext_id_col"]
        self.display_name_field = self.properties.get("display_name_field")

    @staticmethod
    def _loads_ewkb(value):
        if value is None:
            return None
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, str):
            return wkb.loads(value, hex=True)
        return wkb.loads(bytes(value))

    def _candidate_sql(self) -> str:
        table_name = quote_table_name(self.table_name)
        geog_col = quote_identifier(self.geog_col)
        ext_id_col = quote_identifier(self.ext_id_col)
        name_expr = (
            f"{quote_identifier(self.display_name_field)}::text"
            if self.display_name_field
            else f"{ext_id_col}::text"
        )
        return f"""
            WITH candidates AS (
                SELECT
                    {ext_id_col}::text AS ext_id,
                    {name_expr} AS name,
                    {geog_col}::geometry AS geom
                FROM {table_name}
                WHERE {ext_id_col} IS NOT NULL
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
        self, engine: AsyncEngine, scene_bounds: Sequence[float]
    ) -> gpd.GeoDataFrame:
        minx, miny, maxx, maxy = scene_bounds
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
                "geometry": self._loads_ewkb(row["geometry"]),
            }
            for row in rows
        ]
        return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


class DbLocalAoiAccessor(DbBaseAoiAccessor):
    """AOI accessor for local PostGIS tables or views."""

    def __init__(self, row: Mapping[str, Any], local_engine: AsyncEngine) -> None:
        super().__init__(row)
        self.local_engine = local_engine

    async def load_candidates(self, scene_bounds: Sequence[float]) -> gpd.GeoDataFrame:
        return await self._load_candidates_from_engine(self.local_engine, scene_bounds)


class DbRemoteAoiAccessor(DbBaseAoiAccessor):
    """AOI accessor for remote PostGIS tables or views."""

    _engine_cache: Dict[str, AsyncEngine] = {}

    def __init__(self, row: Mapping[str, Any]) -> None:
        super().__init__(row)
        self.db_conn_str = self.properties["db_conn_str"]

    def _engine(self) -> AsyncEngine:
        db_url = self.db_conn_str
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif db_url.startswith("postgresql+psycopg2://"):
            db_url = db_url.replace(
                "postgresql+psycopg2://", "postgresql+asyncpg://", 1
            )
        if self.db_conn_str not in self._engine_cache:
            self._engine_cache[self.db_conn_str] = create_async_engine(
                db_url,
                echo=False,
                connect_args={"command_timeout": 60},
                pool_size=1,
                max_overflow=0,
                pool_timeout=300,
                pool_recycle=600,
            )
        return self._engine_cache[self.db_conn_str]

    async def load_candidates(self, scene_bounds: Sequence[float]) -> gpd.GeoDataFrame:
        return await self._load_candidates_from_engine(self._engine(), scene_bounds)


def build_aoi_accessor(
    row: Mapping[str, Any],
    *,
    local_engine: Optional[AsyncEngine] = None,
) -> BaseAoiAccessor:
    """Build an AOI accessor from an `aoi_type` access row."""
    access_type = row["access_type"]
    if access_type == "GCS":
        return GCSAoiAccessor(row)
    if access_type == "DB_LOCAL":
        return DbLocalAoiAccessor(row, local_engine)
    if access_type == "DB_REMOTE":
        return DbRemoteAoiAccessor(row)
    raise NotImplementedError(
        f"Unsupported AOI access_type={access_type!r} "
        f"for AOI type {row['short_name']!r}"
    )


class AOIJoiner:
    """Coordinate AOI accessors and merge their per-slick match payloads."""

    def __init__(
        self,
        scene_bounds: Sequence[float],
        accessors: Iterable[BaseAoiAccessor],
    ) -> None:
        self.scene_bounds = tuple(scene_bounds)
        self.accessors = tuple(accessors)

    async def compute_aoi_matches(
        self, slick_gdf: gpd.GeoDataFrame
    ) -> List[Dict[str, List[Dict[str, str]]]]:
        """
        Return compact AOI matches for each slick across all accessors.

        Caller provides EPSG:4326 slick geometries with a dense default index.
        """
        if slick_gdf.empty:
            return []

        results: List[Dict[str, List[Dict[str, str]]]] = [
            {accessor.short_name: [] for accessor in self.accessors}
            for _ in range(len(slick_gdf))
        ]
        for accessor in self.accessors:
            accessor_results = await accessor.matches_for_scene(
                self.scene_bounds, slick_gdf
            )
            for idx, accessor_result in enumerate(accessor_results):
                results[idx].update(accessor_result)
        return results
