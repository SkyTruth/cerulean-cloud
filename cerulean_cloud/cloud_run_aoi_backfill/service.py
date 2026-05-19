"""Operator-oriented shared-dataset AOI backfill workflow."""

from __future__ import annotations

import hashlib
import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir


LOGGER = logging.getLogger("backfill_shared_dataset_aoi")
# Notebook imports do not run the CLI's basicConfig(), so set the module logger's
# threshold explicitly to keep INFO logs visible in interactive runs.
LOGGER.setLevel(logging.INFO)
REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_SCRIPT = REPO_ROOT / "scripts/backfill_shared_dataset_aoi.sql"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

EXT_ID_FIELD_CANDIDATES = (
    "ext_id",
    "external_id",
    "mrgid",
    "primkey",
    "site_id",
    "wdpaid",
    "objectid",
    "fid",
    "id",
)
NAME_FIELD_CANDIDATES = (
    "name",
    "display_name",
    "title",
    "label",
    "geoname",
    "site_name",
)
DEFAULT_CACHE_DIR = str(Path(gettempdir()) / "cerulean_aoi_backfill_cache")
LARGE_DATASET_WARN_BYTES = 250 * 1024 * 1024
TARGET_CHUNK_BYTES = 64 * 1024 * 1024
TARGET_CHUNK_FEATURES = 10000
DEFAULT_RUN_MAX_BATCHES = 5
DEFAULT_RUN_STATEMENT_TIMEOUT = "4min"
DEFAULT_MAX_CHUNK_STAGE_ROWS = 20000
DEFAULT_SPLIT_CANDIDATE_SLICKS = 50000
DEFAULT_MAX_SPLIT_DEPTH = 6
DEFAULT_STALE_RETENTION = "7 days"
MAX_CHUNK_RETRIES = 3


@dataclass(frozen=True)
class AoiConfig:
    asset_slug: str
    short_name: str
    long_name: str
    ext_id_field: str
    display_name_field: str | None
    stage_table: str
    dataset_version: str
    source_url: str
    citation: str
    slick_to_aoi_buffer_m: float = 0.0


@dataclass(frozen=True)
class ChunkSpec:
    chunk_index: int
    minx: float
    miny: float
    maxx: float
    maxy: float
    split_depth: int = 0
    parent_chunk_id: int | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)


@dataclass(frozen=True)
class RunContext:
    asset_slug: str
    short_name: str
    dataset_version: str
    stage_table: str
    ext_id_field: str
    display_name_field: str | None
    batch_size: int
    slick_to_aoi_buffer_m: float


@dataclass(frozen=True)
class LocalCatalogAsset:
    slug: str
    title: str
    canonical_path: str
    available_formats: tuple[str, ...]
    citation: str = ""

    def path_for_format(self, fmt: str) -> str:
        if fmt == "fgb":
            return self.canonical_path
        return ""


def load_catalog(catalog_source: str | None):
    try:
        from skytruth_shared_datasets import Catalog, DEFAULT_CATALOG_GS_URI
    except ModuleNotFoundError:
        if not catalog_source or catalog_source.startswith("gs://"):
            raise
        with open(catalog_source, newline="") as src:
            reader = csv.DictReader(src)
            return [
                LocalCatalogAsset(
                    slug=row["asset_slug"],
                    title=row.get("title") or row["asset_slug"],
                    canonical_path=row.get("canonical_path") or "",
                    available_formats=tuple(
                        item
                        for item in (row.get("available_formats") or "").split(";")
                        if item
                    ),
                    citation=row.get("citation") or "",
                )
                for row in reader
            ]

    source = catalog_source or DEFAULT_CATALOG_GS_URI
    if source.startswith("gs://"):
        return Catalog.load_gcs(source)
    return Catalog.load(source)


def get_catalog_asset(asset_slug: str, catalog_source: str | None = None):
    for asset in load_catalog(catalog_source):
        if asset.slug == asset_slug:
            return asset
    raise ValueError(f"Could not find asset slug in catalog: {asset_slug!r}")


def resolve_asset_slug(asset_selector: str, catalog_source: str | None = None) -> str:
    if not asset_selector.startswith("gs://"):
        return asset_selector

    matches = [
        asset.slug
        for asset in load_catalog(catalog_source)
        if asset.canonical_path == asset_selector
        or any(
            asset.path_for_format(fmt) == asset_selector
            for fmt in asset.available_formats
        )
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "Could not resolve shared-datasets GCS URI to an asset slug: "
            f"{asset_selector}"
        )
    raise ValueError(
        "Shared-datasets GCS URI matched multiple asset slugs: " + ", ".join(matches)
    )


def normalize_db_url(db_url: str) -> str:
    if db_url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + db_url[len("postgresql+asyncpg://") :]
    if db_url.startswith("postgresql+psycopg2://"):
        return "postgresql://" + db_url[len("postgresql+psycopg2://") :]
    return db_url


def slug_to_short_name(slug: str) -> str:
    short_name = re.sub(r"[^A-Za-z0-9]+", "_", slug).strip("_").upper()
    if short_name and short_name[0].isdigit():
        short_name = f"AOI_{short_name}"
    if not short_name or not IDENTIFIER_RE.fullmatch(short_name):
        raise ValueError(
            f"Cannot derive a safe AOI short_name from asset slug {slug!r}"
        )
    return short_name


def slug_to_stage_table(short_name: str) -> str:
    table = f"aoi_stage_{short_name.lower()}"
    if len(table) > 63:
        digest = hashlib.sha1(short_name.encode("utf-8")).hexdigest()[:10]
        table = f"aoi_stage_{short_name.lower()[:42]}_{digest}"
    return f"maintenance.{table}"


def resolve_existing_shared_dataset_short_name(
    db_url: str, asset_slug: str
) -> str | None:
    row = query_one_row(
        db_url,
        """
        SELECT short_name
        FROM public.aoi_type
        WHERE access_type = 'SHARED_DATASET'
          AND properties->>'asset_slug' = %s
        ORDER BY id
        LIMIT 1
        """,
        (asset_slug,),
    )
    return row[0] if row is not None else None


def _field_lookup(columns: Sequence[str]) -> dict[str, str]:
    return {column.lower(): column for column in columns}


def infer_ext_id_field(columns: Sequence[str]) -> str:
    lookup = _field_lookup(columns)
    if "ext_id" in lookup:
        return lookup["ext_id"]

    matches = [
        lookup[candidate]
        for candidate in EXT_ID_FIELD_CANDIDATES
        if candidate in lookup and candidate != "ext_id"
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "Could not infer ext_id field. Re-run with --ext-id-field. "
            f"Available fields: {', '.join(columns)}"
        )
    raise ValueError(
        "Multiple plausible ext_id fields found. Re-run with --ext-id-field. "
        f"Candidates: {', '.join(matches)}"
    )


def infer_display_name_field(columns: Sequence[str], ext_id_field: str) -> str | None:
    lookup = _field_lookup(column for column in columns if column != ext_id_field)
    for candidate in NAME_FIELD_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    return None


def parse_table_name(table_name: str) -> tuple[str, str]:
    parts = table_name.split(".")
    if len(parts) != 2:
        raise ValueError(
            "Stage table must be schema-qualified, e.g. maintenance.aoi_stage_example"
        )
    schema, table = parts
    for part in parts:
        if not IDENTIFIER_RE.fullmatch(part):
            raise ValueError(
                f"Unsafe SQL identifier in stage table name: {table_name!r}"
            )
        if len(part) > 63:
            raise ValueError(
                f"SQL identifier is too long in stage table name: {table_name!r}"
            )
    if schema == "public":
        raise ValueError("Stage table must not live in public")
    return schema, table


def quote_identifier(identifier: str) -> str:
    if not IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def open_connection(db_url: str, *, autocommit: bool = True):
    import psycopg2

    conn = psycopg2.connect(normalize_db_url(db_url))
    conn.autocommit = autocommit
    return conn


def fetch_dataset_ref(
    asset_slug: str,
    *,
    version: str,
    cache_dir: Path,
    force: bool,
    catalog_source: str | None,
):
    from skytruth_shared_datasets import fetch_dataset

    if catalog_source:
        catalog = load_catalog(catalog_source)
        ref = catalog.fetch(
            asset_slug,
            "fgb",
            version=version,
            cache_dir=cache_dir,
            force=force,
            access="public",
        )
    else:
        ref = fetch_dataset(
            asset_slug,
            "fgb",
            version=version,
            cache_dir=cache_dir,
            force=force,
        )
    if ref.cache_path is None:
        raise RuntimeError(f"Shared dataset {asset_slug!r} did not return a cache path")
    return ref


def derive_catalog_citation(asset) -> str:
    citation = getattr(asset, "citation", None)
    return citation or ""


def derive_catalog_source_url(asset, ref) -> str:
    return getattr(ref, "url", None) or getattr(asset, "canonical_path", None) or ""


def normalize_dataset_version(
    asset_slug: str, resolved_id: object, version: str
) -> str:
    raw_version = str(resolved_id or version or "")
    if raw_version == f"{asset_slug}@":
        return ""
    return raw_version


def shared_dataset_fetch_version(asset_slug: str, dataset_version: str | None) -> str:
    if not dataset_version:
        return "latest"
    if dataset_version == f"{asset_slug}@":
        return "latest"
    prefix = f"{asset_slug}@"
    if dataset_version.startswith(prefix):
        suffix = dataset_version[len(prefix) :]
        return suffix or "latest"
    return dataset_version


def inspect_dataset_size(path: Path) -> dict[str, object]:
    size_bytes = path.stat().st_size
    result = {
        "dataset_size_bytes": size_bytes,
        "dataset_size_mb": round(size_bytes / (1024 * 1024), 1),
    }
    if size_bytes > LARGE_DATASET_WARN_BYTES:
        result["dataset_size_warning"] = (
            "Dataset is large enough that chunked AOI staging will be used "
            f"({result['dataset_size_mb']} MB)"
        )
    return result


def inspect_fields(path: Path) -> list[str]:
    import geopandas as gpd

    sample = gpd.read_file(path, rows=1)
    return [column for column in sample.columns if column != sample.geometry.name]


def _dataset_metadata(path: Path) -> dict[str, object]:
    try:
        import pyogrio

        info = pyogrio.read_info(path)
        return {
            "feature_count": int(info["features"]),
            "bounds": tuple(float(value) for value in info["total_bounds"]),
            "crs": str(info.get("crs") or ""),
        }
    except ImportError:
        import fiona

        with fiona.open(path) as src:
            bounds = tuple(float(value) for value in src.bounds)
            feature_count = len(src)
            crs = src.crs_wkt or src.crs
        return {
            "feature_count": int(feature_count),
            "bounds": bounds,
            "crs": str(crs),
        }


def inspect_stage_readiness(path: Path, ext_id_field: str) -> dict[str, object]:
    import geopandas as gpd

    gdf = gpd.read_file(path, columns=[ext_id_field])
    ext_ids = normalize_stage_text(gdf[ext_id_field]).fillna("")
    duplicated = ext_ids[ext_ids.duplicated(keep=False)]
    return {
        "feature_count": len(gdf),
        "crs": str(gdf.crs),
        "geometry_types": dict(gdf.geometry.geom_type.value_counts()),
        "null_or_empty_geometry_rows": int(
            (gdf.geometry.isna() | gdf.geometry.is_empty).sum()
        ),
        "invalid_geometry_rows": int((~gdf.geometry.is_valid).sum()),
        "empty_ext_id_rows": int((ext_ids.str.len() == 0).sum()),
        "duplicate_ext_id_values": int(duplicated.nunique()),
        "duplicate_ext_id_rows": int(len(duplicated)),
    }


def to_plain_python(value):
    if isinstance(value, Mapping):
        return {key: to_plain_python(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_python(inner_value) for inner_value in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except (TypeError, ValueError):
            pass
    return value


def promote_polygons(geometry):
    from shapely.geometry import MultiPolygon, Polygon

    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, MultiPolygon):
        return geometry
    if isinstance(geometry, Polygon):
        return MultiPolygon([geometry])
    return None


def sanitize_stage_text(series):
    return series.astype("string").str.replace("\x00", "", regex=False)


def normalize_stage_text(series):
    def stringify_stage_value(value):
        if value is None:
            return None
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)

    return sanitize_stage_text(series.map(stringify_stage_value))


def normalize_stage_gdf(gdf, config: AoiConfig):
    import geopandas as gpd

    if gdf.empty:
        return gpd.GeoDataFrame(columns=["ext_id", "name", "geom"], geometry="geom")

    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    gdf = gdf.rename(columns={config.ext_id_field: "ext_id"})
    gdf["ext_id"] = normalize_stage_text(gdf["ext_id"])
    if config.display_name_field:
        if config.display_name_field == config.ext_id_field:
            gdf["name"] = gdf["ext_id"]
        else:
            gdf = gdf.rename(columns={config.display_name_field: "name"})
            gdf["name"] = sanitize_stage_text(gdf["name"])
    else:
        gdf["name"] = gdf["ext_id"]
    gdf["name"] = gdf["name"].fillna(gdf["ext_id"])
    gdf["geom"] = gdf.geometry.map(promote_polygons)
    stage_gdf = gdf[["ext_id", "name", "geom"]].dropna(subset=["ext_id", "geom"])
    return gpd.GeoDataFrame(stage_gdf, geometry="geom", crs="EPSG:4326")


def _uniform_chunk_grid(
    bounds: tuple[float, float, float, float], grid_side: int
) -> list[ChunkSpec]:
    minx, miny, maxx, maxy = bounds
    if minx == maxx or miny == maxy:
        return [ChunkSpec(chunk_index=1, minx=minx, miny=miny, maxx=maxx, maxy=maxy)]

    width = (maxx - minx) / grid_side
    height = (maxy - miny) / grid_side
    chunks = []
    chunk_index = 1
    for row in range(grid_side):
        for col in range(grid_side):
            cell_minx = minx + col * width
            cell_maxx = maxx if col == grid_side - 1 else minx + (col + 1) * width
            cell_miny = miny + row * height
            cell_maxy = maxy if row == grid_side - 1 else miny + (row + 1) * height
            chunks.append(
                ChunkSpec(
                    chunk_index=chunk_index,
                    minx=cell_minx,
                    miny=cell_miny,
                    maxx=cell_maxx,
                    maxy=cell_maxy,
                )
            )
            chunk_index += 1
    return chunks


def build_chunk_plan(path: Path) -> dict[str, object]:
    metadata = _dataset_metadata(path)
    feature_count = int(metadata["feature_count"])
    bounds = metadata["bounds"]
    size_bytes = path.stat().st_size
    target_chunks = max(
        1,
        math.ceil(
            max(
                size_bytes / TARGET_CHUNK_BYTES,
                feature_count / TARGET_CHUNK_FEATURES if feature_count else 1,
            )
        ),
    )
    grid_side = max(1, math.ceil(math.sqrt(target_chunks)))
    chunks = _uniform_chunk_grid(bounds, grid_side)
    return {
        "bounds": bounds,
        "feature_count": feature_count,
        "grid_side": grid_side,
        "target_chunk_count": len(chunks),
        "target_chunk_bytes": TARGET_CHUNK_BYTES,
        "target_chunk_features": TARGET_CHUNK_FEATURES,
        "max_chunk_stage_rows": DEFAULT_MAX_CHUNK_STAGE_ROWS,
        "split_candidate_slick_limit": DEFAULT_SPLIT_CANDIDATE_SLICKS,
        "max_split_depth": DEFAULT_MAX_SPLIT_DEPTH,
        "chunks": chunks,
    }


def split_chunk_bbox(chunk: ChunkSpec) -> list[ChunkSpec]:
    midx = (chunk.minx + chunk.maxx) / 2
    midy = (chunk.miny + chunk.maxy) / 2
    if (
        midx == chunk.minx
        or midx == chunk.maxx
        or midy == chunk.miny
        or midy == chunk.maxy
    ):
        return []

    next_depth = chunk.split_depth + 1
    return [
        ChunkSpec(
            0, chunk.minx, chunk.miny, midx, midy, next_depth, chunk.parent_chunk_id
        ),
        ChunkSpec(
            0, midx, chunk.miny, chunk.maxx, midy, next_depth, chunk.parent_chunk_id
        ),
        ChunkSpec(
            0, chunk.minx, midy, midx, chunk.maxy, next_depth, chunk.parent_chunk_id
        ),
        ChunkSpec(
            0, midx, midy, chunk.maxx, chunk.maxy, next_depth, chunk.parent_chunk_id
        ),
    ]


def load_chunk_gdf(
    config: AoiConfig, path: Path, bbox: tuple[float, float, float, float]
):
    import geopandas as gpd

    read_columns = [config.ext_id_field]
    if config.display_name_field and config.display_name_field != config.ext_id_field:
        read_columns.append(config.display_name_field)
    gdf = gpd.read_file(path, bbox=bbox, columns=read_columns)
    if gdf.empty:
        return normalize_stage_gdf(gdf, config)
    return normalize_stage_gdf(gdf, config)


def clear_stage_table(db_url: str, stage_table: str) -> None:
    schema, table = parse_table_name(stage_table)
    conn = open_connection(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"TRUNCATE TABLE {quote_identifier(schema)}.{quote_identifier(table)}"
            )
    finally:
        conn.close()


def load_stage_table(config: AoiConfig, stage_gdf, db_url: str) -> int:
    import sqlalchemy as sa
    from geoalchemy2 import Geometry

    clear_stage_table(db_url, config.stage_table)
    if stage_gdf.empty:
        return 0

    schema, table = parse_table_name(config.stage_table)
    engine = sa.create_engine(normalize_db_url(db_url))
    try:
        stage_gdf.to_postgis(
            table,
            engine,
            schema=schema,
            if_exists="append",
            index=False,
            dtype={"geom": Geometry("MULTIPOLYGON", srid=4326)},
        )
    finally:
        engine.dispose()

    return len(stage_gdf)


def build_config(
    *,
    resolved_asset_slug: str,
    asset,
    ref,
    columns: Sequence[str],
    short_name: str | None = None,
    long_name: str | None = None,
    ext_id_field: str | None = None,
    display_name_field: str | None = None,
    stage_table: str | None = None,
    source_url: str | None = None,
    citation: str | None = None,
    version: str = "latest",
) -> AoiConfig:
    short_name = short_name or slug_to_short_name(resolved_asset_slug)
    ext_id_field = ext_id_field or infer_ext_id_field(columns)
    if display_name_field is None:
        display_name_field = infer_display_name_field(columns, ext_id_field)

    return AoiConfig(
        asset_slug=resolved_asset_slug,
        short_name=short_name,
        long_name=long_name or getattr(ref, "title", None) or resolved_asset_slug,
        ext_id_field=ext_id_field,
        display_name_field=display_name_field,
        stage_table=stage_table or slug_to_stage_table(short_name),
        dataset_version=normalize_dataset_version(
            resolved_asset_slug, getattr(ref, "resolved_id", None), version
        ),
        source_url=source_url or derive_catalog_source_url(asset, ref),
        citation=citation or derive_catalog_citation(asset),
    )


def run_psql_file(db_url: str, config: AoiConfig, batch_size: int) -> None:
    psql = shutil.which("psql")
    if psql is None:
        raise RuntimeError("psql is required to run the SQL preparation script")

    command = [
        psql,
        normalize_db_url(db_url),
        "-v",
        f"aoi_short_name={config.short_name}",
        "-v",
        f"aoi_long_name={config.long_name}",
        "-v",
        f"asset_slug={config.asset_slug}",
        "-v",
        f"ext_id_field={config.ext_id_field}",
        "-v",
        f"display_name_field={config.display_name_field or ''}",
        "-v",
        f"citation={config.citation}",
        "-v",
        f"source_url={config.source_url}",
        "-v",
        f"dataset_version={config.dataset_version}",
        "-v",
        f"stage_table={config.stage_table}",
        "-v",
        f"batch_size={batch_size}",
        "-f",
        str(SQL_SCRIPT),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        if result.stdout:
            LOGGER.info("psql stdout for %s:\n%s", config.short_name, result.stdout)
        return

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    message = f"psql prepare script failed for {config.short_name} with exit code {result.returncode}"
    if stderr:
        message = f"{message}\nSTDERR:\n{stderr}"
    if stdout:
        message = f"{message}\nSTDOUT:\n{stdout}"
    raise RuntimeError(message)


def call_procedure(db_url: str, sql: str, params: tuple[object, ...]) -> None:
    conn = open_connection(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
    finally:
        conn.close()


def query_rows(db_url: str, sql: str, params: tuple[object, ...]) -> list[tuple]:
    conn = open_connection(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())
    finally:
        conn.close()


def query_one_row(db_url: str, sql: str, params: tuple[object, ...]):
    rows = query_rows(db_url, sql, params)
    return rows[0] if rows else None


def execute_values(db_url: str, sql: str, rows: Sequence[tuple[object, ...]]) -> None:
    if not rows:
        return
    conn = open_connection(db_url)
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
    finally:
        conn.close()


def process_chunk_sub_batches(
    db_url: str,
    short_name: str,
    chunk: Mapping[str, object],
    *,
    lock_timeout: str,
    statement_timeout: str,
    slick_to_aoi_buffer_m: float,
) -> tuple[str, int, int, int, int, int, int]:
    conn = open_connection(db_url, autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM maintenance.start_shared_dataset_aoi_backfill_chunk(
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    short_name,
                    chunk["bbox"][0],
                    chunk["bbox"][1],
                    chunk["bbox"][2],
                    chunk["bbox"][3],
                    lock_timeout,
                    statement_timeout,
                ),
            )
            result = cur.fetchone()
            if result is None:
                raise RuntimeError("Chunk setup returned no result row")

            if result[0] == "split_required":
                conn.commit()
                return (
                    "split_required",
                    int(result[1]),
                    int(result[2]),
                    0,
                    int(result[3]),
                    0,
                    0,
                )

            stage_rows = int(result[1])
            candidate_rows = int(result[2])
            aois_inserted = int(result[3])
            batch_size = int(result[4])
            total_sub_batches = (
                math.ceil(candidate_rows / batch_size)
                if candidate_rows and batch_size
                else 0
            )
            total_match_rows = 0
            total_insert_rows = 0

            for sub_batch_index, seq_start in enumerate(
                range(1, candidate_rows + 1, batch_size),
                start=1,
            ):
                cur.execute(
                    """
                    SELECT *
                    FROM maintenance.process_shared_dataset_aoi_backfill_sub_batch(
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (seq_start, batch_size, slick_to_aoi_buffer_m),
                )
                sub_batch_result = cur.fetchone()
                if sub_batch_result is None:
                    raise RuntimeError(
                        f"Sub-batch {sub_batch_index} returned no result row"
                    )

                batch_match_rows = int(sub_batch_result[0])
                batch_insert_rows = int(sub_batch_result[1])
                total_match_rows += batch_match_rows
                total_insert_rows += batch_insert_rows
                batch_end = min(seq_start + batch_size - 1, candidate_rows)
                LOGGER.info(
                    "run_backfill sub-batch progress short_name=%s chunk_id=%s sub_batch=%s/%s seq_range=%s-%s batch_match_rows=%s batch_links_inserted=%s total_match_rows=%s total_links_inserted=%s",
                    short_name,
                    chunk["id"],
                    sub_batch_index,
                    total_sub_batches,
                    seq_start,
                    batch_end,
                    batch_match_rows,
                    batch_insert_rows,
                    total_match_rows,
                    total_insert_rows,
                )

        conn.commit()
        return (
            "completed",
            stage_rows,
            candidate_rows,
            total_match_rows,
            aois_inserted,
            total_insert_rows,
            total_sub_batches,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_db_url(db_url: str | None = None) -> str:
    resolved_db_url = db_url or os.getenv("DB_URL")
    if not resolved_db_url:
        raise RuntimeError("DB_URL is required, either as env var or --db-url")
    return resolved_db_url


def resolve_backfill_short_name(
    asset_slug: str,
    *,
    db_url: str | None = None,
    short_name: str | None = None,
    catalog_source: str | None = None,
) -> str:
    if short_name:
        return short_name
    resolved_db_url = get_db_url(db_url)
    resolved_asset_slug = resolve_asset_slug(asset_slug, catalog_source)
    existing_short_name = resolve_existing_shared_dataset_short_name(
        resolved_db_url, resolved_asset_slug
    )
    return existing_short_name or slug_to_short_name(resolved_asset_slug)


def require_existing_backfill_short_name(
    asset_slug: str,
    *,
    db_url: str | None = None,
    short_name: str | None = None,
    catalog_source: str | None = None,
) -> str:
    if not short_name:
        raise ValueError("AOI backfill requires an explicit short_name")

    resolved_db_url = get_db_url(db_url)
    resolved_asset_slug = resolve_asset_slug(asset_slug, catalog_source)
    row = query_one_row(
        resolved_db_url,
        """
        SELECT
            short_name,
            access_type,
            COALESCE(properties->>'asset_slug', '')
        FROM public.aoi_type
        WHERE short_name = %s
        ORDER BY id
        LIMIT 1
        """,
        (short_name,),
    )
    if row is None:
        raise ValueError(f"AOI type {short_name!r} does not exist in public.aoi_type")
    if row[1] != "SHARED_DATASET":
        raise ValueError(f"AOI type {short_name!r} is not a SHARED_DATASET aoi_type")
    if row[2] != resolved_asset_slug:
        raise ValueError(
            f"AOI type {short_name!r} is configured for asset_slug {row[2]!r}, "
            f"not {resolved_asset_slug!r}"
        )
    return row[0]


def ensure_https_url(url: str) -> str:
    if url.startswith("http://"):
        return "https://" + url[len("http://") :]
    return url


def inspect_asset(
    asset_slug: str,
    *,
    short_name: str | None = None,
    catalog_source: str | None = None,
    version: str = "latest",
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    long_name: str | None = None,
    ext_id_field: str | None = None,
    display_name_field: str | None = None,
    stage_table: str | None = None,
    source_url: str | None = None,
    citation: str | None = None,
) -> dict[str, object]:
    resolved_asset_slug = resolve_asset_slug(asset_slug, catalog_source)
    asset = get_catalog_asset(resolved_asset_slug, catalog_source)
    ref = fetch_dataset_ref(
        resolved_asset_slug,
        version=version,
        cache_dir=Path(cache_dir),
        force=force_download,
        catalog_source=catalog_source,
    )
    dataset_path = Path(ref.cache_path)
    columns = inspect_fields(dataset_path)
    config = build_config(
        resolved_asset_slug=resolved_asset_slug,
        asset=asset,
        ref=ref,
        columns=columns,
        short_name=short_name,
        long_name=long_name,
        ext_id_field=ext_id_field,
        display_name_field=display_name_field,
        stage_table=stage_table,
        source_url=source_url,
        citation=citation,
        version=version,
    )
    chunk_plan = build_chunk_plan(dataset_path)
    result = {
        "input": asset_slug,
        "asset_slug": config.asset_slug,
        "short_name": config.short_name,
        "long_name": config.long_name,
        "ext_id_field": config.ext_id_field,
        "display_name_field": config.display_name_field or "<ext_id>",
        "stage_table": config.stage_table,
        "dataset_version": config.dataset_version,
        "source_url": config.source_url,
        "cache_path": str(dataset_path),
        "fields": columns,
        "chunk_plan": {
            "bounds": chunk_plan["bounds"],
            "grid_side": chunk_plan["grid_side"],
            "target_chunk_count": chunk_plan["target_chunk_count"],
            "target_chunk_bytes": chunk_plan["target_chunk_bytes"],
            "target_chunk_features": chunk_plan["target_chunk_features"],
            "max_chunk_stage_rows": chunk_plan["max_chunk_stage_rows"],
            "split_candidate_slick_limit": chunk_plan["split_candidate_slick_limit"],
            "max_split_depth": chunk_plan["max_split_depth"],
        },
    }
    result.update(inspect_dataset_size(dataset_path))
    result.update(inspect_stage_readiness(dataset_path, config.ext_id_field))
    return to_plain_python(result)


def cleanup_stale_backfills(
    db_url: str, retention: str = DEFAULT_STALE_RETENTION
) -> None:
    call_procedure(
        db_url,
        "SELECT maintenance.cleanup_shared_dataset_aoi_backfills(%s)",
        (retention,),
    )


def insert_chunk_manifest(
    db_url: str, short_name: str, chunks: Sequence[ChunkSpec]
) -> None:
    rows = [
        (
            short_name,
            chunk.chunk_index,
            chunk.parent_chunk_id,
            chunk.split_depth,
            chunk.minx,
            chunk.miny,
            chunk.maxx,
            chunk.maxy,
        )
        for chunk in chunks
    ]
    execute_values(
        db_url,
        """
        INSERT INTO maintenance.shared_dataset_aoi_backfill_chunk (
            aoi_type_short_name,
            chunk_index,
            parent_chunk_id,
            split_depth,
            minx,
            miny,
            maxx,
            maxy
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (aoi_type_short_name, chunk_index) DO NOTHING
        """,
        rows,
    )


def get_run_context(db_url: str, short_name: str) -> RunContext:
    row = query_one_row(
        db_url,
        """
        SELECT
            r.asset_slug,
            r.aoi_type_short_name,
            COALESCE(r.dataset_version, ''),
            r.stage_table::text,
            COALESCE(at.properties->>'ext_id_field', ''),
            NULLIF(at.properties->>'display_name_field', ''),
            r.batch_size,
            r.slick_to_aoi_buffer_m
        FROM maintenance.shared_dataset_aoi_backfill_run r
        JOIN public.aoi_type at ON at.id = r.aoi_type_id
        WHERE r.aoi_type_short_name = %s
        """,
        (short_name,),
    )
    if row is None:
        raise ValueError(f"No prepared backfill run for AOI type {short_name!r}")
    ext_id_field = row[4]
    if not ext_id_field:
        raise ValueError(f"AOI type {short_name!r} is missing ext_id_field metadata")
    return RunContext(
        asset_slug=row[0],
        short_name=row[1],
        dataset_version=row[2] or "latest",
        stage_table=row[3],
        ext_id_field=ext_id_field,
        display_name_field=row[5],
        batch_size=int(row[6]),
        slick_to_aoi_buffer_m=float(row[7] or 0.0),
    )


def refresh_run_status(db_url: str, short_name: str) -> None:
    call_procedure(
        db_url,
        """
        UPDATE maintenance.shared_dataset_aoi_backfill_run r
        SET
            status = CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM maintenance.shared_dataset_aoi_backfill_chunk c
                    WHERE c.aoi_type_short_name = r.aoi_type_short_name
                      AND c.status = 'failed'
                ) THEN 'failed'
                WHEN EXISTS (
                    SELECT 1
                    FROM maintenance.shared_dataset_aoi_backfill_chunk c
                    WHERE c.aoi_type_short_name = r.aoi_type_short_name
                      AND c.status = 'running'
                ) THEN 'running'
                WHEN EXISTS (
                    SELECT 1
                    FROM maintenance.shared_dataset_aoi_backfill_chunk c
                    WHERE c.aoi_type_short_name = r.aoi_type_short_name
                      AND c.status = 'pending'
                ) THEN 'pending'
                ELSE 'completed'
            END,
            completed_at = CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM maintenance.shared_dataset_aoi_backfill_chunk c
                    WHERE c.aoi_type_short_name = r.aoi_type_short_name
                      AND c.status IN ('pending', 'running', 'failed')
                ) THEN NULL
                ELSE now()
            END,
            updated_at = now()
        WHERE r.aoi_type_short_name = %s
        """,
        (short_name,),
    )


def claim_next_chunk(db_url: str, short_name: str):
    row = query_one_row(
        db_url,
        """
        WITH next_chunk AS (
            SELECT id
            FROM maintenance.shared_dataset_aoi_backfill_chunk
            WHERE aoi_type_short_name = %s
              AND status = 'pending'
            ORDER BY split_depth, chunk_index, id
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE maintenance.shared_dataset_aoi_backfill_chunk c
        SET
            status = 'running',
            started_at = COALESCE(c.started_at, now()),
            updated_at = now()
        FROM next_chunk
        WHERE c.id = next_chunk.id
        RETURNING
            c.id,
            c.chunk_index,
            c.parent_chunk_id,
            c.split_depth,
            c.minx,
            c.miny,
            c.maxx,
            c.maxy,
            c.retry_count
        """,
        (short_name,),
    )
    if row is None:
        return None
    refresh_run_status(db_url, short_name)
    return {
        "id": int(row[0]),
        "chunk_index": int(row[1]),
        "parent_chunk_id": row[2],
        "split_depth": int(row[3]),
        "bbox": (float(row[4]), float(row[5]), float(row[6]), float(row[7])),
        "retry_count": int(row[8]),
    }


def mark_chunk_completed(
    db_url: str,
    short_name: str,
    chunk_id: int,
    *,
    stage_rows_loaded: int,
    candidate_slick_rows: int,
    match_rows: int,
    aois_inserted: int,
    links_inserted: int,
    sub_batches: int,
    runtime_seconds: float,
) -> None:
    call_procedure(
        db_url,
        """
        UPDATE maintenance.shared_dataset_aoi_backfill_chunk
        SET
            status = 'completed',
            stage_rows_loaded = %s,
            candidate_slick_rows = %s,
            match_rows = %s,
            aois_inserted = %s,
            links_inserted = %s,
            sub_batches = %s,
            runtime_seconds = %s,
            finished_at = now(),
            updated_at = now(),
            last_error = NULL
        WHERE aoi_type_short_name = %s
          AND id = %s
        """,
        (
            stage_rows_loaded,
            candidate_slick_rows,
            match_rows,
            aois_inserted,
            links_inserted,
            sub_batches,
            runtime_seconds,
            short_name,
            chunk_id,
        ),
    )
    refresh_run_status(db_url, short_name)


def mark_chunk_split(
    db_url: str,
    short_name: str,
    chunk_id: int,
    chunk_index: int,
    split_depth: int,
    bbox: tuple[float, float, float, float],
    runtime_seconds: float,
) -> None:
    child_specs = split_chunk_bbox(
        ChunkSpec(
            chunk_index=chunk_index,
            minx=bbox[0],
            miny=bbox[1],
            maxx=bbox[2],
            maxy=bbox[3],
            split_depth=split_depth,
            parent_chunk_id=chunk_id,
        )
    )
    if not child_specs:
        raise ValueError(f"Cannot split degenerate chunk {chunk_id}")

    child_rows = []
    for idx, child in enumerate(child_specs, start=1):
        child_rows.append(
            (
                short_name,
                chunk_index * 10 + idx,
                chunk_id,
                child.split_depth,
                child.minx,
                child.miny,
                child.maxx,
                child.maxy,
            )
        )

    execute_values(
        db_url,
        """
        INSERT INTO maintenance.shared_dataset_aoi_backfill_chunk (
            aoi_type_short_name,
            chunk_index,
            parent_chunk_id,
            split_depth,
            minx,
            miny,
            maxx,
            maxy
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        child_rows,
    )
    call_procedure(
        db_url,
        """
        UPDATE maintenance.shared_dataset_aoi_backfill_chunk
        SET
            status = 'split',
            runtime_seconds = %s,
            finished_at = now(),
            updated_at = now()
        WHERE aoi_type_short_name = %s
          AND id = %s
        """,
        (runtime_seconds, short_name, chunk_id),
    )
    refresh_run_status(db_url, short_name)


def mark_chunk_retry(
    db_url: str, short_name: str, chunk_id: int, error: str, runtime_seconds: float
) -> None:
    row = query_one_row(
        db_url,
        """
        SELECT retry_count
        FROM maintenance.shared_dataset_aoi_backfill_chunk
        WHERE aoi_type_short_name = %s
          AND id = %s
        """,
        (short_name, chunk_id),
    )
    retry_count = int(row[0]) if row else MAX_CHUNK_RETRIES
    status = "pending" if retry_count + 1 < MAX_CHUNK_RETRIES else "failed"
    call_procedure(
        db_url,
        """
        UPDATE maintenance.shared_dataset_aoi_backfill_chunk
        SET
            status = %s,
            retry_count = retry_count + 1,
            last_error = %s,
            runtime_seconds = %s,
            updated_at = now(),
            finished_at = CASE WHEN %s = 'failed' THEN now() ELSE finished_at END
        WHERE aoi_type_short_name = %s
          AND id = %s
        """,
        (status, error[:2000], runtime_seconds, status, short_name, chunk_id),
    )
    refresh_run_status(db_url, short_name)


def acquire_run_lock(db_url: str, short_name: str):
    conn = open_connection(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(hashtext(%s))",
                (f"shared_dataset_aoi_backfill:{short_name}",),
            )
            locked = bool(cur.fetchone()[0])
        if not locked:
            conn.close()
            raise RuntimeError(f"Backfill is already running for AOI type {short_name}")
        return conn
    except Exception:
        conn.close()
        raise


def release_run_lock(lock_conn, short_name: str) -> None:
    try:
        with lock_conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_unlock(hashtext(%s))",
                (f"shared_dataset_aoi_backfill:{short_name}",),
            )
    finally:
        lock_conn.close()


def enqueue_backfill_run(
    asset_slug: str,
    *,
    short_name: str | None = None,
    catalog_source: str | None = None,
    max_batches: int | None = DEFAULT_RUN_MAX_BATCHES,
    sleep_seconds: float = 0.05,
    lock_timeout: str = "1s",
    statement_timeout: str = DEFAULT_RUN_STATEMENT_TIMEOUT,
    target_url: str,
) -> str:
    from google.cloud import tasks_v2

    project = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GCPREGION")
    queue = os.getenv("AOI_BACKFILL_QUEUE")
    api_key = os.getenv("API_KEY")
    if not project or not location or not queue or not api_key:
        raise RuntimeError(
            "AOI backfill queue env vars are required: "
            "PROJECT_ID/GOOGLE_CLOUD_PROJECT, GCPREGION, AOI_BACKFILL_QUEUE, API_KEY"
        )

    payload = {
        "asset_slug": asset_slug,
        "short_name": short_name,
        "catalog_source": catalog_source,
        "max_batches": max_batches,
        "sleep_seconds": sleep_seconds,
        "lock_timeout": lock_timeout,
        "statement_timeout": statement_timeout,
    }
    client = tasks_v2.CloudTasksClient()
    response = client.create_task(
        request={
            "parent": client.queue_path(project, location, queue),
            "task": {
                "http_request": {
                    "http_method": tasks_v2.HttpMethod.POST,
                    "url": target_url,
                    "headers": {
                        "Content-type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    "body": json.dumps(payload).encode(),
                }
            },
        }
    )
    return response.name


def backfill_has_pending_work(db_url: str, short_name: str) -> bool:
    row = query_one_row(
        db_url,
        """
        SELECT
            count(*) FILTER (WHERE status = 'pending')::bigint,
            count(*) FILTER (WHERE status = 'failed')::bigint
        FROM maintenance.shared_dataset_aoi_backfill_chunk
        WHERE aoi_type_short_name = %s
        """,
        (short_name,),
    )
    if row is None:
        return False
    pending_chunks = int(row[0])
    failed_chunks = int(row[1])
    return pending_chunks > 0 and failed_chunks == 0


def submit_backfill_run(
    asset_slug: str,
    *,
    db_url: str | None = None,
    short_name: str | None = None,
    catalog_source: str | None = None,
    max_batches: int | None = DEFAULT_RUN_MAX_BATCHES,
    sleep_seconds: float = 0.05,
    lock_timeout: str = "1s",
    statement_timeout: str = DEFAULT_RUN_STATEMENT_TIMEOUT,
    target_url: str,
) -> tuple[str, str]:
    resolved_short_name = require_existing_backfill_short_name(
        asset_slug,
        db_url=db_url,
        short_name=short_name,
        catalog_source=catalog_source,
    )
    task_name = enqueue_backfill_run(
        asset_slug,
        short_name=resolved_short_name,
        catalog_source=catalog_source,
        max_batches=max_batches,
        sleep_seconds=sleep_seconds,
        lock_timeout=lock_timeout,
        statement_timeout=statement_timeout,
        target_url=target_url,
    )
    return resolved_short_name, task_name


def continue_backfill_run(
    asset_slug: str,
    *,
    db_url: str | None = None,
    short_name: str | None = None,
    catalog_source: str | None = None,
    max_batches: int | None = DEFAULT_RUN_MAX_BATCHES,
    sleep_seconds: float = 0.05,
    lock_timeout: str = "1s",
    statement_timeout: str = DEFAULT_RUN_STATEMENT_TIMEOUT,
    target_url: str,
) -> tuple[str, str | None]:
    resolved_db_url = get_db_url(db_url)
    resolved_short_name = require_existing_backfill_short_name(
        asset_slug,
        db_url=resolved_db_url,
        short_name=short_name,
        catalog_source=catalog_source,
    )
    run_backfill(
        asset_slug,
        db_url=resolved_db_url,
        short_name=resolved_short_name,
        catalog_source=catalog_source,
        max_batches=max_batches,
        sleep_seconds=sleep_seconds,
        lock_timeout=lock_timeout,
        statement_timeout=statement_timeout,
    )
    if not backfill_has_pending_work(resolved_db_url, resolved_short_name):
        return resolved_short_name, None
    next_task_name = enqueue_backfill_run(
        asset_slug,
        short_name=resolved_short_name,
        catalog_source=catalog_source,
        max_batches=max_batches,
        sleep_seconds=sleep_seconds,
        lock_timeout=lock_timeout,
        statement_timeout=statement_timeout,
        target_url=target_url,
    )
    return resolved_short_name, next_task_name


def prepare_backfill(
    asset_slug: str,
    *,
    db_url: str | None = None,
    short_name: str | None = None,
    catalog_source: str | None = None,
    version: str = "latest",
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    long_name: str | None = None,
    ext_id_field: str | None = None,
    display_name_field: str | None = None,
    stage_table: str | None = None,
    source_url: str | None = None,
    citation: str | None = None,
    batch_size: int = 5000,
) -> AoiConfig:
    started = time.perf_counter()
    resolved_db_url = get_db_url(db_url)
    resolved_asset_slug = resolve_asset_slug(asset_slug, catalog_source)
    LOGGER.info("prepare_backfill resolved asset slug=%s", resolved_asset_slug)
    resolved_short_name = require_existing_backfill_short_name(
        resolved_asset_slug,
        db_url=resolved_db_url,
        short_name=short_name,
        catalog_source=catalog_source,
    )
    LOGGER.info(
        "prepare_backfill canonical short_name=%s",
        resolved_short_name,
    )
    asset = get_catalog_asset(resolved_asset_slug, catalog_source)
    fetch_started = time.perf_counter()
    ref = fetch_dataset_ref(
        resolved_asset_slug,
        version=version,
        cache_dir=Path(cache_dir),
        force=force_download,
        catalog_source=catalog_source,
    )
    LOGGER.info(
        "prepare_backfill fetched dataset asset_slug=%s elapsed_s=%.3f cache_path=%s",
        resolved_asset_slug,
        time.perf_counter() - fetch_started,
        ref.cache_path,
    )
    dataset_path = Path(ref.cache_path)
    fields_started = time.perf_counter()
    columns = inspect_fields(dataset_path)
    LOGGER.info(
        "prepare_backfill inspected fields asset_slug=%s columns=%s elapsed_s=%.3f",
        resolved_asset_slug,
        len(columns),
        time.perf_counter() - fields_started,
    )
    config = build_config(
        resolved_asset_slug=resolved_asset_slug,
        asset=asset,
        ref=ref,
        columns=columns,
        short_name=resolved_short_name,
        long_name=long_name,
        ext_id_field=ext_id_field,
        display_name_field=display_name_field,
        stage_table=stage_table,
        source_url=source_url,
        citation=citation,
        version=version,
    )
    parse_table_name(config.stage_table)
    chunk_started = time.perf_counter()
    chunk_plan = build_chunk_plan(dataset_path)
    LOGGER.info(
        "prepare_backfill built chunk plan asset_slug=%s chunks=%s grid_side=%s elapsed_s=%.3f",
        resolved_asset_slug,
        chunk_plan["target_chunk_count"],
        chunk_plan["grid_side"],
        time.perf_counter() - chunk_started,
    )

    LOGGER.info("asset_slug=%s", config.asset_slug)
    LOGGER.info("short_name=%s", config.short_name)
    LOGGER.info("long_name=%s", config.long_name)
    LOGGER.info("ext_id_field=%s", config.ext_id_field)
    LOGGER.info("display_name_field=%s", config.display_name_field or "<ext_id>")
    LOGGER.info("stage_table=%s", config.stage_table)
    LOGGER.info("dataset_version=%s", config.dataset_version)
    LOGGER.info("planned_chunk_count=%s", chunk_plan["target_chunk_count"])

    psql_started = time.perf_counter()
    run_psql_file(resolved_db_url, config, batch_size)
    LOGGER.info(
        "prepare_backfill bootstrapped SQL short_name=%s elapsed_s=%.3f",
        config.short_name,
        time.perf_counter() - psql_started,
    )
    cleanup_started = time.perf_counter()
    cleanup_stale_backfills(resolved_db_url)
    LOGGER.info(
        "prepare_backfill cleaned stale runs short_name=%s elapsed_s=%.3f",
        config.short_name,
        time.perf_counter() - cleanup_started,
    )
    manifest_started = time.perf_counter()
    insert_chunk_manifest(resolved_db_url, config.short_name, chunk_plan["chunks"])
    LOGGER.info(
        "prepare_backfill inserted chunk manifest short_name=%s chunks=%s elapsed_s=%.3f",
        config.short_name,
        len(chunk_plan["chunks"]),
        time.perf_counter() - manifest_started,
    )
    status_started = time.perf_counter()
    refresh_run_status(resolved_db_url, config.short_name)
    run_context = get_run_context(resolved_db_url, config.short_name)
    LOGGER.info(
        "prepare_backfill refreshed run status short_name=%s elapsed_s=%.3f total_elapsed_s=%.3f",
        config.short_name,
        time.perf_counter() - status_started,
        time.perf_counter() - started,
    )
    return AoiConfig(
        asset_slug=config.asset_slug,
        short_name=config.short_name,
        long_name=config.long_name,
        ext_id_field=config.ext_id_field,
        display_name_field=config.display_name_field,
        stage_table=config.stage_table,
        dataset_version=config.dataset_version,
        source_url=config.source_url,
        citation=config.citation,
        slick_to_aoi_buffer_m=run_context.slick_to_aoi_buffer_m,
    )


def run_backfill(
    asset_slug: str,
    *,
    db_url: str | None = None,
    short_name: str | None = None,
    catalog_source: str | None = None,
    max_batches: int | None = DEFAULT_RUN_MAX_BATCHES,
    sleep_seconds: float = 0.05,
    lock_timeout: str = "1s",
    statement_timeout: str = DEFAULT_RUN_STATEMENT_TIMEOUT,
) -> None:
    import time

    started = time.perf_counter()
    resolved_db_url = get_db_url(db_url)
    resolved_asset_slug = resolve_asset_slug(asset_slug, catalog_source)
    resolved_short_name = require_existing_backfill_short_name(
        resolved_asset_slug,
        db_url=resolved_db_url,
        short_name=short_name,
        catalog_source=catalog_source,
    )
    LOGGER.info(
        "run_backfill resolved asset_slug=%s short_name=%s max_batches=%s",
        resolved_asset_slug,
        resolved_short_name,
        max_batches,
    )
    run_context = get_run_context(resolved_db_url, resolved_short_name)
    asset = get_catalog_asset(run_context.asset_slug, catalog_source)
    fetch_started = time.perf_counter()
    ref = fetch_dataset_ref(
        run_context.asset_slug,
        version=shared_dataset_fetch_version(
            run_context.asset_slug, run_context.dataset_version
        ),
        cache_dir=Path(DEFAULT_CACHE_DIR),
        force=False,
        catalog_source=catalog_source,
    )
    LOGGER.info(
        "run_backfill fetched dataset short_name=%s elapsed_s=%.3f cache_path=%s",
        resolved_short_name,
        time.perf_counter() - fetch_started,
        ref.cache_path,
    )
    config = AoiConfig(
        asset_slug=run_context.asset_slug,
        short_name=run_context.short_name,
        long_name=getattr(asset, "title", None) or run_context.asset_slug,
        ext_id_field=run_context.ext_id_field,
        display_name_field=run_context.display_name_field,
        stage_table=run_context.stage_table,
        dataset_version=run_context.dataset_version,
        source_url=getattr(asset, "canonical_path", None) or "",
        citation=derive_catalog_citation(asset),
    )
    dataset_path = Path(ref.cache_path)

    lock_started = time.perf_counter()
    lock_conn = acquire_run_lock(resolved_db_url, resolved_short_name)
    LOGGER.info(
        "run_backfill acquired lock short_name=%s elapsed_s=%.3f",
        resolved_short_name,
        time.perf_counter() - lock_started,
    )
    try:
        chunks_run = 0
        while max_batches is None or chunks_run < max_batches:
            claim_started = time.perf_counter()
            chunk = claim_next_chunk(resolved_db_url, resolved_short_name)
            if chunk is None:
                refresh_run_status(resolved_db_url, resolved_short_name)
                LOGGER.info(
                    "run_backfill no pending chunks short_name=%s chunks_run=%s total_elapsed_s=%.3f",
                    resolved_short_name,
                    chunks_run,
                    time.perf_counter() - started,
                )
                return
            LOGGER.info(
                "run_backfill claimed chunk short_name=%s chunk_id=%s chunk_index=%s split_depth=%s bbox=%s claim_elapsed_s=%.3f",
                resolved_short_name,
                chunk["id"],
                chunk["chunk_index"],
                chunk["split_depth"],
                chunk["bbox"],
                time.perf_counter() - claim_started,
            )

            try:
                chunk_started = time.perf_counter()
                load_started = time.perf_counter()
                stage_gdf = load_chunk_gdf(config, dataset_path, chunk["bbox"])
                stage_rows = len(stage_gdf)
                LOGGER.info(
                    "run_backfill loaded chunk data short_name=%s chunk_id=%s stage_rows=%s elapsed_s=%.3f",
                    resolved_short_name,
                    chunk["id"],
                    stage_rows,
                    time.perf_counter() - load_started,
                )
                if stage_rows == 0:
                    clear_stage_table(resolved_db_url, config.stage_table)
                    mark_chunk_completed(
                        resolved_db_url,
                        resolved_short_name,
                        chunk["id"],
                        stage_rows_loaded=0,
                        candidate_slick_rows=0,
                        match_rows=0,
                        aois_inserted=0,
                        links_inserted=0,
                        sub_batches=0,
                        runtime_seconds=time.perf_counter() - chunk_started,
                    )
                    LOGGER.info(
                        "run_backfill completed empty chunk short_name=%s chunk_id=%s",
                        resolved_short_name,
                        chunk["id"],
                    )
                elif (
                    stage_rows > DEFAULT_MAX_CHUNK_STAGE_ROWS
                    and chunk["split_depth"] < DEFAULT_MAX_SPLIT_DEPTH
                ):
                    clear_stage_table(resolved_db_url, config.stage_table)
                    mark_chunk_split(
                        resolved_db_url,
                        resolved_short_name,
                        chunk["id"],
                        chunk["chunk_index"],
                        chunk["split_depth"],
                        chunk["bbox"],
                        time.perf_counter() - chunk_started,
                    )
                    LOGGER.info(
                        "run_backfill split oversized chunk short_name=%s chunk_id=%s stage_rows=%s threshold=%s",
                        resolved_short_name,
                        chunk["id"],
                        stage_rows,
                        DEFAULT_MAX_CHUNK_STAGE_ROWS,
                    )
                else:
                    stage_started = time.perf_counter()
                    load_stage_table(config, stage_gdf, resolved_db_url)
                    LOGGER.info(
                        "run_backfill staged chunk rows short_name=%s chunk_id=%s stage_rows=%s elapsed_s=%.3f",
                        resolved_short_name,
                        chunk["id"],
                        stage_rows,
                        time.perf_counter() - stage_started,
                    )
                    db_started = time.perf_counter()
                    result = process_chunk_sub_batches(
                        resolved_db_url,
                        resolved_short_name,
                        chunk,
                        lock_timeout=lock_timeout,
                        statement_timeout=statement_timeout,
                        slick_to_aoi_buffer_m=config.slick_to_aoi_buffer_m,
                    )
                    LOGGER.info(
                        "run_backfill processed chunk in db short_name=%s chunk_id=%s elapsed_s=%.3f",
                        resolved_short_name,
                        chunk["id"],
                        time.perf_counter() - db_started,
                    )
                    if result is None:
                        raise RuntimeError("Chunk processing returned no result row")
                    if result[0] == "split_required":
                        if chunk["split_depth"] >= DEFAULT_MAX_SPLIT_DEPTH:
                            raise RuntimeError(
                                f"Chunk {chunk['id']} exceeded split depth limit"
                            )
                        mark_chunk_split(
                            resolved_db_url,
                            resolved_short_name,
                            chunk["id"],
                            chunk["chunk_index"],
                            chunk["split_depth"],
                            chunk["bbox"],
                            time.perf_counter() - chunk_started,
                        )
                        LOGGER.info(
                            "run_backfill db requested split short_name=%s chunk_id=%s candidate_slick_rows=%s",
                            resolved_short_name,
                            chunk["id"],
                            int(result[2]),
                        )
                    else:
                        mark_chunk_completed(
                            resolved_db_url,
                            resolved_short_name,
                            chunk["id"],
                            stage_rows_loaded=stage_rows,
                            candidate_slick_rows=int(result[2]),
                            match_rows=int(result[3]),
                            aois_inserted=int(result[4]),
                            links_inserted=int(result[5]),
                            sub_batches=int(result[6]),
                            runtime_seconds=time.perf_counter() - chunk_started,
                        )
                        LOGGER.info(
                            "run_backfill completed chunk short_name=%s chunk_id=%s stage_rows=%s candidate_slick_rows=%s match_rows=%s aois_inserted=%s links_inserted=%s sub_batches=%s",
                            resolved_short_name,
                            chunk["id"],
                            stage_rows,
                            int(result[2]),
                            int(result[3]),
                            int(result[4]),
                            int(result[5]),
                            int(result[6]),
                        )
                chunks_run += 1
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            except Exception as exc:
                mark_chunk_retry(
                    resolved_db_url,
                    resolved_short_name,
                    chunk["id"],
                    str(exc),
                    time.perf_counter() - chunk_started,
                )
                LOGGER.exception(
                    "run_backfill chunk failed short_name=%s chunk_id=%s",
                    resolved_short_name,
                    chunk["id"],
                )
                raise
    finally:
        release_run_lock(lock_conn, resolved_short_name)
        LOGGER.info(
            "run_backfill released lock short_name=%s chunks_run=%s total_elapsed_s=%.3f",
            resolved_short_name,
            locals().get("chunks_run", 0),
            time.perf_counter() - started,
        )


def validate_backfill(
    asset_slug: str,
    *,
    db_url: str | None = None,
    short_name: str | None = None,
    catalog_source: str | None = None,
) -> list[tuple]:
    resolved_db_url = get_db_url(db_url)
    resolved_short_name = require_existing_backfill_short_name(
        asset_slug,
        db_url=resolved_db_url,
        short_name=short_name,
        catalog_source=catalog_source,
    )
    return query_rows(
        resolved_db_url,
        "SELECT * FROM maintenance.validate_shared_dataset_aoi_backfill(%s)",
        (resolved_short_name,),
    )


def get_backfill_status(
    asset_slug: str,
    *,
    db_url: str | None = None,
    short_name: str | None = None,
    catalog_source: str | None = None,
) -> list[tuple]:
    resolved_db_url = get_db_url(db_url)
    resolved_short_name = require_existing_backfill_short_name(
        asset_slug,
        db_url=resolved_db_url,
        short_name=short_name,
        catalog_source=catalog_source,
    )
    return query_rows(
        resolved_db_url,
        """
        SELECT
            r.aoi_type_short_name,
            r.status,
            count(*)::bigint AS total_chunks,
            count(*) FILTER (WHERE c.status = 'completed')::bigint AS completed_chunks,
            count(*) FILTER (WHERE c.status = 'pending')::bigint AS pending_chunks,
            count(*) FILTER (WHERE c.status = 'running')::bigint AS running_chunks,
            count(*) FILTER (WHERE c.status = 'failed')::bigint AS failed_chunks,
            COALESCE(sum(c.stage_rows_loaded), 0)::bigint AS staged_rows_loaded,
            COALESCE(sum(c.candidate_slick_rows), 0)::bigint AS candidate_slick_rows,
            COALESCE(sum(c.match_rows), 0)::bigint AS match_rows,
            COALESCE(sum(c.aois_inserted), 0)::bigint AS aois_inserted,
            COALESCE(sum(c.links_inserted), 0)::bigint AS links_inserted,
            r.slick_to_aoi_buffer_m,
            r.updated_at
        FROM maintenance.shared_dataset_aoi_backfill_run r
        JOIN maintenance.shared_dataset_aoi_backfill_chunk c
          ON c.aoi_type_short_name = r.aoi_type_short_name
        WHERE r.aoi_type_short_name = %s
        GROUP BY
            r.aoi_type_short_name,
            r.status,
            r.slick_to_aoi_buffer_m,
            r.updated_at
        """,
        (resolved_short_name,),
    )


def finish_backfill(
    asset_slug: str,
    *,
    db_url: str | None = None,
    short_name: str | None = None,
    catalog_source: str | None = None,
) -> None:
    resolved_db_url = get_db_url(db_url)
    resolved_short_name = require_existing_backfill_short_name(
        asset_slug,
        db_url=resolved_db_url,
        short_name=short_name,
        catalog_source=catalog_source,
    )
    call_procedure(
        resolved_db_url,
        "CALL maintenance.finish_shared_dataset_aoi_backfill(%s)",
        (resolved_short_name,),
    )
