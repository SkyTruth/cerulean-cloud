"""Operator-oriented shared-dataset AOI backfill workflow."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Sequence


LOGGER = logging.getLogger("backfill_shared_dataset_aoi")
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


def load_catalog(catalog_source: str | None):
    from skytruth_shared_datasets import Catalog, DEFAULT_CATALOG_GS_URI

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


def open_connection(db_url: str):
    import psycopg2

    conn = psycopg2.connect(normalize_db_url(db_url))
    conn.autocommit = True
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


def inspect_fields(path: Path) -> list[str]:
    import geopandas as gpd

    sample = gpd.read_file(path, rows=1)
    return [column for column in sample.columns if column != sample.geometry.name]


def inspect_stage_readiness(path: Path, ext_id_field: str) -> dict[str, object]:
    import geopandas as gpd

    gdf = gpd.read_file(path, columns=[ext_id_field])
    ext_ids = gdf[ext_id_field].astype("string").fillna("")
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
    raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type!r}")


def load_stage_table(config: AoiConfig, path: Path, db_url: str) -> int:
    import geopandas as gpd
    import sqlalchemy as sa
    from geoalchemy2 import Geometry

    schema, table = parse_table_name(config.stage_table)
    conn = open_connection(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema)}")
    finally:
        conn.close()

    LOGGER.info("Reading %s", path)
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Dataset {path} is empty")
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    gdf["ext_id"] = gdf[config.ext_id_field].astype("string")
    if config.display_name_field:
        gdf["name"] = gdf[config.display_name_field].astype("string")
    else:
        gdf["name"] = gdf["ext_id"]
    gdf["name"] = gdf["name"].fillna(gdf["ext_id"])
    gdf["geom"] = gdf.geometry.map(promote_polygons)
    stage_gdf = gdf[["ext_id", "name", "geom"]].dropna(subset=["ext_id", "geom"])
    stage_gdf = gpd.GeoDataFrame(stage_gdf, geometry="geom", crs="EPSG:4326")

    engine = sa.create_engine(normalize_db_url(db_url))
    try:
        stage_gdf.to_postgis(
            table,
            engine,
            schema=schema,
            if_exists="replace",
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
    subprocess.run(command, check=True)


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


def get_db_url(db_url: str | None = None) -> str:
    resolved_db_url = db_url or os.getenv("DB_URL")
    if not resolved_db_url:
        raise RuntimeError("DB_URL is required, either as env var or --db-url")
    return resolved_db_url


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
    }
    result.update(inspect_stage_readiness(dataset_path, config.ext_id_field))
    return to_plain_python(result)


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
    resolved_db_url = get_db_url(db_url)
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
    parse_table_name(config.stage_table)

    LOGGER.info("asset_slug=%s", config.asset_slug)
    LOGGER.info("short_name=%s", config.short_name)
    LOGGER.info("long_name=%s", config.long_name)
    LOGGER.info("ext_id_field=%s", config.ext_id_field)
    LOGGER.info("display_name_field=%s", config.display_name_field or "<ext_id>")
    LOGGER.info("stage_table=%s", config.stage_table)
    LOGGER.info("dataset_version=%s", config.dataset_version)

    row_count = load_stage_table(config, dataset_path, resolved_db_url)
    LOGGER.info("Loaded %s staged AOI rows", row_count)
    run_psql_file(resolved_db_url, config, batch_size)
    return config


def run_backfill(
    asset_slug: str,
    *,
    db_url: str | None = None,
    short_name: str | None = None,
    catalog_source: str | None = None,
    max_batches: int | None = 25,
    sleep_seconds: float = 0.05,
    lock_timeout: str = "1s",
    statement_timeout: str = "10min",
) -> None:
    resolved_db_url = get_db_url(db_url)
    resolved_asset_slug = resolve_asset_slug(asset_slug, catalog_source)
    resolved_short_name = short_name or slug_to_short_name(resolved_asset_slug)
    call_procedure(
        resolved_db_url,
        """
        CALL maintenance.run_shared_dataset_aoi_backfill(
            %s,
            %s,
            %s,
            %s,
            %s
        )
        """,
        (
            resolved_short_name,
            max_batches,
            sleep_seconds,
            lock_timeout,
            statement_timeout,
        ),
    )


def validate_backfill(
    asset_slug: str,
    *,
    db_url: str | None = None,
    short_name: str | None = None,
    catalog_source: str | None = None,
) -> list[tuple]:
    resolved_db_url = get_db_url(db_url)
    resolved_asset_slug = resolve_asset_slug(asset_slug, catalog_source)
    resolved_short_name = short_name or slug_to_short_name(resolved_asset_slug)
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
    resolved_asset_slug = resolve_asset_slug(asset_slug, catalog_source)
    resolved_short_name = short_name or slug_to_short_name(resolved_asset_slug)
    return query_rows(
        resolved_db_url,
        """
        SELECT
            aoi_type_short_name,
            status,
            next_slick_id,
            max_slick_id_at_start,
            total_slick_rows,
            total_match_rows,
            total_aoi_rows_inserted,
            total_slick_to_aoi_rows_inserted,
            updated_at
        FROM maintenance.shared_dataset_aoi_backfill_run
        WHERE aoi_type_short_name = %s
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
    resolved_asset_slug = resolve_asset_slug(asset_slug, catalog_source)
    resolved_short_name = short_name or slug_to_short_name(resolved_asset_slug)
    call_procedure(
        resolved_db_url,
        "CALL maintenance.finish_shared_dataset_aoi_backfill(%s)",
        (resolved_short_name,),
    )
