#!/usr/bin/env python3
"""
Operator wrapper for shared-dataset AOI slick_to_aoi backfills.

The common path only needs the shared-datasets asset slug:

    scripts/backfill_shared_dataset_aoi.py prepare <asset-slug>
    scripts/backfill_shared_dataset_aoi.py run <asset-slug>
    scripts/backfill_shared_dataset_aoi.py status <asset-slug>
    scripts/backfill_shared_dataset_aoi.py validate <asset-slug>
    scripts/backfill_shared_dataset_aoi.py finish <asset-slug>

Optional flags exist for the fields that cannot always be inferred safely.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Sequence


LOGGER = logging.getLogger("backfill_shared_dataset_aoi")
REPO_ROOT = Path(__file__).resolve().parents[1]
SQL_SCRIPT = REPO_ROOT / "scripts/backfill_shared_dataset_aoi.sql"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

EXT_ID_FIELD_CANDIDATES = (
    "ext_id",
    "external_id",
    "mrgid",
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


def fetch_dataset_ref(asset_slug: str, *, version: str, cache_dir: Path, force: bool):
    from skytruth_shared_datasets import fetch_dataset

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


def inspect_fields(path: Path) -> list[str]:
    import geopandas as gpd

    sample = gpd.read_file(path, rows=1)
    return [column for column in sample.columns if column != sample.geometry.name]


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
    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema)}")

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


def build_config(args: argparse.Namespace, ref, columns: Sequence[str]) -> AoiConfig:
    short_name = args.short_name or slug_to_short_name(args.asset_slug)
    ext_id_field = args.ext_id_field or infer_ext_id_field(columns)
    display_name_field = args.display_name_field
    if display_name_field is None:
        display_name_field = infer_display_name_field(columns, ext_id_field)

    return AoiConfig(
        asset_slug=args.asset_slug,
        short_name=short_name,
        long_name=args.long_name or getattr(ref, "title", None) or args.asset_slug,
        ext_id_field=ext_id_field,
        display_name_field=display_name_field,
        stage_table=args.stage_table or slug_to_stage_table(short_name),
        dataset_version=str(getattr(ref, "resolved_id", None) or args.version),
        source_url=args.source_url or getattr(ref, "url", None) or "",
        citation=args.citation or "",
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
    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def query_rows(db_url: str, sql: str, params: tuple[object, ...]) -> list[tuple]:
    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())


def prepare(args: argparse.Namespace) -> None:
    db_url = args.db_url or os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL is required, either as env var or --db-url")

    ref = fetch_dataset_ref(
        args.asset_slug,
        version=args.version,
        cache_dir=Path(args.cache_dir),
        force=args.force_download,
    )
    dataset_path = Path(ref.cache_path)
    columns = inspect_fields(dataset_path)
    config = build_config(args, ref, columns)
    parse_table_name(config.stage_table)

    LOGGER.info("asset_slug=%s", config.asset_slug)
    LOGGER.info("short_name=%s", config.short_name)
    LOGGER.info("long_name=%s", config.long_name)
    LOGGER.info("ext_id_field=%s", config.ext_id_field)
    LOGGER.info("display_name_field=%s", config.display_name_field or "<ext_id>")
    LOGGER.info("stage_table=%s", config.stage_table)
    LOGGER.info("dataset_version=%s", config.dataset_version)

    row_count = load_stage_table(config, dataset_path, db_url)
    LOGGER.info("Loaded %s staged AOI rows", row_count)
    run_psql_file(db_url, config, args.batch_size)


def run(args: argparse.Namespace) -> None:
    db_url = args.db_url or os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL is required, either as env var or --db-url")
    short_name = args.short_name or slug_to_short_name(args.asset_slug)
    call_procedure(
        db_url,
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
            short_name,
            args.max_batches,
            args.sleep_seconds,
            args.lock_timeout,
            args.statement_timeout,
        ),
    )


def validate(args: argparse.Namespace) -> None:
    db_url = args.db_url or os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL is required, either as env var or --db-url")
    short_name = args.short_name or slug_to_short_name(args.asset_slug)
    rows = query_rows(
        db_url,
        "SELECT * FROM maintenance.validate_shared_dataset_aoi_backfill(%s)",
        (short_name,),
    )
    for check_name, value in rows:
        print(f"{check_name}\t{value}")


def status(args: argparse.Namespace) -> None:
    db_url = args.db_url or os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL is required, either as env var or --db-url")
    short_name = args.short_name or slug_to_short_name(args.asset_slug)
    rows = query_rows(
        db_url,
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
        (short_name,),
    )
    if not rows:
        print(f"No prepared backfill run for {short_name}")
        return
    headers = (
        "short_name",
        "status",
        "next_slick_id",
        "max_slick_id_at_start",
        "slicks_scanned",
        "matches",
        "aois_inserted",
        "links_inserted",
        "updated_at",
    )
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(value) for value in row))


def finish(args: argparse.Namespace) -> None:
    db_url = args.db_url or os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError("DB_URL is required, either as env var or --db-url")
    short_name = args.short_name or slug_to_short_name(args.asset_slug)
    call_procedure(
        db_url,
        "CALL maintenance.finish_shared_dataset_aoi_backfill(%s)",
        (short_name,),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        help="Database URL. Defaults to DB_URL.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log debug output.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    add_common_asset_args(prepare_parser)
    prepare_parser.add_argument("--version", default="latest")
    prepare_parser.add_argument(
        "--cache-dir",
        default=str(Path(gettempdir()) / "cerulean_aoi_backfill_cache"),
    )
    prepare_parser.add_argument("--force-download", action="store_true")
    prepare_parser.add_argument("--long-name")
    prepare_parser.add_argument("--ext-id-field")
    prepare_parser.add_argument("--display-name-field")
    prepare_parser.add_argument("--stage-table")
    prepare_parser.add_argument("--source-url")
    prepare_parser.add_argument("--citation")
    prepare_parser.add_argument("--batch-size", type=int, default=5000)
    prepare_parser.set_defaults(func=prepare)

    run_parser = subparsers.add_parser("run")
    add_common_asset_args(run_parser)
    run_parser.add_argument("--max-batches", type=int, default=25)
    run_parser.add_argument("--sleep-seconds", type=float, default=0.05)
    run_parser.add_argument("--lock-timeout", default="1s")
    run_parser.add_argument("--statement-timeout", default="10min")
    run_parser.set_defaults(func=run)

    validate_parser = subparsers.add_parser("validate")
    add_common_asset_args(validate_parser)
    validate_parser.set_defaults(func=validate)

    status_parser = subparsers.add_parser("status")
    add_common_asset_args(status_parser)
    status_parser.set_defaults(func=status)

    finish_parser = subparsers.add_parser("finish")
    add_common_asset_args(finish_parser)
    finish_parser.set_defaults(func=finish)

    return parser


def add_common_asset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("asset_slug")
    parser.add_argument(
        "--short-name",
        help="Override derived AOI type short_name. Defaults to upper snake-case slug.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        args.func(args)
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
