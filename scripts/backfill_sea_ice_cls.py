"""CLI helper for the historical sea-ice classification backfill.

This script automates the workflow around scripts/backfill_sea_ice_cls.sql:
1. Apply the SQL helper objects and rebuild the queue.
2. Resolve and download NOAA MASIE masks for pending scene dates.
3. Stage normalized masks into public.masie_sea_ice_stage.
4. Run one or more DB workers that execute the queue procedure.
"""

from __future__ import annotations

import logging
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import click
import geopandas as gpd
import httpx
import psycopg2
from psycopg2.extras import execute_values


LOGGER = logging.getLogger("sea_ice_backfill")
DEFAULT_SQL_PATH = Path(__file__).with_name("backfill_sea_ice_cls.sql")
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "cerulean-masie-cache"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Context:
    db_url: str
    sql_path: Path
    cache_dir: Path


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def open_connection(db_url: str, autocommit: bool = False):
    conn = psycopg2.connect(db_url)
    conn.autocommit = autocommit
    return conn


def apply_sql_file(db_url: str, sql_path: Path) -> None:
    sql_text = sql_path.read_text()
    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_text)
        conn.commit()


def fetch_scalar(db_url: str, query: str, params: tuple | None = None):
    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
        conn.commit()
    return row[0] if row else None


def fetch_scene_dates(db_url: str, limit: int | None = None) -> list[date]:
    query = """
        SELECT DISTINCT scene_date
        FROM public.sea_ice_backfill_queue
        ORDER BY scene_date
    """
    params: tuple | None = None
    if limit is not None:
        query += " LIMIT %s"
        params = (limit,)

    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        conn.commit()
    return [row[0] for row in rows]


def fetch_stage_dates(db_url: str) -> set[date]:
    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT mask_date FROM public.masie_sea_ice_stage")
            rows = cur.fetchall()
        conn.commit()
    return {row[0] for row in rows}


def fetch_uncovered_scene_dates(db_url: str) -> list[date]:
    query = """
        SELECT DISTINCT q.scene_date
        FROM public.sea_ice_backfill_queue q
        LEFT JOIN LATERAL (
            SELECT MAX(mask_date) AS applied_mask_date
            FROM public.masie_sea_ice_stage sm
            WHERE sm.mask_date <= q.scene_date
        ) sm ON TRUE
        WHERE sm.applied_mask_date IS NULL
        ORDER BY q.scene_date
    """
    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        conn.commit()
    return [row[0] for row in rows]


def fetch_counts(db_url: str) -> dict[str, int]:
    query = """
        SELECT
            (SELECT COUNT(*) FROM public.sea_ice_backfill_queue) AS queued_runs,
            (SELECT COUNT(*) FROM public.sea_ice_backfill_audit) AS processed_runs,
            (
                SELECT COALESCE(SUM(updated_slick_count), 0)
                FROM public.sea_ice_backfill_audit
            ) AS updated_slicks
    """
    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
        conn.commit()
    return {
        "queued_runs": row[0],
        "processed_runs": row[1],
        "updated_slicks": row[2],
    }


def is_retryable_sea_ice_mask_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(exc, httpx.TransportError)


def get_retry_sleep_seconds(attempt: int, retry_backoff_seconds: float) -> float:
    exponential_backoff_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
    jitter_seconds = random.uniform(0, exponential_backoff_seconds)
    return exponential_backoff_seconds + jitter_seconds


def archive_url_for_date(mask_date: date) -> str:
    return (
        "https://noaadata.apps.nsidc.org/NOAA/G02186/shapefiles/4km/"
        f"{mask_date.year}/masie_ice_r00_v01_{mask_date:%Y%j}_4km.zip"
    )


def download_archive(
    mask_date: date,
    cache_dir: Path,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"masie_ice_r00_v01_{mask_date:%Y%j}_4km.zip"
    if archive_path.exists():
        return archive_path

    url = archive_url_for_date(mask_date)
    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.stream(
                "GET", url, timeout=60, follow_redirects=True
            ) as response:
                response.raise_for_status()
                with archive_path.open("wb") as dst:
                    for chunk in response.iter_bytes():
                        dst.write(chunk)
            return archive_path
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise
            if not is_retryable_sea_ice_mask_error(exc) or attempt == max_attempts:
                raise
            sleep_seconds = get_retry_sleep_seconds(attempt, retry_backoff_seconds)
            LOGGER.warning(
                "Retrying mask download after HTTP %s for %s in %.2fs",
                exc.response.status_code,
                mask_date,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
        except httpx.TransportError as exc:
            if attempt == max_attempts:
                raise
            sleep_seconds = get_retry_sleep_seconds(attempt, retry_backoff_seconds)
            LOGGER.warning(
                "Retrying mask download after %s for %s in %.2fs",
                type(exc).__name__,
                mask_date,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to download mask archive for {mask_date}")


def load_mask_gdf_from_archive(archive_path: Path) -> gpd.GeoDataFrame:
    with ZipFile(archive_path) as archive:
        shp_names = [
            name for name in archive.namelist() if name.lower().endswith(".shp")
        ]
        if len(shp_names) != 1:
            raise ValueError(
                f"Expected exactly one shapefile in {archive_path}, found {len(shp_names)}"
            )
        gdf = gpd.read_file(f"zip://{archive_path}!{shp_names[0]}")

    if gdf.crs is None:
        raise ValueError(f"Mask archive {archive_path} has no CRS metadata")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def download_mask_gdf(
    mask_date: date,
    cache_dir: Path,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> gpd.GeoDataFrame:
    archive_path = download_archive(
        mask_date,
        cache_dir=cache_dir,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return load_mask_gdf_from_archive(archive_path)


def resolve_mask_for_scene_date(
    scene_date: date,
    stage_dates: set[date],
    known_missing_dates: set[date],
    downloaded_masks: dict[date, gpd.GeoDataFrame],
    cache_dir: Path,
    earliest_supported_date: date,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> tuple[date, gpd.GeoDataFrame | None]:
    current_date = scene_date
    while current_date >= earliest_supported_date:
        if current_date in stage_dates:
            return current_date, None
        if current_date in downloaded_masks:
            return current_date, downloaded_masks[current_date]
        if current_date in known_missing_dates:
            current_date -= timedelta(days=1)
            continue
        try:
            gdf = download_mask_gdf(
                current_date,
                cache_dir=cache_dir,
                max_attempts=max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            downloaded_masks[current_date] = gdf
            return current_date, gdf
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            known_missing_dates.add(current_date)
            current_date -= timedelta(days=1)
    raise RuntimeError(
        f"No MASIE mask found on or after {earliest_supported_date} for scene date {scene_date}"
    )


def geometry_rows_for_insert(
    mask_date: date, gdf: gpd.GeoDataFrame
) -> list[tuple[date, str]]:
    rows: list[tuple[date, str]] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        rows.append((mask_date, geom.wkb_hex))
    return rows


def replace_mask_rows(db_url: str, mask_date: date, gdf: gpd.GeoDataFrame) -> int:
    rows = geometry_rows_for_insert(mask_date, gdf)
    if not rows:
        raise ValueError(f"Mask {mask_date} produced no geometries after normalization")

    insert_sql = """
        WITH data(mask_date, geom_hex) AS (VALUES %s),
        normalized AS (
            SELECT
                data.mask_date,
                ST_Multi(
                    ST_CollectionExtract(
                        ST_MakeValid(
                            ST_GeomFromWKB(decode(data.geom_hex, 'hex'), 4326)
                        ),
                        3
                    )
                )::geometry(MultiPolygon, 4326) AS geom
            FROM data
        )
        INSERT INTO public.masie_sea_ice_stage (mask_date, geom)
        SELECT mask_date, geom
        FROM normalized
        WHERE geom IS NOT NULL
          AND NOT ST_IsEmpty(geom)
    """

    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM public.masie_sea_ice_stage WHERE mask_date = %s",
                (mask_date,),
            )
            execute_values(cur, insert_sql, rows, page_size=500)
        conn.commit()
    return len(rows)


def load_masks_for_pending_scene_dates(
    ctx: Context,
    earliest_supported_date: date,
    max_attempts: int,
    retry_backoff_seconds: float,
    limit_scene_dates: int | None = None,
) -> dict[str, int]:
    scene_dates = fetch_scene_dates(ctx.db_url, limit=limit_scene_dates)
    if not scene_dates:
        LOGGER.info("No queued scene dates found")
        return {"scene_dates": 0, "mask_dates_loaded": 0}

    stage_dates = fetch_stage_dates(ctx.db_url)
    known_missing_dates: set[date] = set()
    downloaded_masks: dict[date, gpd.GeoDataFrame] = {}
    loaded_mask_dates = 0

    for idx, scene_date in enumerate(scene_dates, start=1):
        resolved_mask_date, gdf = resolve_mask_for_scene_date(
            scene_date=scene_date,
            stage_dates=stage_dates,
            known_missing_dates=known_missing_dates,
            downloaded_masks=downloaded_masks,
            cache_dir=ctx.cache_dir,
            earliest_supported_date=earliest_supported_date,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        if resolved_mask_date in stage_dates:
            LOGGER.info(
                "[%s/%s] scene_date=%s already covered by staged mask_date=%s",
                idx,
                len(scene_dates),
                scene_date,
                resolved_mask_date,
            )
            continue

        if gdf is None:
            archive_path = download_archive(
                resolved_mask_date,
                cache_dir=ctx.cache_dir,
                max_attempts=max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            gdf = load_mask_gdf_from_archive(archive_path)

        row_count = replace_mask_rows(ctx.db_url, resolved_mask_date, gdf)
        stage_dates.add(resolved_mask_date)
        loaded_mask_dates += 1
        LOGGER.info(
            "[%s/%s] staged mask_date=%s for scene_date=%s (%s geometries)",
            idx,
            len(scene_dates),
            resolved_mask_date,
            scene_date,
            row_count,
        )

    return {"scene_dates": len(scene_dates), "mask_dates_loaded": loaded_mask_dates}


def run_single_worker(
    db_url: str,
    worker_id: int,
    run_batch_size: int,
    sleep_sec: float,
    buffer_m: float,
    stop_on_missing_mask: bool,
) -> None:
    LOGGER.info("Worker %s starting", worker_id)
    with open_connection(db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CALL public.backfill_sea_ice_slick_cls_from_queue(%s, %s, %s, %s)
                """,
                (run_batch_size, sleep_sec, buffer_m, stop_on_missing_mask),
            )
    LOGGER.info("Worker %s finished", worker_id)


def run_workers(
    db_url: str,
    workers: int,
    run_batch_size: int,
    sleep_sec: float,
    buffer_m: float,
    stop_on_missing_mask: bool,
) -> None:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                run_single_worker,
                db_url,
                worker_id,
                run_batch_size,
                sleep_sec,
                buffer_m,
                stop_on_missing_mask,
            )
            for worker_id in range(1, workers + 1)
        ]
        for future in futures:
            future.result()


@click.group()
@click.option(
    "--db-url", envvar="DB_URL", required=True, help="Postgres connection URL."
)
@click.option(
    "--sql-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_SQL_PATH,
    show_default=True,
    help="Path to the SQL helper script.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_CACHE_DIR,
    show_default=True,
    help="Directory to cache MASIE zip archives.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def cli(
    ctx: click.Context, db_url: str, sql_path: Path, cache_dir: Path, verbose: bool
):
    """Automate the historical SEA_ICE backfill workflow."""
    configure_logging(verbose)
    ctx.obj = Context(db_url=db_url, sql_path=sql_path, cache_dir=cache_dir)


@cli.command("prepare")
@click.pass_obj
def prepare_command(ctx: Context) -> None:
    """Apply the SQL helper script and rebuild the queue."""
    apply_sql_file(ctx.db_url, ctx.sql_path)
    counts = fetch_counts(ctx.db_url)
    LOGGER.info("Prepared queue: %s", counts)


@cli.command("load-masks")
@click.option(
    "--earliest-supported-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default="2010-01-01",
    show_default=True,
    help="Stop searching for fallback masks before this date.",
)
@click.option(
    "--max-attempts",
    type=int,
    default=5,
    show_default=True,
    help="Retry attempts for retryable NOAA failures.",
)
@click.option(
    "--retry-backoff-seconds",
    type=float,
    default=0.5,
    show_default=True,
    help="Base exponential backoff for retryable NOAA failures.",
)
@click.option(
    "--limit-scene-dates",
    type=int,
    default=None,
    help="Only resolve this many queued scene dates, for canary runs.",
)
@click.pass_obj
def load_masks_command(
    ctx: Context,
    earliest_supported_date: datetime,
    max_attempts: int,
    retry_backoff_seconds: float,
    limit_scene_dates: int | None,
) -> None:
    """Resolve pending scene dates and stage any missing MASIE masks."""
    result = load_masks_for_pending_scene_dates(
        ctx=ctx,
        earliest_supported_date=earliest_supported_date.date(),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        limit_scene_dates=limit_scene_dates,
    )
    uncovered_scene_dates = fetch_uncovered_scene_dates(ctx.db_url)
    LOGGER.info(
        "Mask loading complete: scene_dates=%s mask_dates_loaded=%s uncovered_scene_dates=%s",
        result["scene_dates"],
        result["mask_dates_loaded"],
        len(uncovered_scene_dates),
    )
    if uncovered_scene_dates:
        LOGGER.warning("Uncovered scene dates remain: %s", uncovered_scene_dates[:10])


@cli.command("run")
@click.option("--workers", type=int, default=1, show_default=True)
@click.option("--run-batch-size", type=int, default=50, show_default=True)
@click.option("--sleep-sec", type=float, default=0.05, show_default=True)
@click.option("--buffer-m", type=float, default=1000, show_default=True)
@click.option(
    "--stop-on-missing-mask/--skip-missing-mask",
    default=True,
    show_default=True,
    help="Mirror the SQL procedure's missing-mask behavior.",
)
@click.pass_obj
def run_command(
    ctx: Context,
    workers: int,
    run_batch_size: int,
    sleep_sec: float,
    buffer_m: float,
    stop_on_missing_mask: bool,
) -> None:
    """Run one or more DB backfill workers."""
    run_workers(
        db_url=ctx.db_url,
        workers=workers,
        run_batch_size=run_batch_size,
        sleep_sec=sleep_sec,
        buffer_m=buffer_m,
        stop_on_missing_mask=stop_on_missing_mask,
    )
    LOGGER.info("Backfill workers finished: %s", fetch_counts(ctx.db_url))


@cli.command("all")
@click.option(
    "--earliest-supported-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default="2010-01-01",
    show_default=True,
)
@click.option("--max-attempts", type=int, default=5, show_default=True)
@click.option(
    "--retry-backoff-seconds",
    type=float,
    default=0.5,
    show_default=True,
)
@click.option("--limit-scene-dates", type=int, default=None)
@click.option("--workers", type=int, default=1, show_default=True)
@click.option("--run-batch-size", type=int, default=50, show_default=True)
@click.option("--sleep-sec", type=float, default=0.05, show_default=True)
@click.option("--buffer-m", type=float, default=1000, show_default=True)
@click.option(
    "--stop-on-missing-mask/--skip-missing-mask",
    default=True,
    show_default=True,
)
@click.pass_obj
def all_command(
    ctx: Context,
    earliest_supported_date: datetime,
    max_attempts: int,
    retry_backoff_seconds: float,
    limit_scene_dates: int | None,
    workers: int,
    run_batch_size: int,
    sleep_sec: float,
    buffer_m: float,
    stop_on_missing_mask: bool,
) -> None:
    """Prepare the queue, load masks, and run the backfill workers."""
    apply_sql_file(ctx.db_url, ctx.sql_path)
    load_masks_for_pending_scene_dates(
        ctx=ctx,
        earliest_supported_date=earliest_supported_date.date(),
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        limit_scene_dates=limit_scene_dates,
    )
    uncovered_scene_dates = fetch_uncovered_scene_dates(ctx.db_url)
    if uncovered_scene_dates and stop_on_missing_mask:
        raise click.ClickException(
            f"{len(uncovered_scene_dates)} uncovered scene dates remain; "
            "load more masks or rerun with --skip-missing-mask."
        )
    run_workers(
        db_url=ctx.db_url,
        workers=workers,
        run_batch_size=run_batch_size,
        sleep_sec=sleep_sec,
        buffer_m=buffer_m,
        stop_on_missing_mask=stop_on_missing_mask,
    )
    LOGGER.info("All steps finished: %s", fetch_counts(ctx.db_url))


if __name__ == "__main__":
    cli()
