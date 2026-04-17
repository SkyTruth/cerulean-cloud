#!/usr/bin/env python3
"""
Backfill SEA_ICE classifications for historical north-polar runs.

This helper intentionally stays small:
- no persistent queue table
- no persistent stage table
- no audit table

`orchestrator_run.sea_ice_date` is the durable progress marker. The script
processes one `scene_date` per transaction so frontend users do not pay for one
large write.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import geopandas as gpd
import httpx
import psycopg2

LOGGER = logging.getLogger("backfill_sea_ice_cls")
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MASIE_MIN_SCENE_MAX_LAT = 50


@dataclass(frozen=True)
class ClassIds:
    not_oil_root_id: int
    sea_ice_cls_id: int


@dataclass(frozen=True)
class WorkerResult:
    processed_dates: int
    updated_slicks: int
    stamped_runs: int
    unresolved_dates: int


class MaskUnavailableError(Exception):
    """Raised when a mask date should fall back to an earlier available day."""


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def normalize_db_url(db_url: str) -> str:
    if db_url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + db_url[len("postgresql+asyncpg://") :]
    return db_url


def open_connection(db_url: str):
    return psycopg2.connect(normalize_db_url(db_url))


def fetch_class_ids(db_url: str) -> ClassIds:
    query = """
        SELECT
            (SELECT id FROM public.cls WHERE short_name = 'NOT_OIL') AS not_oil_root_id,
            (SELECT id FROM public.cls WHERE short_name = 'SEA_ICE') AS sea_ice_cls_id
    """
    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()

    if row is None or row[0] is None or row[1] is None:
        raise RuntimeError(
            "Missing cls.short_name rows for NOT_OIL or SEA_ICE. Apply the taxonomy migration first."
        )

    return ClassIds(not_oil_root_id=row[0], sea_ice_cls_id=row[1])


def fetch_pending_scene_dates(db_url: str, limit_scene_dates: int | None) -> list[date]:
    query = """
        SELECT DISTINCT sg.start_time::date AS scene_date
        FROM public.orchestrator_run o
        JOIN public.sentinel1_grd sg
          ON sg.id = o.sentinel1_grd
        WHERE o.success IS TRUE
          AND o.sea_ice_date IS NULL
          AND ST_YMax(sg.geometry::geometry) >= %s
        ORDER BY scene_date DESC
    """
    params: list[object] = [MASIE_MIN_SCENE_MAX_LAT]
    if limit_scene_dates is not None:
        query += " LIMIT %s"
        params.append(limit_scene_dates)

    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

    return [row[0] for row in rows]


def archive_url_for_date(mask_date: date) -> str:
    return (
        "https://noaadata.apps.nsidc.org/NOAA/G02186/shapefiles/4km/"
        f"{mask_date.year}/masie_ice_r00_v01_{mask_date:%Y%j}_4km.zip"
    )


def get_retry_sleep_seconds(attempt: int, retry_backoff_seconds: float) -> float:
    exponential_backoff_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
    jitter_seconds = random.uniform(0, exponential_backoff_seconds)
    return exponential_backoff_seconds + jitter_seconds


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
    return gdf


def get_mask_proj4(mask_gdf: gpd.GeoDataFrame) -> str:
    if mask_gdf.crs is None:
        raise ValueError("Mask has no CRS metadata")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You will likely lose important projection information",
            category=UserWarning,
        )
        proj4 = mask_gdf.crs.to_proj4()
    if not proj4:
        raise ValueError(f"Mask CRS {mask_gdf.crs!r} could not be converted to PROJ.4")

    return proj4.replace(" +type=crs", "")


def get_mask_geom_hex(mask_gdf: gpd.GeoDataFrame) -> str:
    if hasattr(mask_gdf.geometry, "union_all"):
        mask_geometry = mask_gdf.geometry.union_all()
    else:
        mask_geometry = mask_gdf.geometry.unary_union
    if mask_geometry is None or mask_geometry.is_empty:
        raise ValueError("Mask produced no geometry after dissolve")
    return mask_geometry.wkb_hex


def download_mask_gdf(
    mask_date: date,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> gpd.GeoDataFrame:
    url = archive_url_for_date(mask_date)
    with TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / f"masie_ice_r00_v01_{mask_date:%Y%j}_4km.zip"
        for attempt in range(1, max_attempts + 1):
            try:
                with httpx.stream(
                    "GET", url, timeout=60, follow_redirects=True
                ) as response:
                    response.raise_for_status()
                    with archive_path.open("wb") as dst:
                        for chunk in response.iter_bytes():
                            dst.write(chunk)
                return load_mask_gdf_from_archive(archive_path)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise MaskUnavailableError(mask_date.isoformat()) from exc
                if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                    raise
                if attempt == max_attempts:
                    raise MaskUnavailableError(mask_date.isoformat()) from exc
                sleep_seconds = get_retry_sleep_seconds(attempt, retry_backoff_seconds)
                LOGGER.warning(
                    "Retrying MASIE download after HTTP %s for %s in %.2fs",
                    exc.response.status_code,
                    mask_date,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
            except httpx.TransportError as exc:
                if attempt == max_attempts:
                    raise MaskUnavailableError(mask_date.isoformat()) from exc
                sleep_seconds = get_retry_sleep_seconds(attempt, retry_backoff_seconds)
                LOGGER.warning(
                    "Retrying MASIE download after %s for %s in %.2fs",
                    type(exc).__name__,
                    mask_date,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to resolve mask for {mask_date}")


def resolve_mask_for_scene_date(
    scene_date: date,
    earliest_supported_date: date,
    max_attempts: int,
    retry_backoff_seconds: float,
    mask_cache: dict[date, gpd.GeoDataFrame],
    unavailable_mask_dates: set[date],
) -> tuple[date | None, gpd.GeoDataFrame | None]:
    current_date = scene_date
    while current_date >= earliest_supported_date:
        if current_date in mask_cache:
            return current_date, mask_cache[current_date]
        if current_date in unavailable_mask_dates:
            current_date -= timedelta(days=1)
            continue

        try:
            gdf = download_mask_gdf(
                current_date,
                max_attempts=max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            mask_cache[current_date] = gdf
            return current_date, gdf
        except MaskUnavailableError:
            unavailable_mask_dates.add(current_date)
            current_date -= timedelta(days=1)

    return None, None


def backfill_scene_date(
    db_url: str,
    scene_date: date,
    mask_date: date,
    mask_gdf: gpd.GeoDataFrame,
    class_ids: ClassIds,
    buffer_m: float,
) -> tuple[int, int]:
    mask_proj4 = get_mask_proj4(mask_gdf)
    mask_geom_hex = get_mask_geom_hex(mask_gdf)
    query = """
        WITH mask_geom AS (
            SELECT
                ST_Multi(
                    ST_CollectionExtract(
                        ST_MakeValid(ST_GeomFromWKB(decode(%(mask_geom_hex)s, 'hex'))),
                        3
                    )
                )::geometry(MultiPolygon) AS geom
        ),
        not_oil_clses AS (
            SELECT c.id
            FROM public.get_slick_subclses(%(not_oil_root_id)s) AS c
        ),
        candidate_runs AS (
            SELECT o.id
            FROM public.orchestrator_run o
            JOIN public.sentinel1_grd sg
              ON sg.id = o.sentinel1_grd
            WHERE o.success IS TRUE
              AND o.sea_ice_date IS NULL
              AND sg.start_time::date = %(scene_date)s
              AND ST_YMax(sg.geometry::geometry) >= %(min_scene_lat)s
        ),
        candidate_slicks AS (
            SELECT s.id, s.orchestrator_run
            FROM public.slick s
            JOIN candidate_runs r
              ON r.id = s.orchestrator_run
            WHERE s.active
              AND NOT EXISTS (
                  SELECT 1
                  FROM public.hitl_slick hs
                  WHERE hs.slick = s.id
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM not_oil_clses noc
                  WHERE noc.id = s.cls
              )
        ),
        matched_slicks AS (
            SELECT DISTINCT cs.id, cs.orchestrator_run
            FROM candidate_slicks cs
            JOIN public.slick s
              ON s.id = cs.id
            JOIN mask_geom mg
              ON mg.geom IS NOT NULL
             AND NOT ST_IsEmpty(mg.geom)
             AND ST_DWithin(
                    ST_Transform(s.geometry::geometry, %(mask_proj4)s),
                    mg.geom,
                    %(buffer_m)s
                 )
        ),
        updated AS (
            UPDATE public.slick s
            SET cls = %(sea_ice_cls_id)s
            FROM matched_slicks ms
            WHERE s.id = ms.id
              AND s.cls IS DISTINCT FROM %(sea_ice_cls_id)s
            RETURNING 1
        ),
        stamped AS (
            UPDATE public.orchestrator_run o
            SET sea_ice_date = %(mask_date)s
            FROM candidate_runs r
            WHERE o.id = r.id
              AND o.sea_ice_date IS DISTINCT FROM %(mask_date)s
            RETURNING 1
        )
        SELECT
            (SELECT COUNT(*) FROM updated) AS updated_slicks,
            (SELECT COUNT(*) FROM stamped) AS stamped_runs
    """
    params = {
        "not_oil_root_id": class_ids.not_oil_root_id,
        "scene_date": scene_date,
        "min_scene_lat": MASIE_MIN_SCENE_MAX_LAT,
        "buffer_m": buffer_m,
        "mask_geom_hex": mask_geom_hex,
        "mask_proj4": mask_proj4,
        "sea_ice_cls_id": class_ids.sea_ice_cls_id,
        "mask_date": mask_date,
    }

    with open_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            updated_slicks, stamped_runs = cur.fetchone()
        conn.commit()

    return updated_slicks, stamped_runs


def backfill_scene_dates_worker(
    worker_id: int,
    db_url: str,
    scene_dates: list[date],
    class_ids: ClassIds,
    earliest_supported_date: date,
    max_attempts: int,
    retry_backoff_seconds: float,
    buffer_m: float,
    sleep_sec: float,
) -> WorkerResult:
    mask_cache: dict[date, gpd.GeoDataFrame] = {}
    unavailable_mask_dates: set[date] = set()
    processed_dates = 0
    updated_slicks_total = 0
    stamped_runs_total = 0
    unresolved_dates = 0

    for scene_date in scene_dates:
        mask_date, mask_gdf = resolve_mask_for_scene_date(
            scene_date=scene_date,
            earliest_supported_date=earliest_supported_date,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            mask_cache=mask_cache,
            unavailable_mask_dates=unavailable_mask_dates,
        )

        if mask_date is None or mask_gdf is None:
            unresolved_dates += 1
            LOGGER.warning(
                "worker=%s scene_date=%s no MASIE mask found on or before date; leaving pending",
                worker_id,
                scene_date,
            )
            continue

        updated_slicks, stamped_runs = backfill_scene_date(
            db_url=db_url,
            scene_date=scene_date,
            mask_date=mask_date,
            mask_gdf=mask_gdf,
            class_ids=class_ids,
            buffer_m=buffer_m,
        )
        processed_dates += 1
        updated_slicks_total += updated_slicks
        stamped_runs_total += stamped_runs

        LOGGER.info(
            "worker=%s scene_date=%s mask_date=%s stamped_runs=%s updated_slicks=%s",
            worker_id,
            scene_date,
            mask_date,
            stamped_runs,
            updated_slicks,
        )
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return WorkerResult(
        processed_dates=processed_dates,
        updated_slicks=updated_slicks_total,
        stamped_runs=stamped_runs_total,
        unresolved_dates=unresolved_dates,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical SEA_ICE slick classifications by scene date."
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Postgres connection URL. Defaults to DB_URL from the environment.",
    )
    parser.add_argument(
        "--earliest-supported-date",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=date(2010, 1, 1),
        help="Earliest mask date to search, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--limit-scene-dates",
        type=int,
        default=None,
        help="Process at most this many pending scene dates, in ascending date order.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Retry attempts for retryable NOAA failures.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=0.5,
        help="Base exponential backoff for retryable NOAA failures.",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=0,
        help="Distance threshold, in meters, used to match slicks to the mask.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.05,
        help="Sleep between scene dates to smooth DB load.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers over disjoint scene-date slices.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    db_url = args.db_url or os.environ.get("DB_URL")
    if not db_url:
        raise SystemExit("Pass --db-url or set DB_URL in the environment.")

    class_ids = fetch_class_ids(db_url)
    scene_dates = fetch_pending_scene_dates(db_url, args.limit_scene_dates)
    if not scene_dates:
        LOGGER.info("No pending north-polar scene dates found.")
        return 0

    LOGGER.info("Found %s pending scene dates", len(scene_dates))
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    worker_scene_dates = [
        scene_dates[worker_index :: args.workers]
        for worker_index in range(args.workers)
    ]
    worker_scene_dates = [dates for dates in worker_scene_dates if dates]

    worker_results: list[WorkerResult] = []
    if len(worker_scene_dates) == 1:
        worker_results.append(
            backfill_scene_dates_worker(
                worker_id=1,
                db_url=db_url,
                scene_dates=worker_scene_dates[0],
                class_ids=class_ids,
                earliest_supported_date=args.earliest_supported_date,
                max_attempts=args.max_attempts,
                retry_backoff_seconds=args.retry_backoff_seconds,
                buffer_m=args.buffer_m,
                sleep_sec=args.sleep_sec,
            )
        )
    else:
        with ThreadPoolExecutor(max_workers=len(worker_scene_dates)) as executor:
            futures = [
                executor.submit(
                    backfill_scene_dates_worker,
                    worker_id,
                    db_url,
                    dates,
                    class_ids,
                    args.earliest_supported_date,
                    args.max_attempts,
                    args.retry_backoff_seconds,
                    args.buffer_m,
                    args.sleep_sec,
                )
                for worker_id, dates in enumerate(worker_scene_dates, start=1)
            ]
            for future in as_completed(futures):
                worker_results.append(future.result())

    processed_dates = sum(result.processed_dates for result in worker_results)
    updated_slicks_total = sum(result.updated_slicks for result in worker_results)
    stamped_runs_total = sum(result.stamped_runs for result in worker_results)
    unresolved_dates = sum(result.unresolved_dates for result in worker_results)

    LOGGER.info(
        "Finished: processed_dates=%s updated_slicks=%s stamped_runs=%s unresolved_dates=%s",
        processed_dates,
        updated_slicks_total,
        stamped_runs_total,
        unresolved_dates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
