"""cloud function Automatic Source Association handler"""

import asyncio
import json
import os
import random
import time
from datetime import date, datetime, timedelta, timezone
from itertools import groupby

import geopandas as gpd
import google.auth.transport.requests
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from flask import abort
from geometric_slick_potential import (
    predict_geometric_slick_potential,
)
from google.api_core.exceptions import GoogleAPIError, NotFound, PreconditionFailed
from google.auth import compute_engine
from google.cloud import bigquery, bigquery_storage, storage
from google.oauth2 import id_token
from google.oauth2.service_account import Credentials
from queuer import add_to_asa_queue
from shapely import wkb
from sqlalchemy import case, func, or_, select, update
from utils import constants as c
from utils.analyzer import ASA_MAPPING

import cerulean_cloud.database_schema as db
from cerulean_cloud.centerlines import calculate_centerlines
from cerulean_cloud.database_client import DatabaseClient, get_engine, update_object


# NATURAL is a placeholder analyzer; track it here once it is implemented.
ASA_TIME_COLUMNS = {
    flag: f"asa_{flag.lower()}_time" for flag in ASA_MAPPING if flag != "NATURAL"
}

# A batch request may set lookback_days to reach older scenes. These caps bound one
# run: runs for the 1800 s timeout, AIS dates for the GFW partitions scanned
# (~10.7 GB per date plus ~28 GB per query), with a byte ceiling as a backstop.
LIVE_LOOKBACK_DAYS = 32
ASA_BATCH_LOCK_KEY = 0x415341  # "ASA", shared by batch producers in this database
MAX_BATCH_RUNS = 1000
MAX_BATCH_AIS_DATES = 30
MAX_BATCH_BYTES_BILLED = 10**12
# A run queued less than this long ago may still have a task in Cloud Tasks
# (3 attempts within 600 s, each up to 1800 s), so automatic retries wait.
REQUEUE_AFTER = timedelta(hours=1)

# make_valid repairs a self-intersecting envelope instead of failing the query.
BATCH_AIS_SQL = """
WITH covered AS (
    SELECT
        scene_id,
        start_time,
        end_time,
        ST_GEOGFROMTEXT(envelope, make_valid => TRUE) AS geometry
    FROM UNNEST(@scenes)
    WHERE DATE(end_time) <= (
        SELECT MAX(date)
        FROM `global-fishing-watch.pipe_ais_v5_published.segs_activity_daily`
        WHERE date != '1979-01-01'
    )
), matches AS (
    SELECT
        scene.scene_id,
        seg.ssvid,
        seg.timestamp,
        seg.lon,
        seg.lat,
        seg.course,
        seg.speed_knots,
        ves.ais_identity.shipname_mostcommon.value AS shipname,
        ves.ais_identity.shiptype[SAFE_OFFSET(0)].value AS shiptype,
        ves.best.best_flag AS flag,
        ves.random_forest_info.best_vessel_class AS best_shiptype
    FROM `global-fishing-watch.pipe_ais_v5_published.messages` AS seg
    JOIN covered AS scene
      ON seg.timestamp BETWEEN scene.start_time AND scene.end_time
      AND ST_COVEREDBY(ST_GEOGPOINT(seg.lon, seg.lat), scene.geometry)
    LEFT JOIN `global-fishing-watch.pipe_ais_identity_v5_partners_published.vi_ssvid` AS ves
      ON seg.ssvid = ves.ssvid
    WHERE {time_filter}
)
SELECT scene.scene_id, matches.* EXCEPT (scene_id)
FROM covered AS scene
LEFT JOIN matches USING (scene_id)
ORDER BY scene_id
"""


# ASA advances its AIS eligibility cutoff at 12:00 UTC, through the day three days earlier.
# scripts/backfill_asa_completion_times*.sql repeat published_through() in SQL.
GFW_RELEASE_HOUR = 12


def published_through(now):
    """End of the newest AIS day GFW has published by `now` (naive UTC)."""
    release_day = (now - timedelta(hours=GFW_RELEASE_HOUR)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return release_day - timedelta(days=2)


def batch_cutoff(now):
    """Latest capture time whose whole AIS window GFW has published by `now`."""
    return published_through(now) - timedelta(hours=c.HOURS_AFTER)


def batch_options(request_json, now):
    """Check a batch request's options; HTTP 400 on a bad one.

    lookback_days: whole days back from `now`, LIVE_LOOKBACK_DAYS by default. More
        makes a one-off run, which also takes a pivot.
    pivot: a UTC date, "YYYY-MM-DD". The run takes scenes from its midnight onward
        in capture order, then wraps around to the oldest. Random by default.
    retry_incomplete: true immediately requeues eligible unfinished work, even
        if an earlier task is still queued or running.
    """
    lookback_days = request_json.get("lookback_days", LIVE_LOOKBACK_DAYS)
    if type(lookback_days) is not int or lookback_days < 1:
        abort(400, description="lookback_days must be a positive whole number")
    pivot = request_json.get("pivot")
    retry_incomplete = request_json.get("retry_incomplete", False)
    if not isinstance(retry_incomplete, bool):
        abort(400, description="retry_incomplete must be true or false")
    if lookback_days <= LIVE_LOOKBACK_DAYS and pivot is not None:
        abort(
            400,
            description=f"pivot needs lookback_days > {LIVE_LOOKBACK_DAYS}",
        )
    if pivot is not None:
        try:
            pivot = datetime.combine(date.fromisoformat(pivot), datetime.min.time())
        except (TypeError, ValueError):
            abort(400, description='pivot must be a UTC date, "YYYY-MM-DD"')
        first = (now - timedelta(days=lookback_days)).date()
        last = batch_cutoff(now).date()
        if not first <= pivot.date() <= last:
            abort(400, description=f"pivot must fall from {first} to {last}")
    return lookback_days, pivot, retry_incomplete


def batch_query(now, lookback_days, pivot=None, retry_incomplete=False):
    """(scene, run) pairs with ASA work left, in the order a batch takes them."""
    run = db.OrchestratorRun
    scene = db.Sentinel1Grd
    slick = db.Slick
    vessel_pending = run.asa_vessel_time.is_(None)
    queued = run.asa_batch_queued_at
    start = now - timedelta(days=lookback_days)
    cutoff = batch_cutoff(now)
    if lookback_days > LIVE_LOOKBACK_DAYS:
        # One-off runs: work no batch has queued (with retry_incomplete, also work
        # queued before and still incomplete), in capture order from the pivot,
        # wrapping around, so scenes that fail every time can't use up the caps on
        # every run.
        if pivot is None:
            pivot = start + (cutoff - start) * random.random()
        order = (scene.start_time < pivot, scene.start_time, run.id)
        queue_filter = queued.is_(None)
    else:
        # Daily runs: VESSEL work first, runs never queued before retries, each
        # oldest first, so a backlog clears before it ages out; the rest at random,
        # so retries that never succeed (e.g. scenes GFW SAR never covers) can't
        # hold the same slots every day.
        order = (
            vessel_pending.desc(),
            case((vessel_pending, queued.is_not(None))),
            case((vessel_pending, scene.start_time)),
            func.random(),
        )
        queue_filter = or_(queued.is_(None), queued < now - REQUEUE_AFTER)
    return (
        select(scene, run)
        .join(run, run.sentinel1_grd == scene.id)
        .where(
            scene.start_time >= start,
            scene.start_time < cutoff,
            retry_incomplete or queue_filter,
            or_(
                *(
                    getattr(run, column).is_(None)
                    for column in ASA_TIME_COLUMNS.values()
                )
            ),
            select(slick.id)
            .where(
                slick.orchestrator_run == run.id,
                slick.active.is_(True),
                slick.machine_confidence > 0,
            )
            .exists(),
        )
        .order_by(*order)
        .limit(MAX_BATCH_RUNS)
    )


async def queue_scene(
    session, scene_id, flags, run_ids, ais_uri=None, *, retry_incomplete=False
):
    """Queue one scene's task, and stamp its runs only if a task was created.

    Task names carry the run IDs, so a re-orchestrated run isn't deduped against an
    earlier run's task from the same day.
    """
    task = add_to_asa_queue(
        scene_id,
        run_flags=sorted(flags),
        ais_uri=ais_uri,
        task_suffix=sorted(run_ids),
        force_retry=retry_incomplete,
    )
    if task is not None:
        async with session.begin():
            await session.execute(
                update(db.OrchestratorRun)
                .where(db.OrchestratorRun.id.in_(run_ids))
                .values(asa_batch_queued_at=datetime.utcnow())
            )


def pending_run_flags(run):
    return [
        flag
        for flag, column in ASA_TIME_COLUMNS.items()
        if getattr(run, column) is None
    ]


def completed_run_flags(run_flags, analyzer_cache):
    return [
        flag
        for flag in run_flags
        if getattr(analyzer_cache.get(flag), "data_is_available", None) is not False
    ]


def ais_time_filter(windows):
    """OR one BETWEEN per stretch of overlapping (start, end) AIS windows.

    GFW's messages table prunes partitions per range. A single range from the
    oldest to the newest scene would scan every day in between.
    """
    ranges = []
    for start, end in sorted(windows):
        if ranges and start <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
    sql = " OR ".join(
        f"seg.timestamp BETWEEN @t{i}_start AND @t{i}_end" for i in range(len(ranges))
    )
    params = [
        bigquery.ScalarQueryParameter(f"t{i}_{edge}", "TIMESTAMP", value)
        for i, bounds in enumerate(ranges)
        for edge, value in zip(("start", "end"), bounds)
    ]
    return f"({sql})", params


def cached_ais_slices(bucket, object_names):
    """Find candidate scenes in the seven-day cache, including run-date paths."""
    cached = {}
    for blob in bucket.list_blobs(
        prefix="ais/",
        match_glob="ais/????-??-??/*.parquet",
        fields="items(name,generation,timeCreated),nextPageToken",
    ):
        scene_id = blob.name.rsplit("/", 1)[1].removesuffix(".parquet")
        if scene_id in object_names:
            previous = cached.get(scene_id)
            if previous is None or previous.name != object_names[scene_id]:
                # GCS lists names in order; prefer the scene-date copy, then newest run date.
                cached[scene_id] = blob

    # Leave a full day for queued workers before the bucket's seven-day deletion.
    fresh_since = datetime.now(timezone.utc) - timedelta(days=6)
    uris = {}
    for scene_id, blob in cached.items():
        if blob.time_created <= fresh_since:
            name = object_names[scene_id]
            try:
                # Renew the stored bytes in GCS without running the GFW query again.
                blob = bucket.copy_blob(
                    blob,
                    bucket,
                    name,
                    if_source_generation_match=blob.generation,
                    if_generation_match=blob.generation if blob.name == name else 0,
                )
            except (NotFound, PreconditionFailed):
                # Expired after listing, or another batch refreshed it first.
                blob = bucket.get_blob(name)
                if blob is None:
                    continue
            if blob.time_created <= fresh_since:
                raise GoogleAPIError(f"Could not refresh AIS slice {name}")
        uris[scene_id] = f"gs://{bucket.name}/{blob.name}"
    return uris


def write_ais_slices(batches, bucket, object_names):
    """Write one Parquet file per scene from Arrow batches ordered by scene_id.

    Holds one scene's rows at a time and yields (scene_id, gs:// URI) after each
    upload, or (scene_id, None) if that scene's file failed, so the rest go on.
    """

    def write(scene_id, parts):
        try:
            table = pa.Table.from_batches(parts).drop_columns(["scene_id"])
            table = table.filter(pc.is_valid(table["timestamp"]))  # coverage markers
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink)
            blob = bucket.blob(object_names[scene_id])
            try:
                blob.upload_from_string(
                    sink.getvalue().to_pybytes(), if_generation_match=0
                )
            except PreconditionFailed:
                pass  # An overlapping batch already created this scene's file.
        except Exception as e:
            print(f"Scene {scene_id}: writing its AIS slice failed: {e!r}")
            return scene_id, None
        return scene_id, f"gs://{bucket.name}/{blob.name}"

    current, parts = None, []
    for batch in batches:
        start = 0
        for scene_id, run in groupby(batch.column("scene_id").to_pylist()):
            length = sum(1 for _ in run)
            if parts and scene_id != current:
                yield write(current, parts)
                parts = []
            current = scene_id
            parts.append(batch.slice(start, length))
            start += length
    if parts:
        yield write(current, parts)


async def handle_asa_batch(
    lookback_days=LIVE_LOOKBACK_DAYS, pivot=None, retry_incomplete=False
):
    if os.getenv("ASA_IS_DRY_RUN", "").lower() == "true":
        return "Success!"

    now = datetime.utcnow()
    engine = get_engine(db_url=os.getenv("DB_URL"))
    try:
        # Pin the session to one connection: the lock must survive queue_batch's
        # per-scene commits. Disposing this invocation's engine releases it.
        async with engine.connect() as conn, DatabaseClient(conn) as db_client:
            async with db_client.session.begin():
                acquired = await db_client.session.scalar(
                    select(func.pg_try_advisory_lock(ASA_BATCH_LOCK_KEY))
                )
            if not acquired:
                if lookback_days <= LIVE_LOOKBACK_DAYS:
                    abort(503, description="Another ASA batch is running")
                print("Skipping backfill tick: another ASA batch is running")
                return "Skipped: another ASA batch is running"
            failed = await queue_batch(
                db_client.session,
                batch_query(now, lookback_days, pivot, retry_incomplete),
                retry_incomplete=retry_incomplete,
            )
    finally:
        await engine.dispose()
    if failed:
        raise RuntimeError(f"{len(failed)} scenes skipped after errors: {failed}")
    return "Success!"


async def queue_batch(session, query, *, retry_incomplete=False):
    """Queue a task for each scene the query selects; return the scenes that failed.

    A scene that raises is logged and skipped, so it can't stop the others.
    """
    async with session.begin():
        rows = (await session.execute(query)).all()

    object_names = {
        scene.scene_id: f"ais/{scene.start_time.date()}/{scene.scene_id}.parquet"
        for scene, run in rows
        if run.asa_vessel_time is None
    }
    cached = {}
    lookup_error = None
    if object_names:
        bucket = storage.Client(
            project=os.environ["PROJECT_ID"], credentials=compute_engine.Credentials()
        ).bucket(os.environ["ASA_BUCKET"])
        try:
            cached = cached_ais_slices(bucket, object_names)
        except GoogleAPIError as e:
            # Queue runs with no VESSEL work left without spending on a blind query.
            lookup_error = e

    candidates = {}
    ais_dates = set()
    for s1_scene, candidate_run in rows:
        if candidate_run.asa_vessel_time is None and s1_scene.scene_id not in cached:
            if lookup_error is not None:
                continue
            scene_dates = {
                (s1_scene.start_time + timedelta(hours=hours)).date()
                for hours in (-c.HOURS_BEFORE, c.HOURS_AFTER)
            }
            if len(ais_dates | scene_dates) > MAX_BATCH_AIS_DATES:
                continue
            ais_dates |= scene_dates
        _, flags, run_ids = candidates.setdefault(
            s1_scene.scene_id, (s1_scene, set(), set())
        )
        flags.update(pending_run_flags(candidate_run))
        run_ids.add(candidate_run.id)

    failed = []

    def skip(scene_id, step, error):
        print(f"Scene {scene_id}: {step} failed, skipping it: {error!r}")
        failed.append(scene_id)

    # Queue each scene as soon as it is ready, so a timeout keeps earlier work.
    searched, scenes, windows = [], [], []
    for scene_id, (s1_scene, flags, run_ids) in candidates.items():
        if "VESSEL" not in flags or scene_id in cached:
            try:
                await queue_scene(
                    session,
                    scene_id,
                    flags,
                    run_ids,
                    ais_uri=cached.get(scene_id),
                    retry_incomplete=retry_incomplete,
                )
            except Exception as e:
                skip(scene_id, "queueing", e)
            continue
        try:
            analyzer = ASA_MAPPING["VESSEL"](s1_scene)
            envelope = analyzer.ais_envelope.iloc[0].wkt
        except Exception as e:
            skip(s1_scene.scene_id, "building its AIS search area", e)
            continue
        searched.append(s1_scene.scene_id)
        scenes.append(
            bigquery.StructQueryParameter(
                None,
                bigquery.ScalarQueryParameter("scene_id", "STRING", s1_scene.scene_id),
                bigquery.ScalarQueryParameter(
                    "start_time", "TIMESTAMP", analyzer.ais_start_time
                ),
                bigquery.ScalarQueryParameter(
                    "end_time", "TIMESTAMP", analyzer.ais_end_time
                ),
                bigquery.ScalarQueryParameter("envelope", "STRING", envelope),
            )
        )
        windows.append((analyzer.ais_start_time, analyzer.ais_end_time))
    if lookup_error is not None:
        raise lookup_error
    if not scenes:
        return failed

    credentials = Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    )
    time_filter, time_params = ais_time_filter(windows)
    client = bigquery.Client(project=c.GFW_PROJECT_ID, credentials=credentials)
    job = client.query(
        BATCH_AIS_SQL.format(time_filter=time_filter),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("scenes", "STRUCT", scenes),
                *time_params,
            ],
            maximum_bytes_billed=MAX_BATCH_BYTES_BILLED,
        ),
    )
    job.result()
    # Row-by-row REST download measured 6.7k rows/s against ~17.8M rows a day,
    # so the Storage Read API streams the result as Arrow batches. GFW's account
    # can't open read sessions in GFW's project, so the session opens in ours;
    # one stream keeps the ORDER BY.
    batches = (
        bigquery.Client(project=os.environ["PROJECT_ID"], credentials=credentials)
        .list_rows(job.destination)
        .to_arrow_iterable(
            bqstorage_client=bigquery_storage.BigQueryReadClient(
                credentials=credentials
            ),
            max_stream_count=1,
        )
    )
    sliced = set()
    for scene_id, ais_uri in write_ais_slices(batches, bucket, object_names):
        sliced.add(scene_id)
        if ais_uri is None:
            failed.append(scene_id)  # write_ais_slices logged why
            continue
        _, flags, run_ids = candidates[scene_id]
        try:
            await queue_scene(
                session,
                scene_id,
                flags,
                run_ids,
                ais_uri=ais_uri,
                retry_incomplete=retry_incomplete,
            )
        except Exception as e:
            skip(scene_id, "queueing", e)
    withheld = [scene_id for scene_id in searched if scene_id not in sliced]
    if withheld:
        # The query's `covered` step drops scenes ending after GFW's newest
        # published day. They stay unqueued, so a later batch takes them.
        print(f"{len(withheld)} scenes wait for GFW to publish their AIS: {withheld}")
    return failed


def verify_api_key(request):
    """Function to verify API key"""
    expected_api_key = os.getenv("API_KEY")
    auth_header = request.headers.get("Authorization")

    # Check if the Authorization header is present
    if not auth_header:
        abort(401, description="Unauthorized: No Authorization header")

    # Split the header into 'Bearer' and the token part
    parts = auth_header.split()

    # Check if the header is formed correctly
    if parts[0].lower() != "bearer" or len(parts) != 2:
        abort(401, description="Unauthorized: Invalid Authorization header format")

    request_api_key = parts[1]

    # Compare the token part with your expected API key
    if request_api_key != expected_api_key:
        abort(401, description="Unauthorized: Invalid API key")


_google_certs = {}


def _cached_certs_request(url, method="GET", **kwargs):
    """Fetch Google's ID-token certificates at most hourly per instance.

    Rejected batch requests then cost no outbound call.
    """
    expires, response = _google_certs.get(url, (0, None))
    if time.time() > expires:
        response = google.auth.transport.requests.Request()(
            url, method=method, **kwargs
        )
        if response.status == 200:
            _google_certs[url] = (time.time() + 3600, response)
    return response


def verify_scheduler_token(request):
    """Accept a batch request only with the ASA Scheduler account's OIDC token."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not token:
        abort(401, description="Unauthorized: batch runs need a Scheduler token")
    try:
        claims = id_token.verify_oauth2_token(
            token, _cached_certs_request, audience=os.getenv("FUNCTION_URL")
        )
    except ValueError:
        abort(401, description="Unauthorized: batch runs need a Scheduler token")
    if claims.get("email") != os.getenv("ASA_SCHEDULER_SA"):
        abort(403, description="Forbidden: batch runs come from Cloud Scheduler")


def main(request):
    """
    Entry point for handling HTTP requests.

    Args:
        request (flask.Request): The incoming HTTP request object.

    Returns:
        Any: The response object or any value that can be converted to a
        Response object using Flask's `make_response`.

    Notes:
        - This function sets up an asyncio event loop and delegates the actual
          request handling to `handle_asa_request`.
        - It's important to set up a new event loop if the function is running
          in a context where the default event loop is not available (e.g., in some WSGI servers).
    """
    # Authenticate from the header before reading the body. The API key admits
    # scene requests; only the Scheduler's OIDC token admits a batch.
    batch = request.headers.get("Authorization") != f"Bearer {os.getenv('API_KEY')}"
    if batch:
        verify_scheduler_token(request)
    else:
        verify_api_key(request)
    request_json = request.get_json(silent=True)
    if (
        not isinstance(request_json, dict)
        or (request_json.get("batch") is True) != batch
    ):
        abort(
            400, description="Batch runs need the Scheduler token, scenes the API key"
        )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        res = loop.run_until_complete(handle_asa_request(request))
        return res
    finally:
        # Ensure all async generators finish and close the loop properly
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


async def handle_asa_request(request):
    """
    Asynchronously handles the main logic of Automatic Source Analysis (ASA) for a given scene.

    Args:
        request (flask.Request): The incoming HTTP request object containing scene information.

    Returns:
        str: A string "Success!" upon successful completion.
    """
    request_json = request.get_json()
    if request_json.get("batch") is True:
        return await handle_asa_batch(*batch_options(request_json, datetime.utcnow()))
    if not request_json.get("dry_run"):
        scene_id = request_json.get("scene_id")
        run_flags = request_json.get("run_flags")  # list of source_type short_names
        if run_flags is None:
            run_flags = list(ASA_MAPPING.keys())
        elif any(run_flag not in ASA_MAPPING.keys() for run_flag in run_flags):
            raise ValueError(
                f"Invalid run_flag provided. {run_flags} not in {ASA_MAPPING.keys()}"
            )

        db_engine = get_engine(db_url=os.getenv("DB_URL"))
        async with DatabaseClient(db_engine) as db_client:
            async with db_client.session.begin():
                s1_scene = await db_client.get_scene_from_id(scene_id)
                slicks = await db_client.get_slicks_from_scene_id(scene_id)

            await db_client.session.close()
            if s1_scene and s1_scene.start_time + timedelta(
                hours=c.HOURS_AFTER
            ) >= published_through(datetime.utcnow()):
                run_flags = [flag for flag in run_flags if flag != "DARK"]
            print(f"Running ASA ({run_flags}) on scene_id: {scene_id}")
            print(f"{len(slicks)} slicks in scene {scene_id}: {[s.id for s in slicks]}")
            if len(slicks) > 0:
                analyzer_cache = {}
                random.shuffle(slicks)  # Allows rerunning a scene to skip bugs
                for slick in slicks:
                    # Convert slick geometry to GeoDataFrame
                    slick_geom = wkb.loads(str(slick.geometry)).buffer(0)
                    slick_gdf = gpd.GeoDataFrame(
                        {"geometry": [slick_geom], "centerlines": [slick.centerlines]},
                        crs="4326",
                    )
                    if slick.centerlines is None:
                        crs_meters = slick_gdf.estimate_utm_crs(datum_name="WGS 84")
                        slick.centerlines, slick.aspect_ratio_factor = (
                            calculate_centerlines(slick_gdf, crs_meters)
                        )
                        slick_gdf["centerlines"] = [slick.centerlines]
                        async with db_client.session.begin():
                            await update_object(
                                db_client.session,
                                db.Slick,
                                filter_kwargs={"id": slick.id},
                                update_kwargs={
                                    "centerlines": slick.centerlines,
                                    "aspect_ratio_factor": slick.aspect_ratio_factor,
                                },
                            )
                        await db_client.session.close()

                    if slick.geometric_slick_potential is None:
                        print(
                            f"Computing geometric slick potential for slick {slick.id}"
                        )
                        slick.geometric_slick_potential = (
                            predict_geometric_slick_potential(slick)
                        )
                        async with db_client.session.begin():
                            await update_object(
                                db_client.session,
                                db.Slick,
                                filter_kwargs={"id": slick.id},
                                update_kwargs={
                                    "geometric_slick_potential": slick.geometric_slick_potential,
                                },
                            )
                        await db_client.session.close()

                    # Determine analyzers_to_run
                    skip_analyzers = (
                        []
                        if slick.aspect_ratio_factor > 0.35
                        # Calculated to get rid of as many false positives as possible without removing any true positives
                        else ["VESSEL", "DARK"]
                    )

                    if skip_analyzers:
                        print(
                            f"Skipping analyzers: {skip_analyzers} for slick {slick.id} because aspect ratio is too low: {slick.aspect_ratio_factor}"
                        )
                    analyzer_names_to_run = [
                        flag for flag in run_flags if flag not in skip_analyzers
                    ]
                    analyzers_to_run = []
                    for analyzer_name in analyzer_names_to_run:
                        analyzer = analyzer_cache.get(analyzer_name)
                        if analyzer is None:
                            analyzer = ASA_MAPPING[analyzer_name](
                                s1_scene, ais_uri=request_json.get("ais_uri")
                            )
                            analyzer_cache[analyzer_name] = analyzer
                        analyzers_to_run.append(analyzer)

                    # Run the analyzers
                    fresh_ranked_sources = pd.DataFrame()
                    for analyzer in analyzers_to_run:
                        res = analyzer.compute_coincidence_scores(slick_gdf)
                        fresh_ranked_sources = pd.concat(
                            [fresh_ranked_sources, res], ignore_index=True
                        )
                    print(
                        f"{len(fresh_ranked_sources)} sources found for Slick ID: {slick.id}, after running these analyzers:{[analyzer.short_name for analyzer in analyzers_to_run]}"
                    )

                    analyzer_names_to_run = completed_run_flags(
                        analyzer_names_to_run, analyzer_cache
                    )
                    if analyzer_names_to_run:
                        async with db_client.session.begin():
                            await db_client.lock_slick(slick.id)

                            # Read retained rows after taking the slick lock so reranking
                            # reflects the latest committed state for this slick.
                            current_collated_scores = (
                                await db_client.get_id_collated_score_pairs(slick.id)
                            )
                            old_ranked_sources = pd.DataFrame(
                                [
                                    (slick_to_source_id, collated_score)
                                    for slick_to_source_id, collated_score, source_type_short_name in current_collated_scores
                                    if source_type_short_name
                                    not in analyzer_names_to_run
                                ],
                                columns=["slick_to_source_id", "collated_score"],
                            )

                            # Deactivate old sources for analyzers that were run
                            print(
                                f"Deactivating sources for slick {slick.id}: {analyzer_names_to_run}"
                            )
                            await db_client.deactivate_sources_for_slick_by_source_type(
                                slick.id, analyzer_names_to_run
                            )

                            # Begin reranking logic
                            combined_df = pd.concat(
                                [old_ranked_sources, fresh_ranked_sources],
                                ignore_index=True,
                            )
                            combined_df = combined_df.sort_values(
                                "collated_score", ascending=False
                            ).reset_index(drop=True)
                            combined_df["rank"] = combined_df.index + 1

                            # Might want to increase this to save the top 3 per Source Type
                            only_record_top = 2 * len(ASA_MAPPING)

                            print(f"Adding sources for slick {slick.id}")
                            for idx, source_row in combined_df.iterrows():
                                if pd.isna(source_row["slick_to_source_id"]):
                                    # Insert slick to source association
                                    if idx < only_record_top:
                                        source = (
                                            await db_client.get_or_insert_ranked_source(
                                                source_row
                                            )
                                        )

                                        await db_client.insert_slick_to_source(
                                            source=source.id,
                                            slick=slick.id,
                                            active=True,
                                            git_hash=os.getenv("GIT_HASH"),
                                            git_tag=os.getenv("GIT_TAG"),
                                            coincidence_score=source_row[
                                                "coincidence_score"
                                            ],
                                            collated_score=source_row["collated_score"],
                                            rank=source_row["rank"],
                                            geojson_fc=source_row.get("geojson_fc"),
                                            geometry=(
                                                source_row["geometry"]
                                                if isinstance(
                                                    source_row["geometry"], str
                                                )
                                                else source_row["geometry"].wkt
                                                # XXX HACK TODO Need to figure out WHY this is a string 95% of the time, but then sometimes a shapely.point and sometimes a shapely.linestring
                                            ),
                                        )
                                else:
                                    await db_client.update_slick_to_source(
                                        filter_kwargs={
                                            "id": source_row["slick_to_source_id"]
                                        },
                                        update_kwargs={
                                            "rank": source_row["rank"],
                                            "active": idx < only_record_top,
                                        },
                                    )
                        await db_client.session.close()
                    else:
                        print(f"No analyzers to run or reranking for slick {slick.id}")
                    print(f"ASA complete for slick {slick.id}")
                completed_at = datetime.utcnow()
                completion_times = {
                    ASA_TIME_COLUMNS[flag]: completed_at
                    for flag in completed_run_flags(run_flags, analyzer_cache)
                    if flag in ASA_TIME_COLUMNS
                    and (flag != "VESSEL" or request_json.get("ais_uri"))
                }
                if completion_times:
                    async with db_client.session.begin():
                        await db_client.session.execute(
                            update(db.OrchestratorRun)
                            .where(
                                db.OrchestratorRun.id.in_(
                                    {slick.orchestrator_run for slick in slicks}
                                )
                            )
                            .values(**completion_times)
                        )
                print(f"ASA completed for {len(slicks)} slicks in scene {scene_id}")

        # Dispose the engine after finishing all DB operations.
        await db_engine.dispose()

    return "Success!"
