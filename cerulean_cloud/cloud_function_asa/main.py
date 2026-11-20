"""cloud function Automatic Source Association handler"""

import asyncio
import json
import os
import random
from datetime import datetime, timedelta
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
from google.auth import compute_engine
from google.cloud import bigquery, bigquery_storage, storage
from google.oauth2 import id_token
from google.oauth2.service_account import Credentials
from queuer import add_to_asa_queue
from shapely import wkb
from sqlalchemy import and_, or_, select, update
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
# run: scenes for the 540 s timeout, capture dates for the GFW partitions scanned.
LIVE_LOOKBACK_DAYS = 32
MAX_BATCH_SCENES = 1000
MAX_BATCH_CAPTURE_DATES = 30

BATCH_AIS_SQL = """
WITH covered AS (
    SELECT scene_id, start_time, end_time, ST_GEOGFROMTEXT(envelope) AS geometry
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


def published_through(now):
    return now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=3)


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


def write_ais_slices(batches, bucket, prefix):
    """Write one Parquet file per scene from Arrow batches ordered by scene_id.

    Holds one scene's rows at a time and yields (scene_id, gs:// URI) after each upload.
    """

    def write(scene_id, parts):
        table = pa.Table.from_batches(parts).drop_columns(["scene_id"])
        table = table.filter(pc.is_valid(table["timestamp"]))  # drop coverage markers
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        blob = bucket.blob(f"{prefix}/{scene_id}.parquet")
        blob.upload_from_string(sink.getvalue().to_pybytes())
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


async def handle_asa_batch(lookback_days=LIVE_LOOKBACK_DAYS):
    if os.getenv("ASA_IS_DRY_RUN", "").lower() == "true":
        return "Success!"

    now = datetime.utcnow()
    live = now - timedelta(days=LIVE_LOOKBACK_DAYS)
    run = db.OrchestratorRun
    scene = db.Sentinel1Grd
    slick = db.Slick
    query = (
        select(scene, run)
        .join(run, run.sentinel1_grd == scene.id)
        .where(
            scene.start_time >= now - timedelta(days=lookback_days),
            scene.start_time <= published_through(now) - timedelta(hours=6),
            or_(
                *(
                    getattr(run, column).is_(None)
                    for flag, column in ASA_TIME_COLUMNS.items()
                    if flag != "DARK"
                ),
                # Older scenes whose only gap is DARK are not re-selected; they get
                # another DARK attempt whenever they are selected for another type.
                and_(run.asa_dark_time.is_(None), scene.start_time >= live),
            ),
            select(slick.id)
            .where(
                slick.orchestrator_run == run.id,
                slick.active.is_(True),
                slick.machine_confidence > 0,
            )
            .exists(),
        )
        .order_by(scene.start_time.desc())
        .limit(MAX_BATCH_SCENES)
    )
    candidates = {}
    capture_dates = set()
    engine = get_engine(db_url=os.getenv("DB_URL"))
    try:
        async with DatabaseClient(engine) as db_client:
            async with db_client.session.begin():
                for s1_scene, candidate_run in await db_client.session.execute(query):
                    capture_dates.add(s1_scene.start_time.date())
                    if len(capture_dates) > MAX_BATCH_CAPTURE_DATES:
                        break  # newest first, so every later row is older
                    _, flags = candidates.setdefault(
                        s1_scene.scene_id, (s1_scene, set())
                    )
                    flags.update(pending_run_flags(candidate_run))
    finally:
        await engine.dispose()

    # Queue each scene as soon as it is ready, so a timeout keeps earlier work.
    for scene_id, (_, flags) in candidates.items():
        if "VESSEL" not in flags:
            add_to_asa_queue(scene_id, run_flags=sorted(flags))

    vessel_analyzers = [
        ASA_MAPPING["VESSEL"](s1_scene)
        for s1_scene, flags in candidates.values()
        if "VESSEL" in flags
    ]
    if vessel_analyzers:
        scenes = [
            bigquery.StructQueryParameter(
                None,
                bigquery.ScalarQueryParameter(
                    "scene_id", "STRING", analyzer.s1_scene.scene_id
                ),
                bigquery.ScalarQueryParameter(
                    "start_time", "TIMESTAMP", analyzer.ais_start_time
                ),
                bigquery.ScalarQueryParameter(
                    "end_time", "TIMESTAMP", analyzer.ais_end_time
                ),
                bigquery.ScalarQueryParameter(
                    "envelope", "STRING", analyzer.ais_envelope.iloc[0].wkt
                ),
            )
            for analyzer in vessel_analyzers
        ]
        credentials = Credentials.from_service_account_info(
            json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        )
        time_filter, time_params = ais_time_filter(
            (analyzer.ais_start_time, analyzer.ais_end_time)
            for analyzer in vessel_analyzers
        )
        client = bigquery.Client(project=c.GFW_PROJECT_ID, credentials=credentials)
        rows = client.query(
            BATCH_AIS_SQL.format(time_filter=time_filter),
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("scenes", "STRUCT", scenes),
                    *time_params,
                ]
            ),
        ).result()
        # Row-by-row REST download measured 6.7k rows/s against ~17.8M rows a day;
        # the Storage Read API streams the ordered result as Arrow batches.
        batches = rows.to_arrow_iterable(
            bqstorage_client=bigquery_storage.BigQueryReadClient(
                credentials=credentials
            )
        )
        bucket = storage.Client(
            project=os.environ["PROJECT_ID"], credentials=compute_engine.Credentials()
        ).bucket(os.environ["ASA_BUCKET"])
        for scene_id, ais_uri in write_ais_slices(batches, bucket, f"ais/{now.date()}"):
            add_to_asa_queue(
                scene_id, run_flags=sorted(candidates[scene_id][1]), ais_uri=ais_uri
            )
    return "Success!"


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


def verify_scheduler_token(request):
    """Accept a batch request only with the ASA Scheduler account's OIDC token."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    try:
        claims = id_token.verify_oauth2_token(
            token,
            google.auth.transport.requests.Request(),
            audience=os.getenv("FUNCTION_URL"),
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
    if (request.get_json(silent=True) or {}).get("batch") is True:
        verify_scheduler_token(request)
    else:
        verify_api_key(request)
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
        return await handle_asa_batch(
            int(request_json.get("lookback_days", LIVE_LOOKBACK_DAYS))
        )
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
