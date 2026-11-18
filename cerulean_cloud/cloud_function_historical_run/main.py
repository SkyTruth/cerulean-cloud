"""Discover Sentinel-1 scenes and submit tracked historical processing batches."""

import asyncio
import base64
import json
import os
import time
import urllib.parse as urlparse
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import partial

import asyncpg
import geojson
import google.auth
from flask import abort
from google.auth.transport.requests import AuthorizedSession
from shapely.geometry import MultiPolygon, shape


CLOUD_TASKS_API = "https://cloudtasks.googleapis.com/v2"
# Leave headroom before the deployed function's 500-second timeout.
ENQUEUE_TIMEOUT_SECONDS = 450
BATCH_SIZE = 100
MAX_BATCH_WORKERS = 32
MAX_BATCH_HISTORY_CHECKS = 4


# Cloud Functions imports main.py at the root of the deployed source archive;
# repository callers import it as part of cerulean_cloud.
if __package__:
    from .queues import historical_queue_name
    from .relevance import load_ocean_poly
else:
    from queues import historical_queue_name
    from relevance import load_ocean_poly


def make_cloud_function_logs_url(
    function_name: str, start_time, project_id: str, duration=2
) -> str:
    """Forges a cloud log url given a service name, a date and a project id

    Args:
        function_name (str): the cloud run service name
        start_time (datetime.datetime): the start time of the cloud run
        project_id (str): a project id
        duration (int, optional): Default duration of the logs to show (in min). Defaults to 2.

    Returns:
        str: A cloud log url.
    """
    base_url = "https://console.cloud.google.com/logs/query;query="
    query = f'resource.type = "cloud_function" resource.labels.function_name = "{function_name}"'
    formatted_time = start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    end_time = (start_time + timedelta(minutes=duration)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    t_range = f";timeRange={formatted_time}%2F{end_time}"
    t = f";cursorTimestamp={formatted_time}"
    project_id = f"?project={project_id}"
    url = base_url + urlparse.quote(query) + t_range + t + project_id
    return url


async def add_trigger_row(n_scenes=1, n_filtered_scenes=1, logs_url=""):
    """get a row"""
    conn = await asyncpg.connect(os.getenv("DB_URL"))
    try:
        return await conn.fetchrow(
            """
            INSERT INTO trigger(scene_count, filtered_scene_count, trigger_logs, trigger_type)
            VALUES($1, $2, $3, $4) RETURNING id
            """,
            n_scenes,
            n_filtered_scenes,
            logs_url,
            "HISTORICAL_RUN",
        )
    finally:
        await conn.close()


async def get_processed_scene_ids(scene_ids):
    """Find scenes with any successful inference run in this stack's database."""
    if not scene_ids:
        return set()
    conn = await asyncpg.connect(os.environ["DB_URL"], timeout=30, command_timeout=30)
    try:
        rows = await conn.fetch(
            """
            SELECT s.scene_id
            FROM sentinel1_grd s
            WHERE s.scene_id = ANY($1::text[])
              AND EXISTS (
                  SELECT 1 FROM orchestrator_run r
                  WHERE r.sentinel1_grd = s.id AND r.success IS TRUE
              )
            """,
            scene_ids,
        )
        return {row["scene_id"] for row in rows}
    finally:
        await conn.close()


async def exclude_processed_scenes(scene_ids, *, overwrite):
    if overwrite:
        return scene_ids, 0
    processed = await get_processed_scene_ids(scene_ids)
    return [scene for scene in scene_ids if scene not in processed], len(processed)


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
    if len(parts) != 2 or parts[0].lower() != "bearer":
        abort(401, description="Unauthorized: Invalid Authorization header format")

    request_api_key = parts[1]

    # Compare the token part with your expected API key
    if request_api_key != expected_api_key:
        abort(401, description="Unauthorized: Invalid API key")


def validate_request(request_json):
    """Reject ambiguous selectors and invalid options before creating a batch."""
    if not isinstance(request_json, dict):
        raise ValueError("Expected a JSON object")
    if request_json.get("action", "enqueue") not in ("list", "enqueue"):
        raise ValueError("action must be list or enqueue")
    for key in ("dry_run", "manual_trigger", "overwrite"):
        if key in request_json and not isinstance(request_json[key], bool):
            raise ValueError(f"{key} must be a boolean")
    for key in ("polarizations", "apply_relevancy"):
        if key in request_json:
            raise ValueError(f"{key} is not supported")
    if "modes" in request_json:
        modes = request_json["modes"]
        if (
            not isinstance(modes, list)
            or not modes
            or any(mode not in ("EW", "IW", "SM") for mode in modes)
        ):
            raise ValueError("modes must be a nonempty list containing EW, IW, or SM")
    if "scene_ids" in request_json:
        scenes = request_json["scene_ids"]
        if not isinstance(scenes, list) or any(
            not isinstance(scene, str) or not scene.strip() for scene in scenes
        ):
            raise ValueError("scene_ids must be a list of nonempty strings")
        if any(
            key in request_json
            for key in (
                "source",
                "start",
                "end",
                "geometry",
                "modes",
                "polarizations",
                "apply_relevancy",
            )
        ):
            raise ValueError("scene_ids cannot be combined with search options")
        return
    source = request_json.get("source", "eodag")
    if source != "eodag":
        raise ValueError("source must be eodag")
    try:
        start = datetime.fromisoformat(request_json["start"])
        end = datetime.fromisoformat(request_json["end"])
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "start and end must be ISO dates (or EODAG timestamps)"
        ) from exc
    if end < start:
        raise ValueError("end must be on or after start")
    if request_json.get("geometry") is not None and not isinstance(
        request_json["geometry"], dict
    ):
        raise ValueError("geometry must be a GeoJSON FeatureCollection when provided")


def main(request):
    """List scenes or queue a tracked batch, defaulting to dry-run processing."""
    start_time = datetime.now()
    enqueue_deadline = time.monotonic() + ENQUEUE_TIMEOUT_SECONDS
    verify_api_key(request)
    request_json = request.get_json(silent=True)
    try:
        validate_request(request_json)
    except ValueError as exc:
        abort(400, description=str(exc))

    logs_url = make_cloud_function_logs_url(
        os.getenv("FUNCTIONNAME"), start_time, os.getenv("GCPPROJECT")
    )
    return asyncio.run(
        run_historical(request_json, logs_url=logs_url, deadline=enqueue_deadline)
    )


async def run_historical(request_json, *, logs_url="", deadline=None, trigger_id=None):
    """Run selection, DB exclusion, tracking, and queueing locally or over HTTP.

    Notebook callers can await this directly. Without an HTTP deadline,
    discovery has no overall time limit and queue submission gets its own budget.
    Local runs leave trigger_logs empty unless the caller supplies a log URL.
    A supplied trigger_id shares one trigger across a caller's batches.
    """
    validate_request(request_json)
    list_only = request_json.get("action") == "list"
    if not list_only:
        queue_name = historical_queue_name(os.getenv("HISTORICAL_RUN_STAGE"))
    total_scenes, filtered_scenes = handle_search(request_json)
    # Submit each scene once, even when a catalog or supplied list repeats it.
    filtered_scenes = list(dict.fromkeys(filtered_scenes))
    filtered_scenes, already_processed_scene_count = await exclude_processed_scenes(
        filtered_scenes, overwrite=request_json.get("overwrite", False)
    )
    summary = {
        "scene_count": total_scenes,
        "filtered_scene_count": len(filtered_scenes),
        "already_processed_scene_count": already_processed_scene_count,
    }
    print(f"Historical scene selection: {summary}")
    if list_only:
        return {**summary, "scene_ids": filtered_scenes}

    dry_run = request_json.get("dry_run", True)
    if trigger_id is None:
        row = await add_trigger_row(
            total_scenes, len(filtered_scenes), logs_url=logs_url
        )
        trigger_id = row["id"]
    batches = await enqueue_batches(
        filtered_scenes,
        trigger_id,
        queue_name=queue_name,
        dry_run=dry_run,
        manual_trigger=request_json.get("manual_trigger", False),
        deadline=deadline,
    )
    return {
        **summary,
        "trigger_id": trigger_id,
        "queued_count": sum(batch["queued_count"] for batch in batches),
        "dry_run": dry_run,
    }


def handle_search(request_json, ocean_poly=None):
    """Select explicit IDs or discover scenes through EODAG's GEODES catalog."""
    if "scene_ids" in request_json:
        scenes = [scene.strip() for scene in request_json["scene_ids"]]
        return len(scenes), list(dict.fromkeys(scenes))
    # Explicit IDs do not require the catalog dependencies.
    if __package__:
        from .geodes_catalog import search_catalog
    else:
        from geodes_catalog import search_catalog

    if ocean_poly is None:
        ocean_poly = load_ocean_poly()

    default_search_criteria = {
        "productType": "S1_SAR_GRD",
        # Accept VV in either single (SV) or dual (DV) polarization products.
        "polarizationMode": json.dumps(["VV", "VV VH"]),
        "start": request_json["start"],
        "end": request_json["end"],
    }
    if request_json.get("geometry") is not None:
        fc = geojson.FeatureCollection(**request_json["geometry"])
        overall_geom = MultiPolygon([shape(f.get("geometry")) for f in fc.features])
        default_search_criteria["geom"] = overall_geom.wkt
    if request_json.get("modes"):
        # EODAG 2.5 interpolates strings into the provider's JSON query template.
        default_search_criteria["sensorMode"] = json.dumps(
            list(dict.fromkeys(request_json["modes"]))
        )

    return search_catalog(default_search_criteria, ocean_poly)


def batch_create_tasks(session, parent, task_requests, *, deadline, batch_number):
    """Call Google's batchCreate REST API and wait for its operation to finish."""
    request_id = str(uuid.uuid4())

    def request_timeout():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Cloud Tasks batchCreate timed out (request_id={request_id})"
            )
        return min(30, remaining)

    print(
        f"Submitting batch {batch_number}: {len(task_requests)} tasks (request_id={request_id})",
        flush=True,
    )
    timeout = request_timeout()
    response = session.post(
        f"{CLOUD_TASKS_API}/{parent}/tasks:batchCreate",
        json={"requests": task_requests, "requestId": request_id},
        timeout=timeout,
        max_allowed_time=timeout,
    )
    response.raise_for_status()
    operation = response.json()
    print(
        f"Batch {batch_number} Cloud Tasks operation: {operation['name']}", flush=True
    )
    while not operation.get("done", False):
        time.sleep(min(1, request_timeout()))
        timeout = request_timeout()
        response = session.get(
            f"{CLOUD_TASKS_API}/{operation['name']}",
            timeout=timeout,
            max_allowed_time=timeout,
        )
        response.raise_for_status()
        operation = response.json()
    return operation


def cloud_tasks_session():
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return AuthorizedSession(credentials)


async def enqueue_batches(
    scene_ids,
    trigger_id,
    *,
    queue_name,
    dry_run=True,
    manual_trigger=False,
    deadline=None,
    recheck_history=False,
):
    """Bound concurrent submits, drain started operations, and return results in input order."""
    if not scene_ids:
        return []
    workers = int(os.getenv("HISTORICAL_BATCH_WORKERS", str(MAX_BATCH_WORKERS)))
    if not 1 <= workers <= MAX_BATCH_WORKERS:
        raise ValueError(
            f"HISTORICAL_BATCH_WORKERS must be from 1 to {MAX_BATCH_WORKERS}"
        )
    starts = range(0, len(scene_ids), BATCH_SIZE)
    workers = min(workers, len(starts))
    jobs = iter(enumerate(starts, 1))
    results = {}
    stopping = False
    first_error = None
    history_slots = asyncio.Semaphore(MAX_BATCH_HISTORY_CHECKS)
    loop = asyncio.get_running_loop()

    def in_thread(function, *args, **kwargs):
        return loop.run_in_executor(pool, partial(function, *args, **kwargs))

    async def worker():
        nonlocal stopping, first_error
        session = None
        try:
            while not stopping:
                job = next(jobs, None)
                if job is None:
                    return
                number, start = job
                scenes = scene_ids[start : start + BATCH_SIZE]
                try:
                    eligible, excluded = scenes, 0
                    if recheck_history:
                        async with history_slots:
                            if stopping:
                                return
                            eligible, excluded = await exclude_processed_scenes(
                                scenes, overwrite=False
                            )
                    if stopping:
                        return
                    queued = 0
                    if eligible:
                        if session is None:
                            session = await in_thread(cloud_tasks_session)
                        if stopping:
                            return
                        queued = await in_thread(
                            handler_queue,
                            eligible,
                            trigger_id,
                            session=session,
                            queue_name=queue_name,
                            batch_number=number,
                            dry_run=dry_run,
                            manual_trigger=manual_trigger,
                            deadline=deadline,
                        )
                    result = {
                        "scene_count": len(scenes),
                        "filtered_scene_count": len(eligible),
                        "already_processed_scene_count": excluded,
                        "trigger_id": trigger_id,
                        "queued_count": queued,
                        "dry_run": dry_run,
                    }
                    results[number] = result
                    print(f"Batch {number} confirmed: {json.dumps(result)}", flush=True)
                except Exception as exc:
                    stopping = True
                    if first_error is None:
                        first_error = exc
                    print(f"Batch {number} failed ({type(exc).__name__})", flush=True)
                    return
        finally:
            if session is not None:
                session.close()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = asyncio.gather(*(worker() for _ in range(workers)))
        try:
            await asyncio.shield(active)
        except asyncio.CancelledError:
            # A thread that started a submit may still be confirming its operation.
            stopping = True
            await active
            raise
    if first_error is not None:
        raise first_error
    return [results[number] for number in sorted(results)]


def handler_queue(
    filtered_scenes,
    trigger_id,
    *,
    session,
    queue_name,
    batch_number,
    dry_run=True,
    manual_trigger=False,
    deadline=None,
):
    """Submit one batch and verify that every requested task was created."""
    if deadline is None:
        deadline = time.monotonic() + ENQUEUE_TIMEOUT_SECONDS
    parent = (
        f"projects/{os.environ['GCPPROJECT']}/locations/{os.environ['GCPREGION']}"
        f"/queues/{queue_name}"
    )
    task_requests = []
    for scene in filtered_scenes:
        payload = {"scene_id": scene, "trigger": trigger_id, "dry_run": dry_run}
        if manual_trigger:
            payload["manual_trigger"] = True
        task = {
            "dispatchDeadline": "1800s",
            "httpRequest": {
                "httpMethod": "POST",
                "url": os.environ["ORCHESTRATOR_URL"].rstrip("/") + "/orchestrate",
                "headers": {
                    "Content-type": "application/json",
                    "Authorization": f"Bearer {os.environ['API_KEY']}",
                },
                # REST represents protobuf bytes as a base64 string.
                "body": base64.b64encode(json.dumps(payload).encode()).decode(),
            },
        }
        task_requests.append({"parent": parent, "task": task})
    operation = batch_create_tasks(
        session, parent, task_requests, deadline=deadline, batch_number=batch_number
    )
    created = operation.get("response", {}).get("tasks", [])
    failed_requests = operation.get("metadata", {}).get("failedRequests", {})
    if "error" in operation or failed_requests or len(created) != len(task_requests):
        raise RuntimeError(
            f"Cloud Tasks batchCreate batch {batch_number} ({operation['name']}) incomplete: "
            f"{len(created)}/{len(task_requests)} tasks confirmed in this batch; "
            f"error={operation.get('error')}; failed_requests={failed_requests}"
        )
    return len(created)
