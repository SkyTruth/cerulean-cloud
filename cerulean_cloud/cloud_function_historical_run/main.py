"""Discover Sentinel-1 scenes and submit tracked historical processing batches."""

import asyncio
import json
import os
import urllib.parse as urlparse
from datetime import date, datetime, timedelta, timezone

import asyncpg
import geojson
from flask import abort
from google.cloud import tasks_v2
from shapely.geometry import MultiPolygon, shape


# Cloud Functions imports main.py at the root of the deployed source archive;
# repository callers import it as part of cerulean_cloud.
if __package__:
    from .inventory import (
        list_sentinel1_scene_ids,
        filter_sentinel1_scene_ids,
        load_ocean_poly,
        region_from_geojson,
    )
else:
    from inventory import (
        list_sentinel1_scene_ids,
        filter_sentinel1_scene_ids,
        load_ocean_poly,
        region_from_geojson,
    )


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
    for key in ("dry_run", "manual_trigger", "apply_relevancy"):
        if key in request_json and not isinstance(request_json[key], bool):
            raise ValueError(f"{key} must be a boolean")
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
    if source not in ("eodag", "s3"):
        raise ValueError("source must be eodag or s3")
    try:
        if source == "s3":
            start = date.fromisoformat(request_json["start"])
            end = date.fromisoformat(request_json["end"])
        else:
            # Existing EODAG callers may supply full ISO timestamps, not just dates.
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
    if source == "eodag":
        if "apply_relevancy" in request_json:
            raise ValueError(
                "apply_relevancy applies to S3 discovery only; explicit scene_ids bypass filtering"
            )
        if not isinstance(request_json.get("geometry"), dict):
            raise ValueError(
                "EODAG search requires a GeoJSON FeatureCollection geometry"
            )
        if any(key in request_json for key in ("modes", "polarizations")):
            raise ValueError("modes and polarizations apply to S3 discovery only")
    else:
        if "geometry" in request_json:
            region_from_geojson(request_json["geometry"])
        for key in ("modes", "polarizations"):
            value = request_json.get(key)
            if value is not None and (
                not isinstance(value, list)
                or any(not isinstance(item, str) or not item for item in value)
            ):
                raise ValueError(f"{key} must be a list of strings or null")


def main(request):
    """List scenes or queue a tracked batch, preserving the stack dry-run default."""
    start_time = datetime.now()
    verify_api_key(request)
    request_json = request.get_json(silent=True)
    try:
        validate_request(request_json)
    except ValueError as exc:
        abort(400, description=str(exc))

    total_scenes, filtered_scenes = handle_search(request_json)
    # Submit each scene once, even when a catalog or supplied list repeats it.
    filtered_scenes = list(dict.fromkeys(filtered_scenes))
    summary = {
        "scene_count": total_scenes,
        "filtered_scene_count": len(filtered_scenes),
    }
    if request_json.get("action") == "list":
        return {**summary, "scene_ids": filtered_scenes}

    dry_run = request_json.get("dry_run", os.getenv("IS_DRY_RUN", "").lower() == "true")
    logs_url = make_cloud_function_logs_url(
        os.getenv("FUNCTIONNAME"), start_time, os.getenv("GCPPROJECT")
    )
    row = asyncio.run(
        add_trigger_row(total_scenes, len(filtered_scenes), logs_url=logs_url)
    )
    handler_queue(
        filtered_scenes,
        row["id"],
        dry_run=dry_run,
        manual_trigger=request_json.get("manual_trigger", False),
    )
    return {
        **summary,
        "trigger_id": row["id"],
        "queued_count": len(filtered_scenes),
        "dry_run": dry_run,
    }


def handle_search(request_json, ocean_poly=None):
    """Select explicit IDs, archive folders, or the existing spatial catalog search."""
    if "scene_ids" in request_json:
        scenes = [scene.strip() for scene in request_json["scene_ids"]]
        return len(scenes), list(dict.fromkeys(scenes))
    if request_json.get("source", "eodag") == "s3":
        scenes = list_sentinel1_scene_ids(
            date.fromisoformat(request_json["start"]),
            date.fromisoformat(request_json["end"]),
            modes=request_json.get("modes"),
            polarizations=request_json.get("polarizations"),
        )
        return len(scenes), filter_sentinel1_scene_ids(
            scenes,
            geometry=request_json.get("geometry"),
            ocean_poly=ocean_poly,
            apply_relevancy=request_json.get("apply_relevancy", True),
        )

    # Catalog dependencies are only needed for the existing EODAG search path.
    from eodag import EODataAccessGateway, setup_logging

    setup_logging(2)
    if ocean_poly is None:
        ocean_poly = load_ocean_poly()
    dag = EODataAccessGateway(
        user_conf_file_path=os.path.join(os.path.dirname(__file__), "geodes.yml")
    )
    dag.set_preferred_provider("geodes")

    fc = geojson.FeatureCollection(**request_json["geometry"])

    overall_geom = MultiPolygon([shape(f.get("geometry")) for f in fc.features])

    default_search_criteria = {
        "productType": "S1_SAR_GRD",
        "polarizationMode": "VV VH",
        "start": request_json["start"],
        "end": request_json["end"],
        "geom": overall_geom.wkt,
    }

    all_results = []
    for i, page_results in enumerate(
        dag.search_iter_page(**default_search_criteria, items_per_page=25)
    ):
        print(f"Got a hand on {len(page_results)} products on page {i + 1}")
        all_results.extend(page_results)

    search_results = all_results

    print(f"Got a hand on {len(search_results)} products.")

    len_total_scenes = len(search_results)

    filtered_scenes = []
    for result in search_results:
        print(f"Adding {result}...")
        is_oceanic = result.geometry.intersects(ocean_poly)
        if is_oceanic:
            filtered_scenes.append(result.properties.get("id"))

    return len_total_scenes, filtered_scenes


def handler_queue(filtered_scenes, trigger_id, *, dry_run=None, manual_trigger=False):
    """Queue scenes with their historical batch and the selected execution options."""
    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(
        os.environ["GCPPROJECT"], os.environ["GCPREGION"], os.environ["QUEUE"]
    )
    if dry_run is None:
        dry_run = os.getenv("IS_DRY_RUN", "").lower() == "true"
    for scene in filtered_scenes:
        payload = {"scene_id": scene, "trigger": trigger_id, "dry_run": dry_run}
        if manual_trigger:
            payload["manual_trigger"] = True
        task = {
            "dispatch_deadline": {"seconds": 1800},
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": os.environ["ORCHESTRATOR_URL"].rstrip("/") + "/orchestrate",
                "headers": {
                    "Content-type": "application/json",
                    "Authorization": f"Bearer {os.environ['API_KEY']}",
                },
                "body": json.dumps(payload).encode(),
            },
        }
        response = client.create_task(request={"parent": parent, "task": task})
        print(f"Created task {response.name}")
