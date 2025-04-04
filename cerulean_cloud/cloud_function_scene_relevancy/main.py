"""cloud function scene relevancy handler
inspired by https://github.com/jonaraphael/ceruleanserver/tree/master/lambda/Machinable
"""

import asyncio
import json
import os
import urllib.parse as urlparse
from datetime import datetime, timedelta
from typing import Optional

import asyncpg
import shapely.geometry as sh  # https://docs.aws.amazon.com/lambda/latest/dg/python-package.html
from flask import abort
from google.cloud import tasks_v2


def load_ocean_poly(file_path: Optional[str] = None):
    """load ocean boundary polygon"""

    if not file_path:
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "OceanGeoJSON_lowres.geojson"
        )

    with open(file_path) as f:
        ocean_features = json.load(f)["features"]
    geom = sh.GeometryCollection(
        [sh.shape(feature["geometry"]).buffer(0) for feature in ocean_features]
    ).geoms[0]
    return geom


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
    row = await conn.fetchrow("SELECT * FROM trigger")

    row = await conn.fetchrow(
        """
        INSERT INTO trigger(scene_count, filtered_scene_count, trigger_logs, trigger_type) VALUES($1, $2, $3, $4) RETURNING id
    """,
        n_scenes,
        n_filtered_scenes,
        logs_url,
        "SNS_TOPIC",
    )
    return row


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


def main(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    start_time = datetime.now()
    verify_api_key(request)

    request_json = request.get_json()
    print(request_json)
    ocean_poly = load_ocean_poly()

    scenes_count = len(request_json.get("Records"))
    filtered_scenes = handle_notification(request_json, ocean_poly=ocean_poly)
    filtered_scene_count = len(filtered_scenes)
    print(filtered_scenes)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logs_url = make_cloud_function_logs_url(
        os.getenv("FUNCTIONNAME"), start_time, os.getenv("GCPPROJECT")
    )
    print(logs_url)
    row = loop.run_until_complete(
        add_trigger_row(scenes_count, filtered_scene_count, logs_url=logs_url)
    )
    print(row)

    handler_queue(filtered_scenes, row["id"])

    return "Success!"


def handle_notification(request_json, ocean_poly):
    """handle notification"""
    filtered_scenes = []
    for r in request_json.get("Records"):
        sns = r["Sns"]
        msg = json.loads(sns["Message"])

        if not (msg["mode"] == "IW"):
            # Check Beam Mode
            # XXX This is workaround a bug in Titiler.get_bounds (404 not found) that fails if the beam mode is not IW
            continue
        if not (msg["resolution"] == "H"):
            # Check High Definition
            continue
        if not (msg["polarization"][1] == "V"):
            # Check Polarization
            # XXX This is hardcoded in the server, where we look for a vv.grd file
            continue
        scene_poly = sh.polygon.Polygon(msg["footprint"]["coordinates"][0][0])
        if not (scene_poly.intersects(ocean_poly)):
            # Check Oceanic
            continue
        filtered_scenes.append(msg["id"])
    return filtered_scenes


def handler_queue(filtered_scenes, trigger_id):
    """handler queue"""
    # Create a client.
    client = tasks_v2.CloudTasksClient()

    project = os.getenv("GCPPROJECT")
    queue = os.getenv("QUEUE")
    location = os.getenv("GCPREGION")
    url = os.getenv("ORCHESTRATOR_URL")
    dry_run = os.getenv("IS_DRY_RUN", "").lower() == "true"

    # Construct the fully qualified queue name.
    parent = client.queue_path(project, location, queue)

    for scene in filtered_scenes:
        # Construct the request body.
        # TODO: Add orchestrate and POST method instead
        task = {
            "http_request": {  # Specify the type of request.
                "http_method": tasks_v2.HttpMethod.POST,
                "url": urlparse.urljoin(
                    url, "orchestrate"
                ),  # The full url path that the task will be sent to.
            }
        }

        payload = {"scene_id": scene, "trigger": trigger_id, "dry_run": dry_run}
        print(payload)
        # Add the payload to the request.
        if payload is not None:
            if isinstance(payload, dict):
                # Convert dict to JSON string
                payload = json.dumps(payload)
                # specify http content-type to application/json
                task["http_request"]["headers"] = {
                    "Content-type": "application/json",
                    "Authorization": f"Bearer {os.getenv('API_KEY')}",
                }

            # The API expects a payload of type bytes.
            converted_payload = payload.encode()

            # Add the payload to the request.
            task["http_request"]["body"] = converted_payload

        # Use the client to build and send the task.
        response = client.create_task(request={"parent": parent, "task": task})

        print(f"Created task {response.name}")
