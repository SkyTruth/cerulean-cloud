"""cloud function scene relevancy handler
inspired by https://github.com/jonaraphael/ceruleanserver/tree/master/lambda/Machinable
"""

import asyncio
import json
import os
import urllib.parse as urlparse
from datetime import datetime, timedelta

import asyncpg
import shapely.geometry as sh  # https://docs.aws.amazon.com/lambda/latest/dg/python-package.html
from google.cloud import tasks_v2


def load_ocean_poly(file_path="OceanGeoJSON_lowres.geojson"):
    """load ocean boundary polygon"""
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
        os.getenv("FUNCTION_NAME"), start_time, os.getenv("GCP_PROJECT")
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
        sns = request_json["Records"][0]["Sns"]
        msg = json.loads(sns["Message"])
        scene_poly = sh.polygon.Polygon(msg["footprint"]["coordinates"][0][0])

        is_highdef = "H" == msg["id"][10]
        is_vv = (
            "V" == msg["id"][15]
        )  # we don't want to process any polarization other than vv XXX This is hardcoded in the server, where we look for a vv.grd file
        is_oceanic = scene_poly.intersects(ocean_poly)
        print(is_highdef, is_vv, is_oceanic)
        if is_highdef and is_vv and is_oceanic:
            filtered_scenes.append(msg["id"])
    return filtered_scenes


def handler_queue(filtered_scenes, trigger_id):
    """handler queue"""
    # Create a client.
    client = tasks_v2.CloudTasksClient()

    project = os.getenv("GCP_PROJECT")
    queue = os.getenv("QUEUE")
    location = os.getenv("GCP_LOCATION")
    url = os.getenv("ORCHESTRATOR_URL")
    dry_run = bool(os.getenv("IS_DRY_RUN"))

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

        payload = {"sceneid": scene, "trigger": trigger_id, "dry_run": dry_run}
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

        print("Created task {}".format(response.name))
