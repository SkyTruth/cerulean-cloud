"""cloud function scene relevancy handler
inspired by https://github.com/jonaraphael/ceruleanserver/tree/master/lambda/Machinable
"""

import asyncio
import json
import os

import asyncpg
import shapely.geometry as sh  # https://docs.aws.amazon.com/lambda/latest/dg/python-package.html
from google.cloud import tasks_v2


def load_ocean_poly(file_path="OceanGeoJSON_lowres.geojson"):
    """load ocean boundary polygon"""
    with open(file_path) as f:
        ocean_features = json.load(f)["features"]
    geom = sh.GeometryCollection(
        [sh.shape(feature["geometry"]).buffer(0) for feature in ocean_features]
    )[0]
    return geom


async def add_trigger_row(n_scenes=1, n_filtered_scenes=1):
    """get a row"""
    conn = await asyncpg.connect(os.getenv("DB_URL"))
    row = await conn.fetchrow("SELECT * FROM trigger")

    row = await conn.fetchrow(
        """
        INSERT INTO trigger(scene_count, filtered_scene_count, trigger_logs, trigger_type) VALUES($1, $2, $3, $4) RETURNING id
    """,
        n_scenes,
        n_filtered_scenes,
        "",
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

    request_json = request.get_json()
    print(request_json)
    ocean_poly = load_ocean_poly()

    scenes_count = len(request_json.get("Records"))
    filtered_scenes = handle_notification(request_json, ocean_poly=ocean_poly)
    filtered_scene_count = len(filtered_scenes)
    print(filtered_scenes)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    row = loop.run_until_complete(add_trigger_row(scenes_count, filtered_scene_count))
    print(row)

    handler_queue(filtered_scenes, row.id)

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
            # TODO add to queue
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

    # Construct the fully qualified queue name.
    parent = client.queue_path(project, location, queue)

    for scene in filtered_scenes:
        # Construct the request body.
        task = {
            "http_request": {  # Specify the type of request.
                "http_method": tasks_v2.HttpMethod.GET,
                "url": url,  # The full url path that the task will be sent to.
            }
        }

        payload = {"sceneid": scene, "trigger": trigger_id}
        # Add the payload to the request.
        if payload is not None:
            # The API expects a payload of type bytes.
            converted_payload = payload.encode()

            # Add the payload to the request.
            task["http_request"]["body"] = converted_payload

        # Use the client to build and send the task.
        response = client.create_task(request={"parent": parent, "task": task})

        print("Created task {}".format(response.name))
    return response
