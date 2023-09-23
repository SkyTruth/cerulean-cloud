"""cloud function AIS analysis handler
"""

import asyncio
import json
import os
from datetime import datetime

from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2

from cerulean_cloud.database_client import DatabaseClient, get_engine

from .utils.ais import AISConstructor
from .utils.associate import associate_ais_to_slicks, slicks_to_curves

# mypy: ignore-errors


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
          request handling to `handle_aaa_request`.
        - It's important to set up a new event loop if the function is running
          in a context where the default event loop is not available (e.g., in some WSGI servers).
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    res = loop.run_until_complete(handle_aaa_request(request))
    return res


async def handle_aaa_request(request):
    """
    Asynchronously handles the main logic of Automatic AIS Analysis (AAA) for a given scene.

    Args:
        request (flask.Request): The incoming HTTP request object containing scene information.

    Returns:
        str: A string "Success!" upon successful completion.

    Notes:
        - The function gets scene information from the request.
        - It uses the `DatabaseClient` for database operations.
    """
    request_json = request.get_json()
    if not request_json.get("dry_run"):
        scene_id = request_json.get("scene_id")
        db_engine = get_engine(db_url=os.getenv("DB_URL"))
        async with DatabaseClient(db_engine) as db_client:
            async with db_client.session.begin():
                s1 = await db_client.get_scene_from_id(scene_id)
                slicks_without_sources = (
                    await db_client.get_slicks_without_sources_from_scene_id(scene_id)
                )
                if len(slicks_without_sources) > 0:
                    ais_constructor = AISConstructor(s1)
                    ais_constructor.retrieve_ais()
                    if ais_constructor.ais_df:
                        ais_constructor.build_trajectories()
                        ais_constructor.buffer_trajectories()
                        for slick in slicks_without_sources:
                            automatic_ais_analysis(ais_constructor, slick)
    return "Success!"


def automatic_ais_analysis(ais_constructor, slick):
    """
    Perform automatic analysis to associate AIS trajectories with slicks.

    Parameters:
        ais_constructor (AISTrajectoryAnalysis): An instance of the AISTrajectoryAnalysis class.
        slick (GeoDataFrame): A GeoDataFrame containing the slick geometries.

    Returns:
        GroupBy object: The AIS-slick associations sorted and grouped by slick index.
    """
    slicks = slick.to_crs(ais_constructor.ais_df.estimate_utm_crs())
    slicks_clean, slicks_curves = slicks_to_curves(slicks)
    slick_ais = associate_ais_to_slicks(
        ais_constructor.ais_trajectories,
        ais_constructor.ais_buffered,
        ais_constructor.ais_weighted,
        slicks_clean,
        slicks_curves,
    )
    results = slick_ais.sort_values(
        ["slick_index", "slick_size", "total_score"], ascending=[True, False, False]
    ).groupby("slick_index")
    return results


def add_to_aaa_queue(scene_id):
    """
    Adds a new task to Google Cloud Tasks for automatic AIS analysis.

    Args:
        scene_id (str): The ID of the scene for which AIS analysis is needed.

    Returns:
        google.cloud.tasks_v2.types.Task: The created Task object.

    Notes:
        - The function uses Google Cloud Tasks API to schedule the AIS analysis.
        - Multiple retries are scheduled with different delays.
    """
    # Create a client.
    client = tasks_v2.CloudTasksClient()

    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION")
    queue = os.getenv("QUEUE")
    url = os.getenv("FUNCTION_URL")
    dry_run = bool(os.getenv("IS_DRY_RUN"))

    # Construct the fully qualified queue name.
    parent = client.queue_path(project, location, queue)

    # Construct the request body.
    payload = {"sceneid": scene_id, "dry_run": dry_run}

    task = {
        "http_request": {  # Specify the type of request.
            "http_method": tasks_v2.HttpMethod.POST,
            "url": url,  # The url path that the task will be sent to.
        },
        "headers": {
            "Content-type": "application/json",
            "Authorization": f"Bearer {os.getenv('API_KEY')}",
        },
        "body": json.dumps(payload).encode(),
    }

    # Number of days that the Automatic AIS Analysis should be run after
    # Each entry is another retry
    ais_delays = [0, 3, 7]  # TODO Magic number >>> Where should this live?
    for delay in ais_delays:
        d = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(
            days=delay
        )

        # Create Timestamp protobuf.
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(d)

        # Add the timestamp to the tasks.
        task["schedule_time"] = timestamp

        # Use the client to build and send the task.
        response = client.create_task(request={"parent": parent, "task": task})

        print("Created task {}".format(response.name))
    return response
