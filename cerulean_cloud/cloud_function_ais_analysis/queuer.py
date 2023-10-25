"""
Code for handling queue requests for Automatic AIS Analysis
"""
import json
import os
from datetime import datetime, timedelta, timezone

from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2

# mypy: ignore-errors


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
        d = datetime.now(tz=timezone.utc) + timedelta(days=delay)

        # Create Timestamp protobuf.
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(d)

        # Add the timestamp to the tasks.
        task["schedule_time"] = timestamp

        # Use the client to build and send the task.
        response = client.create_task(request={"parent": parent, "task": task})

        print("Created task {}".format(response.name))
    return response
