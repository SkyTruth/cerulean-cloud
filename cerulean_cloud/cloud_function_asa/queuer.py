"""
Code for handling queue requests for Automatic Source Association
"""

import json
import os
from datetime import datetime, timedelta, timezone

from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2

# mypy: ignore-errors


def add_to_asa_queue(scene_id):
    """
    Adds a new task to Google Cloud Tasks for Automatic Source Association.

    Args:
        scene_id (str): The ID of the scene for which Automatic Source Association is needed.

    Returns:
        google.cloud.tasks_v2.types.Task: The created Task object.

    Notes:
        - The function uses Google Cloud Tasks API to schedule the Automatic Source Association.
        - Multiple retries are scheduled with different delays.
    """
    # Create a client.
    client = tasks_v2.CloudTasksClient()

    project = os.getenv("PROJECT_ID")
    location = os.getenv("GCPREGION")
    queue = os.getenv("ASA_QUEUE")
    url = os.getenv("FUNCTION_URL")
    dry_run = os.getenv("ASA_IS_DRY_RUN", "").lower() == "true"

    # Construct the fully qualified queue name.
    parent = client.queue_path(project, location, queue)

    # Construct the request body.
    payload = {"scene_id": scene_id, "dry_run": dry_run}

    task = {
        "http_request": {  # Specify the type of request.
            "http_method": tasks_v2.HttpMethod.POST,
            "url": url,  # The url path that the task will be sent to.
            "headers": {
                "Content-type": "application/json",
                "Authorization": f"Bearer {os.getenv('API_KEY')}",
            },
            "body": json.dumps(payload).encode(),
        }
    }

    # Number of days that the Automatic Source Association should be run after
    # Each entry is another retry
    asa_delays = [0, 3, 7]  # TODO Magic number >>> Where should this live?
    for delay in asa_delays:
        d = datetime.now(tz=timezone.utc) + timedelta(days=delay)

        # Create Timestamp protobuf.
        timestamp = timestamp_pb2.Timestamp()
        timestamp.FromDatetime(d)

        # Add the timestamp to the tasks.
        task["schedule_time"] = timestamp

        # Use the client to build and send the task.
        response = client.create_task(request={"parent": parent, "task": task})

        print(f"Created task {response.name}")
    return response
