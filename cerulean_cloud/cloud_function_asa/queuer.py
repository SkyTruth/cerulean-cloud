"""
Code for handling queue requests for Automatic Source Association
"""

import json
import os
from datetime import datetime, timedelta, timezone

from google.auth import compute_engine
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2


def add_to_asa_queue(scene_id, run_flags=[], days_to_delay=0):
    """
    Adds a new task to Google Cloud Tasks for Automatic Source Association.

    Args:
        scene_id (str): The ID of the scene for which Automatic Source Association is needed.
        run_flags (list): A list of flags to run the task with.
        days_to_delay (int): The number of days to delay the task.
    Returns:
        google.cloud.tasks_v2.types.Task: The created Task object.

    Notes:
        - The function uses Google Cloud Tasks API to schedule the Automatic Source Association.
        - Multiple retries are scheduled with different delays.
    """
    # Create a client.
    # Use Compute Engine credentials (metadata server) to guarantee default ADC
    client = tasks_v2.CloudTasksClient(credentials=compute_engine.Credentials())

    project = os.getenv("PROJECT_ID")
    location = os.getenv("GCPREGION")
    queue = os.getenv("ASA_QUEUE")
    url = os.getenv("FUNCTION_URL")
    dry_run = os.getenv("ASA_IS_DRY_RUN", "").lower() == "true"

    # Construct the fully qualified queue name.
    parent = client.queue_path(project, location, queue)

    # Construct the request body.
    payload = {"scene_id": scene_id, "dry_run": dry_run}
    if run_flags:
        payload["run_flags"] = run_flags

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

    d = datetime.now(tz=timezone.utc) + timedelta(days=days_to_delay)

    # Create Timestamp protobuf.
    timestamp = timestamp_pb2.Timestamp()
    timestamp.FromDatetime(d)

    # Add the timestamp to the tasks.
    task["schedule_time"] = timestamp

    # Use the client to build and send the task.
    response = client.create_task(request={"parent": parent, "task": task})

    print(f"Created task {response.name}")
    return response
