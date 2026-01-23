"""
Code for handling queue requests for Automatic Source Association
"""

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone

from google.api_core.exceptions import AlreadyExists
from google.auth import compute_engine
from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2


def _canonical_run_flags(run_flags):
    """Return a stable string representation for run_flags."""
    if not run_flags:
        return "ALL"
    return "-".join(sorted(set(run_flags)))


def _task_id(scene_id, run_flags, scheduled_date_utc):
    """Build a deterministic Cloud Tasks task_id for ASA.

    Task IDs must be unique within the queue. We dedupe by (scene_id, run_flags, scheduled day).
    """
    flags = _canonical_run_flags(run_flags)
    # Keep task_id short and safe by hashing the flags segment (in case flags grow).
    flags_hash = hashlib.sha256(flags.encode("utf-8")).hexdigest()[:10]
    yyyymmdd = scheduled_date_utc.strftime("%Y%m%d")
    return f"asa-{scene_id}-{flags_hash}-{yyyymmdd}"


def add_to_asa_queue(scene_id, run_flags=[], days_to_delay=0, task_suffix=None):
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

    if not project or not location or not queue or not url:
        raise ValueError(
            "Missing required env vars for Cloud Tasks enqueue: "
            "PROJECT_ID, GCPREGION, ASA_QUEUE, FUNCTION_URL"
        )

    # Construct the fully qualified queue name.
    parent = client.queue_path(project, location, queue)

    # Construct the request body.
    payload = {"scene_id": scene_id, "dry_run": dry_run}
    if run_flags:
        payload["run_flags"] = run_flags

    d = datetime.now(tz=timezone.utc) + timedelta(days=days_to_delay)
    task_id = _task_id(scene_id, run_flags, d.date())
    if task_suffix:
        suffix_hash = hashlib.sha256(str(task_suffix).encode("utf-8")).hexdigest()[:8]
        task_id = f"{task_id}-{suffix_hash}"
    task_name = client.task_path(project, location, queue, task_id)

    task = {
        "name": task_name,
        "http_request": {  # Specify the type of request.
            "http_method": tasks_v2.HttpMethod.POST,
            "url": url,  # The url path that the task will be sent to.
            "headers": {
                "Content-type": "application/json",
                "Authorization": f"Bearer {os.getenv('API_KEY')}",
            },
            "body": json.dumps(payload).encode(),
        },
    }

    # Create Timestamp protobuf.
    timestamp = timestamp_pb2.Timestamp()
    timestamp.FromDatetime(d)

    # Add the timestamp to the tasks.
    task["schedule_time"] = timestamp

    # Use the client to build and send the task.
    try:
        response = client.create_task(request={"parent": parent, "task": task})
    except AlreadyExists:
        print(f"Task already exists (deduped): {task_name}")
        return task_name

    print(f"Created task {response.name}")
    return response
