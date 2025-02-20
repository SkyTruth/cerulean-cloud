"""Utility functions for cloud run orchestrator"""

import contextvars
import datetime
import json
import logging
import re
import sys
from typing import Any, Dict

import pandas as pd
from google.cloud import logging as google_logging

# A ContextVar to store the context_dict for each request
context_dict_var: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "context_dict", default={}
)


class StructuredLogFilter(logging.Filter):
    """
    A logging filter that injects `context_dict` and `severity` into the log record.
    """

    def filter(self, record):
        """transform log record to structured log and inject context_dict and severity"""
        # Retrieve context_dict from the context variable
        context_dict = context_dict_var.get()

        # Determine if the log message is a dict or a string
        if isinstance(record.msg, dict):
            log_dict = record.msg
        else:
            # Treat any non-dict message as {"message": <msg>}
            log_dict = {"message": str(record.msg)}

        # Inject severity based on the logging level
        log_dict["severity"] = record.levelname

        # Inject any context if available
        if context_dict:
            log_dict.update(context_dict)

        # Convert the final log dict to a JSON string
        record.msg = json.dumps(log_dict)
        record.args = ()  # Clear args to prevent unwanted formatting

        return True


def configure_structured_logger(logger_name: str):
    """
    Configure the 'model' logger with StructuredLogFilter and a StreamHandler.
    Ensures only one handler and one filter are attached to prevent duplicate logs.
    """
    logger = logging.getLogger(logger_name)

    # Prevent adding multiple instances of the filter
    if not any(isinstance(f, StructuredLogFilter) for f in logger.filters):
        logger.addFilter(StructuredLogFilter())

    # Prevent adding multiple handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(message)s")  # Only output the message
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set the desired logging level
    logger.setLevel(logging.INFO)

    return logger


# Utils for interacting with CloudRun Logs


def format_date(date, format="%Y-%m-%d"):
    """
    Convert a date or date string into an ISO 8601-like format with millisecond precision.

    Args:
        date (str or datetime.date): The date to format, either as a string matching the given format
                                     or as a `datetime.date` object.
        format (str): The format of the input date string if `date` is a string. Defaults to "%Y-%m-%d".

    Returns:
        str: The formatted date string in "YYYY-MM-DDTHH:MM:SS.sssZ".

    """
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    return date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def jsonify_log_entries(entry):
    """
    Convert a log entry object into a dictionary with selected fields.

    Args:
        entry: A log entry object with attributes such as `timestamp`, `resource`, `labels`,
               `severity`, and `payload`.

    Returns:
        dict: A dictionary containing the log entry fields
    """
    return {
        "timestamp": entry.timestamp,
        "resource_revision_name": (
            entry.resource.labels["revision_name"]
            if entry.resource.labels is not None
            and "revision_name" in entry.resource.labels
            else None
        ),
        "instanceId": (
            entry.labels["instanceId"]
            if entry.labels is not None and "instanceId" in entry.labels
            else None
        ),
        "severity": entry.severity,
        "text_payload": entry.payload if isinstance(entry.payload, str) else None,
        "json_payload": entry.payload if isinstance(entry.payload, dict) else None,
        "labels": entry.labels,
        "resource_labels": entry.resource.labels,
    }


def log_query(
    service_name,
    log_name=None,
    revision_name=None,
    instance_id=None,
    start_time=None,
    end_time=None,
    textPayload=None,
    not_textPayload=None,
    jsonPayload={},
    severity=None,
    min_severity=None,
):
    """
    Construct a query string for filtering Google Cloud Logging entries.

    Args:
        service_name (str): The name of the Cloud Run service.
        revision_name (str, optional): The revision name to filter logs for a specific service revision.
        instance_id (str, optional): The instance ID to filter logs for a specific Cloud Run instance.
        start_time (datetime, optional): The start timestamp for log filtering.
        end_time (datetime, optional): The end timestamp for log filtering.
        textPayload (str, optional): A string to match in the text payload of the logs.
        not_textPayload (str, optional): A string to exclude from the text payload of the logs.
        jsonPayload (dict, optional): A dictionary of key-value pairs to match in the JSON payload of the logs.
        severity (str, optional): The exact severity level to filter logs (e.g., "ERROR").
        min_severity (str, optional): The minimum severity level to filter logs (e.g., "WARNING").

    Returns:
        str: A formatted query string for filtering logs.
    """

    query = f"""
        resource.type="cloud_run_revision"
        resource.labels.service_name="{service_name}"
    """

    if log_name:
        query += f'\n log_name="{log_name}"'
    if revision_name:
        query += f'\n resource.labels.revision_name="{revision_name}"'
    if instance_id:
        query += f'\n labels.instanceId="{instance_id}"'
    if start_time:
        query += f'\n timestamp >= "{format_date(start_time)}"'
    if end_time:
        query += f'\n timestamp <= "{format_date(end_time)}"'
    if textPayload:
        query += f'\n textPayload:"{textPayload}"'
    if not_textPayload:
        query += f'\n NOT textPayload:"{not_textPayload}"'
    if severity:
        query += f'\n severity="{severity}"'
    elif min_severity:
        severities = [
            "DEBUG",
            "INFO",
            "NOTICE",
            "WARNING",
            "ERROR",
            "CRITICAL",
            "ALERT",
            "EMERGENCY",
        ]
        min_index = severities.index(min_severity.upper())
        allowed_severities = severities[min_index:]
        query += (
            "\n("
            + " OR ".join([f'severity="{level}"' for level in allowed_severities])
            + ")"
        )
    for item in jsonPayload:
        query += f'\njsonPayload.{item}="{jsonPayload[item]}"'

    return query


def logs_to_list(logs):
    """
    Convert a collection of log entries into a list of dictionaries.

    Args:
        logs (iterable): An iterable of log entry objects to process.

    Returns:
        list: A list of dictionaries, where each dictionary represents a log entry.
    """

    log_entries = []
    for entry in logs:
        log_entries.append(jsonify_log_entries(entry))

    return log_entries


def query_logger(project_id, query, page_size=1000):
    """
    Execute a Google Cloud Logging query and return the results.

    Args:
        project_id (str): The ID of the Google Cloud project.
        query (str): The filter query string for retrieving logs.
        page_size (int, optional): The number of log entries to fetch per page. Defaults to 1000.

    Returns:
        pd.DataFrame or list: The log entries as a Pandas DataFrame
    """
    client = google_logging.Client(project=project_id)

    logs = client.list_entries(
        filter_=query, order_by=google_logging.DESCENDING, page_size=page_size
    )
    log_entries = pd.DataFrame(logs_to_list(logs))
    if "json_payload" in log_entries.columns:
        log_entries["scene_id"] = log_entries["json_payload"].apply(
            lambda x: x["scene_id"] if x is not None and "scene_id" in x else None
        )
        log_entries["message"] = log_entries["json_payload"].apply(
            lambda x: x["message"] if x is not None and "message" in x else None
        )
    return log_entries


def generate_log_file(
    log_df,
    filename="log.txt",
    print_vars=[
        "timestamp",
        "severity",
        "text_payload",
        "json_payload",
        "instanceId",
        "revision_name",
    ],
):
    """
    Generate a formatted log file from a DataFrame of log entries.

    Args:
        log_df (pd.DataFrame): A DataFrame containing log entries with columns such as
                               `timestamp`, `severity`, `text_payload`, `json_payload`,
                               and nested `labels` and `resource_labels`.
        filename (str, optional): The name of the file to save the logs to. Defaults to "log.txt".
        print_vars (list, optional): A list of log attributes to include in the file, such as
                                     `timestamp`, `severity`, and `json_payload`. Defaults to
                                     a predefined list of common attributes.

    Returns:
        None: Writes the formatted logs to the specified file.
    """

    log_str = ""
    for _, row in log_df.iterrows():
        timestamp = (
            f"{row['timestamp']} - "
            if "timestamp" in print_vars and row["timestamp"]
            else ""
        )
        severity = (
            f"{row['severity']}: "
            if "severity" in print_vars and row["severity"]
            else ""
        )
        text_payload = (
            f"{row['text_payload']} :: "
            if "text_payload" in print_vars and row["text_payload"]
            else ""
        )
        json_payload = (
            f"{json.dumps(row['json_payload'])} :: "
            if "json_payload" in print_vars and row["json_payload"]
            else ""
        )
        instanceId = (
            f"instanceId: {row['labels']['instanceId'][-7:]},  "
            if "instanceId" in print_vars and row["labels"] is not None
            else ""
        )
        revision_name = (
            f"revision_name: {row['resource_labels']['revision_name']}, "
            if "revision_name" in print_vars and row["resource_labels"]["revision_name"]
            else ""
        )

        log_str += (
            severity
            + timestamp
            + text_payload
            + json_payload
            + instanceId
            + revision_name
            + "\n\n"
        )

    with open(filename, "w") as file:
        file.write(log_str)


def get_scene_log_stats(project_id, service_name, revision_name, start_time, scene_id):
    """
    Retrieve and analyze log statistics for a specific scene from Google Cloud Logging.

    Args:
        project_id (str): The Google Cloud project ID.
        service_name (str): The name of the Cloud Run service.
        revision_name (str): The name of the Cloud Run revision.
        scene_id (str): The unique identifier of the scene to query logs for.

    Returns:
        pandas.DataFrame: A DataFrame containing all logs for the specified scene.

    """
    json_payload = {"scene_id": scene_id}

    query = log_query(
        service_name,
        revision_name=revision_name,
        start_time=start_time,
        jsonPayload=json_payload,
    )
    logs = query_logger(project_id, query)
    if logs.empty:
        print(
            f"There are no logs associated with {scene_id} for revision {revision_name}"
        )
        return logs

    logs["json_message"] = logs["json_payload"].apply(lambda x: x["message"])
    started = "unknown"
    try:
        start_time = logs[logs["json_message"] == "Initiating Orchestrator"].iloc[0][
            "json_payload"
        ]["start_time"]
        started = (
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        ).total_seconds() / 60
    except Exception:
        start_time = "unknown"
        started = "unknown"
    try:
        n_tiles_before_filter = logs[
            logs["json_message"] == "Removing invalid tiles (land filter)"
        ].iloc[0]["json_payload"]["n_tiles"]
    except Exception:
        n_tiles_before_filter = "unknown"
    try:
        n_tiles_after_filter = ", ".join(
            [
                str(log["json_payload"]["n_tiles"])
                for _, log in logs[
                    logs["json_message"] == "Starting parallel inference"
                ].iterrows()
            ]
        )
    except Exception:
        n_tiles_after_filter = "unknown"
    try:
        n_empty_images = len(logs[logs["json_message"] == "Empty image"])
    except Exception:
        n_empty_images = "unknown"
    try:
        n_images = len(logs[logs["json_message"] == "Generated image"])
    except Exception:
        n_images = "unknown"
    try:
        n_stale_slicks = logs[
            logs["json_message"] == "Deactivating slicks from stale runs."
        ].iloc[0]["json_payload"]["n_stale_slicks"]
    except Exception:
        n_stale_slicks = "unknown"
    try:
        n_slicks_before_filter = logs[
            logs["json_message"] == "Removing all slicks near land"
        ].iloc[0]["json_payload"]["n_features"]
    except Exception:
        n_slicks_before_filter = "unknown"
    try:
        n_slicks_after_filter = logs[
            logs["json_message"] == "Adding slicks to database"
        ].iloc[0]["json_payload"]["n_slicks"]
    except Exception:
        n_slicks_after_filter = "unknown"
    try:
        n_slicks_added = len(logs[logs["json_message"] == "Added slick"])
    except Exception:
        n_slicks_added = "unknown"
    try:
        end_time = logs[logs["json_message"] == "Orchestration complete!"].iloc[0][
            "json_payload"
        ]["timestamp"]
        dt = logs[logs["json_message"] == "Orchestration complete!"].iloc[0][
            "json_payload"
        ]["duration_minutes"]
        success = logs[logs["json_message"] == "Orchestration complete!"].iloc[0][
            "json_payload"
        ]["success"]
    except Exception:
        end_time, dt, success = "unknown", "unknown", "unknown"

    print(f"scene ID={scene_id}")
    print(f"Initiated Orchestrator at {start_time} - {started} minutes ago")
    print(
        f"{n_tiles_before_filter} tiles before filter; {n_tiles_after_filter} tiles after filter"
    )
    print(f"generated {n_images} images and {n_empty_images} empty images")
    print(f"deactivated {n_stale_slicks} stale slicks")
    print(
        f"{n_slicks_before_filter} slicks before filter; {n_slicks_after_filter} slicks after filter"
    )
    print(f"added {n_slicks_added} slicks")
    if success != "unknown":
        print(
            f"orchestration complete at {end_time}; in {round(dt * 100) / 100} minutes"
        )
        print(f"Success: {success}")
    else:
        print("Not complete")

    return logs


def get_latest_revision(project_id, service_name):
    """
    Retrieve the latest revision name for the given Cloud Run service.

    If no revision name is provided, this function queries the log entries
    for the specified service and determines the latest revision based on
    the revision name's numerical suffix.

    Args:
        project_id (str): The Google Cloud project ID.
        service_name (str): The name of the Cloud Run service.

    Returns:
        str: The latest revision name, or None if no revisions are found.
    """
    query = (
        'resource.type="cloud_run_revision" '
        f'resource.labels.service_name="{service_name}"'
    )
    client = google_logging.Client(project=project_id)
    entries = client.list_entries(
        filter_=query, order_by=google_logging.DESCENDING, page_size=100
    )

    revisions = set()
    for entry in entries:
        revision = entry.resource.labels.get("revision_name")
        if revision:
            revisions.add(revision)

    if not revisions:
        return None

    # Assuming revision names end with a numeric suffix, sort accordingly
    def revision_sort_key(rev):
        match = re.search(r"-(\d+)-", rev)
        if match:
            return int(match.group(1))
        else:
            return 0  # Default if no match

    latest_revision = sorted(revisions, key=revision_sort_key)[-1]
    return latest_revision
