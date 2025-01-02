"""Utility functions for cloud run orchestrator"""

import json


def structured_log(message, severity=None, **kwargs):
    """
    Create a structured log message in JSON format with severity.

    Args:
        message (str): The main log message.
        severity (str, optional): The severity level (e.g., "INFO", "ERROR").
        **kwargs: Additional log details.

    Returns:
        str: A JSON-formatted string.
    """
    log_data = {"message": message}
    if severity:
        log_data["severity"] = severity  # Add severity to the JSON payload
    log_data.update(kwargs)
    return json.dumps(log_data)
