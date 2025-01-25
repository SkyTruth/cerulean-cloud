"""Utility functions for cloud run orchestrator"""

import contextvars
import json
import logging
import sys
from typing import Any, Dict

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
