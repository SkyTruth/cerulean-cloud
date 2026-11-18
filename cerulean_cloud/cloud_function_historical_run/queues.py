"""Stable Cloud Tasks queue IDs shared by deployment and historical clients."""

import re


def historical_queue_name(stage):
    name = f"orchestrator-backlog-{stage}"
    if not stage or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name):
        raise ValueError("Historical submissions require a valid Pulumi stack name")
    return name
