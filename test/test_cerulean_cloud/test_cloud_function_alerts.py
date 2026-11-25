import datetime
import importlib
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "last_attempt, status, expected_alert",
    [
        ("2026-11-24T05:00:00Z", {}, None),
        ("2026-11-24T09:00:05Z", {"code": 4}, "DEADLINE_EXCEEDED"),
        ("2026-11-23T09:00:05Z", {}, "hasn't run since 2026-11-23T09:00:05Z"),
    ],
    ids=["successful-batch", "failed-final-retry", "missed-batch"],
)
def test_daily_batch_alert(monkeypatch, last_attempt, status, expected_alert):
    monkeypatch.setenv("SLACK_ALERTS_WEBHOOK", "https://hooks.example/alerts")
    monkeypatch.setenv("TIPG_URL", "https://tipg.example")
    monkeypatch.setenv("ASA_SCHEDULER_JOB", "projects/p/locations/r/jobs/asa")
    alerts = importlib.reload(
        importlib.import_module("cerulean_cloud.cloud_function_alerts.main")
    )
    send = Mock()
    monkeypatch.setattr(alerts, "send_slack_alert", send)
    session = Mock()
    session.get.return_value.json.return_value = {
        "state": "ENABLED",
        "scheduleTime": "2026-11-25T05:00:04Z",
        "lastAttemptTime": last_attempt,
        "status": status,
    }

    now = datetime.datetime(2026, 11, 24, 9, 45, tzinfo=datetime.timezone.utc)
    alerts.check_asa_batch(session=session, now=now)

    if expected_alert is None:
        send.assert_not_called()
    else:
        send.assert_called_once()
        assert expected_alert in send.call_args.args[1]
