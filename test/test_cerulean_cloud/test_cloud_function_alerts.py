import datetime
import importlib
from unittest.mock import Mock

import pytest
import requests

JOB = "projects/p/locations/europe-west1/jobs/cerulean-cloud-prod-cf-asa-scheduler-1"
# The check runs at 19:45 UTC, after the 15:00 batch and its retries.
NOW = datetime.datetime(2026, 11, 24, 19, 45, tzinfo=datetime.timezone.utc)
NEXT_RUN = "2026-11-25T15:00:04.810843Z"


@pytest.fixture
def alerts(monkeypatch):
    monkeypatch.setenv("SLACK_ALERTS_WEBHOOK", "https://hooks.example/alerts")
    monkeypatch.setenv("TIPG_URL", "https://tipg.example")
    monkeypatch.setenv("ASA_SCHEDULER_JOB", JOB)
    module = importlib.reload(
        importlib.import_module("cerulean_cloud.cloud_function_alerts.main")
    )
    module.sent = []
    monkeypatch.setattr(
        module,
        "send_slack_alert",
        lambda webhook_url, message, dry_run=False: module.sent.append(message),
    )
    return module


def session_returning(job=None, error=None):
    response = Mock()
    response.json.return_value = job
    response.raise_for_status.side_effect = error
    session = Mock()
    session.get.return_value = response
    return session


def job(last=None, code=None, state="ENABLED", updated="2026-11-20T09:00:00Z"):
    return {
        "state": state,
        "scheduleTime": NEXT_RUN,
        "userUpdateTime": updated,
        **({"lastAttemptTime": last} if last else {}),
        "status": {"code": code} if code else {},
    }


@pytest.mark.parametrize(
    "state, alert",
    [
        # Today's attempt, a few seconds before the scheduled 15:00:04, succeeded.
        (job("2026-11-24T15:00:00.849635Z"), None),
        # A retry succeeded after the first attempt failed.
        (job("2026-11-24T16:00:05Z"), None),
        # The fourth retry, the last one, failed too.
        (job("2026-11-24T19:00:05Z", code=4), "DEADLINE_EXCEEDED"),
        (job("2026-11-24T19:00:05Z", code=99), "code 99"),
        # Yesterday's last retry is the last attempt: today's run never happened.
        (job("2026-11-23T19:00:05Z"), "hasn't run since 2026-11-23T19:00:05Z"),
        (job(), "hasn't run since it was created"),
        # Created after today's run time: its first run is tomorrow.
        (job(updated="2026-11-24T15:30:00Z"), None),
        # Staging's job stays paused.
        (job("2026-11-20T15:00:00Z", code=4, state="PAUSED"), None),
    ],
)
def test_check_asa_batch(alerts, state, alert):
    alerts.check_asa_batch(session=session_returning(state), now=NOW)
    if alert is None:
        assert alerts.sent == []
    else:
        assert len(alerts.sent) == 1 and alert in alerts.sent[0]


def test_check_asa_batch_reads_the_configured_job(alerts):
    session = session_returning(job("2026-11-24T15:00:00Z"))
    alerts.check_asa_batch(session=session, now=NOW)
    assert session.get.call_args.args[0] == (
        f"https://cloudscheduler.googleapis.com/v1/{JOB}"
    )


def test_check_asa_batch_alerts_when_the_job_is_unreadable(alerts):
    error = requests.exceptions.HTTPError("403 Forbidden")
    alerts.check_asa_batch(session=session_returning(error=error), now=NOW)
    assert len(alerts.sent) == 1 and "Couldn't read" in alerts.sent[0]


def test_main_runs_the_asa_check_only_when_asked(alerts, monkeypatch):
    calls = []
    monkeypatch.setattr(alerts, "check_asa_batch", lambda **_: calls.append("asa"))
    monkeypatch.setattr(
        alerts, "check_recent_slicks", lambda **_: calls.append("slicks")
    )
    monkeypatch.setattr(alerts, "make_response", lambda *args: args)
    for body in ({"check": "asa_batch"}, None):
        request = Mock()
        request.get_json.return_value = body
        alerts.main(request)
    assert calls == ["asa", "slicks"]
