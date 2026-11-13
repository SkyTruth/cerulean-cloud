"""Resolve actual queue names instead of guessing Pulumi's physical names."""

from unittest.mock import MagicMock

import pytest

from cerulean_cloud.cloud_function_historical_run import local


@pytest.fixture
def session(monkeypatch):
    session = MagicMock()
    session.headers = {}
    session.__enter__.return_value = session
    monkeypatch.setattr(local, "AuthorizedSession", MagicMock(return_value=session))
    monkeypatch.setattr(
        local.google.auth, "default", MagicMock(return_value=(object(), "project"))
    )
    return session


def page(names, **extra):
    response = MagicMock()
    response.json.return_value = {
        "queues": [
            {"name": f"projects/project/locations/region/queues/{name}"}
            for name in names
        ],
        **extra,
    }
    return response


def test_queue_resolution_follows_pagination_and_keeps_physical_suffix(session):
    session.get.side_effect = [
        page(
            ["cerulean-cloud-test-queue-cr-orchestrator-d8ed51e"],
            nextPageToken="second",
        ),
        page(["cerulean-cloud-test-queue-cr-orchestrator-historical-670f5a2"]),
    ]
    assert (
        local.resolve_orchestrator_queue("project", "region", "test")
        == "cerulean-cloud-test-queue-cr-orchestrator-historical-670f5a2"
    )
    assert session.get.call_count == 2
    assert session.get.call_args.kwargs["params"]["pageToken"] == "second"
    assert session.headers["x-goog-user-project"] == "project"


@pytest.mark.parametrize(
    "names",
    [
        [],
        [
            "cerulean-cloud-test-queue-cr-orchestrator-historical-one",
            "cerulean-cloud-test-queue-cr-orchestrator-historical-two",
        ],
    ],
)
def test_queue_resolution_rejects_missing_or_ambiguous_targets(session, names):
    session.get.return_value = page(names)
    with pytest.raises(ValueError, match="Expected one orchestrator queue"):
        local.resolve_orchestrator_queue("project", "region", "test")
