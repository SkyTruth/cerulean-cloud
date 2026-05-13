from cerulean_cloud.cloud_run_aoi_backfill import service


def test_submit_backfill_run_enqueues_resolved_short_name(monkeypatch):
    monkeypatch.setattr(
        service, "resolve_backfill_short_name", lambda *args, **kwargs: "MPA"
    )
    enqueued = {}

    def fake_enqueue(asset_slug, **kwargs):
        enqueued["asset_slug"] = asset_slug
        enqueued["kwargs"] = kwargs
        return "tasks/123"

    monkeypatch.setattr(service, "enqueue_backfill_run", fake_enqueue)

    short_name, task_name = service.submit_backfill_run(
        "mpa",
        target_url="https://example.test/run/execute",
    )

    assert short_name == "MPA"
    assert task_name == "tasks/123"
    assert enqueued["asset_slug"] == "mpa"
    assert enqueued["kwargs"]["short_name"] == "MPA"


def test_continue_backfill_run_enqueues_followup_when_pending(monkeypatch):
    calls = {"run_backfill": 0}

    monkeypatch.setattr(service, "get_db_url", lambda db_url=None: "postgres://db")
    monkeypatch.setattr(
        service, "resolve_backfill_short_name", lambda *args, **kwargs: "MPA"
    )

    def fake_run_backfill(*args, **kwargs):
        calls["run_backfill"] += 1

    monkeypatch.setattr(service, "run_backfill", fake_run_backfill)
    monkeypatch.setattr(
        service, "backfill_has_pending_work", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        service, "enqueue_backfill_run", lambda *args, **kwargs: "tasks/next"
    )

    short_name, task_name = service.continue_backfill_run(
        "mpa",
        target_url="https://example.test/run/execute",
    )

    assert calls["run_backfill"] == 1
    assert short_name == "MPA"
    assert task_name == "tasks/next"


def test_continue_backfill_run_stops_when_no_pending(monkeypatch):
    calls = {"run_backfill": 0, "enqueue": 0}

    monkeypatch.setattr(service, "get_db_url", lambda db_url=None: "postgres://db")
    monkeypatch.setattr(
        service, "resolve_backfill_short_name", lambda *args, **kwargs: "MPA"
    )

    def fake_run_backfill(*args, **kwargs):
        calls["run_backfill"] += 1

    def fake_enqueue(*args, **kwargs):
        calls["enqueue"] += 1
        return "tasks/should-not-run"

    monkeypatch.setattr(service, "run_backfill", fake_run_backfill)
    monkeypatch.setattr(
        service, "backfill_has_pending_work", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(service, "enqueue_backfill_run", fake_enqueue)

    short_name, task_name = service.continue_backfill_run(
        "mpa",
        target_url="https://example.test/run/execute",
    )

    assert calls["run_backfill"] == 1
    assert calls["enqueue"] == 0
    assert short_name == "MPA"
    assert task_name is None
