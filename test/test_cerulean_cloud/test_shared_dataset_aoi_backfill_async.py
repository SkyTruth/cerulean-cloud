from cerulean_cloud.cloud_run_aoi_backfill import service


def test_ensure_https_url_upgrades_http_scheme():
    assert service.ensure_https_url("http://example.test/run/execute") == (
        "https://example.test/run/execute"
    )


def test_ensure_https_url_leaves_https_unchanged():
    assert service.ensure_https_url("https://example.test/run/execute") == (
        "https://example.test/run/execute"
    )


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


def test_get_run_context_includes_snapped_buffer(monkeypatch):
    monkeypatch.setattr(
        service,
        "query_one_row",
        lambda *args, **kwargs: (
            "mpa",
            "MPA",
            "",
            "maintenance.aoi_stage_mpa",
            "MRGID",
            "NAME",
            5000,
            12000.0,
        ),
    )

    context = service.get_run_context("postgres://db", "MPA")

    assert context.slick_to_aoi_buffer_m == 12000.0
    assert context.dataset_version == "latest"


def test_get_backfill_status_selects_snapped_buffer(monkeypatch):
    captured = {}

    monkeypatch.setattr(service, "get_db_url", lambda db_url=None: "postgres://db")
    monkeypatch.setattr(service, "resolve_asset_slug", lambda asset_slug, _: asset_slug)
    monkeypatch.setattr(
        service, "resolve_existing_shared_dataset_short_name", lambda *args: "MPA"
    )

    def fake_query_rows(db_url, sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return [
            (
                "MPA",
                "pending",
                4,
                1,
                3,
                0,
                0,
                100,
                50,
                20,
                5,
                10,
                2500.0,
                "2026-05-14T12:00:00Z",
            )
        ]

    monkeypatch.setattr(service, "query_rows", fake_query_rows)

    rows = service.get_backfill_status("mpa")

    assert rows[0][12] == 2500.0
    assert "r.slick_to_aoi_buffer_m" in captured["sql"]
    assert captured["params"] == ("MPA",)
