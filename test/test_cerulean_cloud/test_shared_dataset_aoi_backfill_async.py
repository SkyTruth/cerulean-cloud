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
        service, "require_existing_backfill_short_name", lambda *args, **kwargs: "MPA"
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
        service, "require_existing_backfill_short_name", lambda *args, **kwargs: "MPA"
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
        service, "require_existing_backfill_short_name", lambda *args, **kwargs: "MPA"
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
            0.001,
        ),
    )

    context = service.get_run_context("postgres://db", "MPA")

    assert context.slick_to_aoi_buffer_m == 12000.0
    assert context.simplify == 0.001
    assert context.dataset_version == "latest"


def test_run_backfill_passes_buffer_to_sub_batches(monkeypatch):
    monkeypatch.setattr(service, "get_db_url", lambda db_url=None: "postgres://db")
    monkeypatch.setattr(service, "resolve_asset_slug", lambda asset_slug, _: asset_slug)
    monkeypatch.setattr(
        service, "require_existing_backfill_short_name", lambda *args, **kwargs: "MPA"
    )
    monkeypatch.setattr(
        service,
        "get_run_context",
        lambda *args, **kwargs: service.RunContext(
            asset_slug="mpa",
            short_name="MPA",
            dataset_version="latest",
            stage_table="maintenance.aoi_stage_mpa",
            ext_id_field="MRGID",
            display_name_field="NAME",
            batch_size=5000,
            slick_to_aoi_buffer_m=12000.0,
            simplify=0.001,
        ),
    )
    monkeypatch.setattr(
        service, "get_catalog_asset", lambda *args, **kwargs: type("Asset", (), {})()
    )
    monkeypatch.setattr(
        service,
        "fetch_dataset_ref",
        lambda *args, **kwargs: type("Ref", (), {"cache_path": "/tmp/mpa.fgb"})(),
    )
    monkeypatch.setattr(service, "acquire_run_lock", lambda *args, **kwargs: object())
    monkeypatch.setattr(service, "release_run_lock", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        service,
        "claim_next_chunk",
        lambda *args, **kwargs: {
            "id": 7,
            "chunk_index": 1,
            "split_depth": 0,
            "bbox": (0.0, 0.0, 1.0, 1.0),
        },
    )
    monkeypatch.setattr(service, "load_chunk_gdf", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(service, "load_stage_table", lambda *args, **kwargs: 1)
    monkeypatch.setattr(service, "derive_catalog_citation", lambda *args, **kwargs: "")
    monkeypatch.setattr(service.time, "sleep", lambda *_args, **_kwargs: None)
    captured = {}

    def fake_process(*args, **kwargs):
        captured["slick_to_aoi_buffer_m"] = kwargs["slick_to_aoi_buffer_m"]
        return ("completed", 1, 1, 1, 0, 1, 1)

    monkeypatch.setattr(service, "process_chunk_sub_batches", fake_process)
    monkeypatch.setattr(service, "mark_chunk_completed", lambda *args, **kwargs: None)

    service.run_backfill("mpa", short_name="MPA", max_batches=1)

    assert captured["slick_to_aoi_buffer_m"] == 12000.0


def test_get_backfill_status_selects_snapped_buffer(monkeypatch):
    captured = {}

    monkeypatch.setattr(service, "get_db_url", lambda db_url=None: "postgres://db")
    monkeypatch.setattr(
        service, "require_existing_backfill_short_name", lambda *args, **kwargs: "MPA"
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


def test_require_existing_backfill_short_name_requires_explicit_short_name():
    try:
        service.require_existing_backfill_short_name("mpa")
    except ValueError as exc:
        assert "explicit short_name" in str(exc)
    else:
        raise AssertionError("Expected ValueError when short_name is omitted")


def test_require_existing_backfill_short_name_rejects_asset_mismatch(monkeypatch):
    monkeypatch.setattr(service, "get_db_url", lambda db_url=None: "postgres://db")
    monkeypatch.setattr(service, "resolve_asset_slug", lambda asset_slug, _: asset_slug)
    monkeypatch.setattr(
        service,
        "query_one_row",
        lambda *args, **kwargs: ("MPA", "SHARED_DATASET", "wdpa-marine"),
    )

    try:
        service.require_existing_backfill_short_name("coral", short_name="MPA")
    except ValueError as exc:
        assert "configured for asset_slug" in str(exc)
        assert "wdpa-marine" in str(exc)
        assert "coral" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched asset_slug")
