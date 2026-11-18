"""Historical-run diagnostics: outcome counting, verdicts, peaks, and report wiring."""

import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from scripts import historical_run_diagnostics as diag

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("override", [None, "orchestrator-backlog"])
def test_cli_defaults_to_stage_queue_and_can_inspect_the_shared_queue(
    monkeypatch, override
):
    monkeypatch.setattr(diag, "connect", MagicMock())
    monkeypatch.setattr(diag, "select_run", MagicMock(return_value=[{}]))
    monkeypatch.setattr(diag, "Gcp", MagicMock())
    monkeypatch.setattr(diag, "render", MagicMock(return_value="report"))
    report = MagicMock()
    monkeypatch.setattr(diag, "build_report", report)
    args = ["--stage", "prod20240903", "--db-url", "postgresql://example/db"]
    if override:
        args.extend(["--queue", override])
    result = CliRunner().invoke(diag.cli, args)
    assert result.exit_code == 0, result.output
    assert report.call_args.args[4]["queue"] == (
        override or "orchestrator-backlog-prod20240903"
    )


def scene(attempts=1, succeeded=False, failed=False, in_progress=False):
    return {
        "attempts": attempts,
        "succeeded": succeeded,
        "failed": failed,
        "in_progress": in_progress,
    }


@pytest.mark.parametrize(
    "flags, outcome",
    [
        ({"succeeded": True, "failed": True, "in_progress": True}, "succeeded"),
        ({"succeeded": False, "failed": True, "in_progress": True}, "in_progress"),
        ({"succeeded": False, "failed": True, "in_progress": False}, "failed"),
        ({"succeeded": False, "failed": False, "in_progress": False}, "crashed"),
    ],
)
def test_scene_outcome_prefers_success_then_live_attempts(flags, outcome):
    assert diag.scene_outcome(**flags) == outcome


def test_summarize_scenes_counts_scenes_not_retry_rows():
    rows = [
        scene(attempts=3, succeeded=True, failed=True),  # retried until it passed
        scene(succeeded=True),
        scene(attempts=3, failed=True),
        scene(),  # died without recording a result
        scene(in_progress=True),
    ]
    summary = diag.summarize_scenes(rows, queued=10, queue_depth=2)
    assert summary["succeeded"] == 2
    assert summary["failed"] == 1
    assert summary["crashed"] == 1
    assert summary["in_progress"] == 1
    assert summary["attempts"] == 9
    assert summary["retried"] == 2
    assert summary["unaccounted"] == 3  # 10 queued - 5 seen - 2 still queued
    assert summary["completed_pct"] == 20.0
    assert summary["failed_pct"] == 20.0


def test_summarize_scenes_without_a_queued_total():
    summary = diag.summarize_scenes(
        [scene(failed=True), scene(succeeded=True)], None, None
    )
    assert summary["unaccounted"] is None
    assert summary["completed_pct"] is None
    assert summary["failed_pct"] == 50.0  # falls back to scenes seen


@pytest.mark.parametrize(
    "depth, in_progress, verdict",
    [(5, 0, "RUNNING"), (0, 2, "RUNNING"), (0, 0, "COMPLETE"), (None, 0, "IDLE")],
)
def test_orchestrator_verdict(depth, in_progress, verdict):
    assert diag.orchestrator_verdict(depth, in_progress) == verdict


@pytest.mark.parametrize(
    "orchestrator, tasks, pending, minutes_ago, verdict",
    [
        ("RUNNING", 0, 0, None, "RUNNING"),
        ("COMPLETE", 4, 0, None, "RUNNING"),
        ("COMPLETE", 0, 0, 500, "COMPLETE"),
        ("IDLE", None, 0, 500, "IDLE"),
        ("COMPLETE", 0, 7, 5, "RUNNING"),  # ASA wrote a source moments ago
        ("COMPLETE", 0, 7, 500, "STALLED"),
        ("COMPLETE", 0, 7, None, "STALLED"),
    ],
)
def test_asa_verdict(orchestrator, tasks, pending, minutes_ago, verdict):
    last = NOW - timedelta(minutes=minutes_ago) if minutes_ago else None
    assert diag.asa_verdict(orchestrator, tasks, pending, last, NOW) == verdict


def test_series_peak_reports_time_and_minutes_near_limit():
    points = [("t1", 10.0), ("t2", 45.0), ("t3", 43.0), ("t4", 20.0)]
    peak = diag.series_peak(points, limit=45)
    assert peak["max"] == 45.0
    assert peak["at"] == "t2"
    assert peak["pct_of_limit"] == 100.0
    assert peak["minutes_near_limit"] == 2
    assert diag.peak_level(peak) == diag.FLAG
    assert diag.peak_level(diag.series_peak(points, limit=55)) == diag.WARN
    assert diag.peak_level(diag.series_peak(points, limit=2000)) == diag.OK
    assert diag.peak_level(diag.series_peak(points)) == diag.OK  # no limit known
    assert diag.series_peak([]) is None
    assert diag.series_peak(None) is None


def stats(**overrides):
    base = {
        "scenes": 100,
        "slicks": 50,
        "oil_slicks": 40,
        "scenes_with_slicks": 20,
        "first_slick": NOW - timedelta(hours=5),
        "last_slick": NOW - timedelta(hours=1),
        "asa_eligible": 50,
        "asa_reached": 40,
        "vessel_dark_skipped": 5,
        "slicks_with_source": 30,
        "sources": 80,
        "first_association": NOW - timedelta(hours=4),
        "last_association": NOW - timedelta(hours=1),
        "by_type": {"VESSEL": {"matches": 60, "slicks": 25}},
    }
    return {**base, **overrides}


def test_yield_metrics_use_reached_slicks_for_source_rates():
    metrics = diag.yield_metrics(stats(), scenes=200)
    assert metrics["slicks_per_scene"] == 0.25  # the caller's scene count wins
    assert metrics["pct_scenes_with_slick"] == 10.0
    assert metrics["sources_per_slick"] == 2.0  # 80 sources / 40 reached slicks
    assert metrics["pct_slicks_with_source"] == 75.0
    assert metrics["vessel_per_slick"] == 1.5
    assert metrics["dark_per_slick"] == 0.0
    empty = diag.yield_metrics(stats(scenes=0, asa_reached=0))
    assert empty["slicks_per_scene"] is None
    assert empty["sources_per_slick"] is None


def test_compare_to_baseline_flags_only_large_deviations():
    rows = diag.compare_to_baseline(
        {"a": 1.2, "b": 0.4, "c": 3.0, "d": None},
        {"a": 1.0, "b": 1.0, "c": 0, "d": 1.0},
        tolerance_pct=50,
    )
    levels = {row["metric"]: row["level"] for row in rows}
    assert levels == {"a": diag.OK, "b": diag.FLAG, "c": diag.OK, "d": diag.OK}
    assert rows[0]["pct_of_expected"] == pytest.approx(120.0)
    assert rows[2]["pct_of_expected"] is None  # no baseline to compare against


def test_scene_date_reads_the_acquisition_date_from_an_asa_task_name():
    name = "asa-S1A_IW_GRDH_1SDV_20221215T054112_20221215T054137_046345_058D0A_1B2C-ab12-20261102"
    assert diag.scene_date(name) == date(2022, 12, 15)
    assert diag.scene_date("unrelated-task") is None


def test_as_utc_treats_naive_database_times_as_utc():
    assert diag.as_utc(datetime(2026, 1, 1, 3)) == datetime(
        2026, 1, 1, 3, tzinfo=timezone.utc
    )
    assert diag.as_utc(None) is None


class FakeGcp:
    """Stands in for Cloud Tasks, Cloud Run, and Cloud Monitoring."""

    def __init__(self, backlog_depth=0, asa_depth=0, fail_monitoring=False):
        self.depths = {"orchestrator-backlog": backlog_depth, "asa-queue": asa_depth}
        self.fail_monitoring = fail_monitoring

    def find_queue(self, prefix):
        assert prefix == "cerulean-cloud-test-queue-cloud-tasks-asa-analysis"
        return "asa-queue"

    def queue(self, queue_id):
        return {
            "queue": queue_id,
            "depth": self.depths[queue_id],
            "oldest_task": None,
            "dispatching_now": 0,
            "max_concurrent_dispatches": 30,
        }

    def task_names(self, queue_id):
        return [
            "asa-S1A_IW_GRDH_1SDV_20221215T054112_20221215T054137_046345_058D0A_1B2C-x",
            "asa-S1A_IW_GRDH_1SDV_20261101T054112_20261101T054137_046345_058D0A_1B2C-x",
        ]

    def max_instances(self, service):
        return {"cerulean-cloud-test-cr-orchestrator": 45}.get(service)

    def instance_series(self, service, start, end):
        if self.fail_monitoring:
            raise RuntimeError("permission denied")
        return [("2026-11-02T01:00:00Z", 45.0), ("2026-11-02T01:01:00Z", 12.0)]

    def db_connection_series(self, stage, start, end):
        return [("2026-11-02T01:00:00Z", 60.0)]

    def queue_depth_series(self, queue_id, start, end):
        return [("2026-11-02T00:30:00Z", 900.0)]


@pytest.fixture
def database(monkeypatch):
    """Replace the SQL layer; the baseline finds far more slicks per scene than the run."""
    rows = [scene(succeeded=True) for _ in range(95)] + [
        scene(attempts=3, failed=True)
    ] * 5

    def fake_query(conn, sql, params=None):
        if "max_connections" in sql:
            return [{"max_connections": "200"}]
        return [{"now": NOW}]

    def fake_yield_stats(conn, scope, params):
        if "SNS_TOPIC" in scope:
            return stats(
                scenes=1000,
                slicks=3000,
                oil_slicks=800,
                scenes_with_slicks=400,
                asa_reached=1000,
                slicks_with_source=750,
                sources=2000,
                by_type={"VESSEL": {"matches": 1500, "slicks": 600}},
            )
        return stats()

    monkeypatch.setattr(diag, "query", fake_query)
    monkeypatch.setattr(diag, "scene_rows", lambda conn, triggers, now: rows)
    monkeypatch.setattr(
        diag,
        "orchestrator_times",
        lambda conn, triggers: {
            "first_start": NOW - timedelta(hours=12),
            "last_start": NOW - timedelta(hours=2),
            "last_end": NOW - timedelta(hours=1),
        },
    )
    monkeypatch.setattr(diag, "not_oil_ids", lambda conn: [1, 2])
    monkeypatch.setattr(diag, "yield_stats", fake_yield_stats)


RUNS = [
    {
        "historical_run": 7,
        "trigger": 42,
        "date_start": date(2022, 12, 1),
        "date_end": date(2023, 1, 1),
        "dry_run": False,
        "eodag_count": 150,
        "git_hash": "abc123",
        "queued": 110,
        "submission_start": NOW - timedelta(hours=13),
        "submission_end": NOW - timedelta(hours=12),
        "submission_success": True,
        "submission_error": None,
    }
]
OPTIONS = {
    "queue": "orchestrator-backlog",
    "asa_queue": None,
    "baseline_start": None,
    "baseline_end": None,
    "tolerance_pct": 50.0,
    "failed_warn_pct": 1.0,
    "failed_flag_pct": 5.0,
}


def messages(report, level):
    return [c["message"] for c in report["checks"] if c["level"] == level]


def test_report_for_a_finished_run_with_gcp(database):
    report = diag.build_report(None, FakeGcp(asa_depth=2), "test", RUNS, OPTIONS)

    assert report["run"]["eodag_count"] == 150
    assert report["run"]["git_hashes"] == ["abc123"]
    text = diag.render(report)
    assert "scenes found by EODAG" in text and "abc123" in text

    orchestrator = report["orchestrator"]
    assert orchestrator["verdict"] == "COMPLETE"
    assert orchestrator["succeeded"] == 95
    assert orchestrator["unaccounted"] == 10
    assert report["asa"]["queued_tasks_for_run"] == 1  # only the Dec 2022 task
    assert report["asa"]["verdict"] == "RUNNING"
    assert report["asa"]["pending_slicks"] == 10
    assert report["asa"]["reached_without_source"] == 10
    assert report["runtime"]["final"] is False
    assert report["runtime"]["elapsed"] == "12:00:00"

    peaks = report["bottlenecks"]
    assert peaks["orchestrator_instances"]["pct_of_limit"] == 100.0
    assert peaks["inference_instances"]["limit"] is None
    assert peaks["db_connections"]["pct_of_limit"] == 30.0
    assert peaks["orchestrator_queue_depth"]["max"] == 900.0

    flags = messages(report, diag.FLAG)
    assert any("orchestrator_instances peaked at 45 of 45" in m for m in flags)
    assert any("slicks_per_scene is 18% of the baseline" in m for m in flags)
    assert not any("db_connections" in m for m in flags)
    warnings = messages(report, diag.WARN)
    assert any("4.5% of scenes failed" in m for m in warnings)
    assert any("10 queued scenes never reached the database" in m for m in warnings)

    text = diag.render(report)
    assert "2. Orchestrator: COMPLETE" in text
    assert "(provisional: still running)" in text
    json.dumps(report, default=str)  # the --json output must serialize


def test_report_without_gcp_marks_queues_unknown_and_stalled_asa(database):
    report = diag.build_report(None, None, "test", RUNS, OPTIONS)
    assert report["orchestrator"]["verdict"] == "IDLE"
    assert report["asa"]["verdict"] == "STALLED"
    assert report["bottlenecks"] == {}
    assert any("never reached" in m for m in messages(report, diag.FLAG))
    assert "not checked" in diag.render(report)


def test_report_survives_a_failing_gcp_lookup(database):
    report = diag.build_report(
        None, FakeGcp(fail_monitoring=True), "test", RUNS, OPTIONS
    )
    assert report["bottlenecks"]["orchestrator_instances"] is None
    assert any("permission denied" in m for m in messages(report, diag.WARN))
    assert report["bottlenecks"]["db_connections"]["max"] == 60.0
    diag.render(report)


def test_small_baseline_is_reported_but_not_graded(database, monkeypatch):
    monkeypatch.setattr(
        diag, "yield_stats", lambda conn, scope, params: stats(scenes=5)
    )
    report = diag.build_report(None, None, "test", RUNS, OPTIONS)
    assert report["benchmarks"]["usable"] is False
    assert any("only 5 live scenes" in m for m in messages(report, diag.WARN))
    assert not any("baseline" in m for m in messages(report, diag.FLAG))


def test_db_url_resolution_order(monkeypatch):
    monkeypatch.setenv(
        "STACK_SECRETS", json.dumps({"TEST": {"DB_URL": "from-secrets"}})
    )
    monkeypatch.setenv("DB_URL", "from-env")
    assert diag.resolve_db_url("test", "explicit") == "explicit"
    assert diag.resolve_db_url("test", None) == "from-secrets"
    with pytest.raises(diag.click.ClickException, match="STAGING"):
        diag.resolve_db_url("staging", None)
    monkeypatch.delenv("STACK_SECRETS")
    assert diag.resolve_db_url("test", None) == "from-env"


POLYGON = "'POLYGON((0 0,1 0,1 1,0 1,0 0))'::geography"
MULTIPOLYGON = "'MULTIPOLYGON(((0 0,1 0,1 1,0 1,0 0)))'::geography"
SEED = f"""
-- Production writes naive UTC timestamps; make this session do the same.
SET TIME ZONE 'UTC';
INSERT INTO cls (id, short_name, supercls) VALUES
    (1, 'NOT_OIL', NULL), (2, 'LAND', 1), (3, 'OIL', NULL);
INSERT INTO source_type (id, short_name) VALUES (1, 'VESSEL'), (2, 'INFRA');
INSERT INTO trigger (id, trigger_logs, trigger_type, filtered_scene_count) VALUES
    (1, '', 'HISTORICAL_RUN', 6), (2, '', 'SNS_TOPIC', 1), (3, '', 'HISTORICAL_RUN', 9);
INSERT INTO historical_run
    (id, trigger, cli_command, date_start, date_end, dry_run, queued_task_count,
     submission_start_time, eodag_count, git_hash)
VALUES
    (1, 1, 'run', '2022-12-01', '2023-01-01', false, 5, now() - interval '1 day',
     8, 'abc123'),
    (2, 3, 'preview', '2022-12-01', '2023-01-01', true, 9, now(), NULL, NULL);
INSERT INTO model
    (id, type, file_path, layers, cls_map, tile_width_m, tile_width_px, thresholds)
VALUES (1, 'm', 'path', '{{}}', '{{}}', 1, 1, '{{}}');
INSERT INTO sentinel1_grd (id, scene_id, start_time, end_time, url, geometry)
SELECT n, 'scene' || n, '2022-12-05', '2022-12-05', '', {POLYGON}
FROM generate_series(1, 5) AS n;
INSERT INTO orchestrator_run
    (id, sentinel1_grd, trigger, success, inference_start_time, inference_end_time,
     inference_run_logs, geometry, model)
SELECT id, scene, trigger, success, now() - started, now() - ended, '', {POLYGON}, 1
FROM (VALUES
    -- scene 1 failed once, then passed on retry
    (1, 1, 1, false, interval '10 hours', interval '9 hours'),
    (2, 1, 1, true, interval '8 hours', interval '7 hours'),
    -- scene 2 died long ago without a result; scene 3 is still in progress
    (3, 2, 1, NULL, interval '5 hours', interval '5 hours'),
    (4, 3, 1, NULL, interval '5 minutes', interval '5 minutes'),
    (5, 4, 1, false, interval '4 hours', interval '3 hours'),
    -- a live scene for the baseline
    (6, 5, 2, true, interval '2 days', interval '2 days')
) AS run(id, scene, trigger, success, started, ended);
INSERT INTO slick
    (id, orchestrator_run, active, cls, machine_confidence,
     geometric_slick_potential, aspect_ratio_factor, slick_timestamp, geometry,
     inference_idx)
SELECT id, run, active, cls, 0.9, gsp, arf, '2022-12-05', {MULTIPOLYGON}, 1
FROM (VALUES
    (1, 1, false, 3, NULL, 0.9),  -- deactivated by the retry
    (2, 2, true, 3, 0.5, 0.9),    -- oil, two sources
    (3, 2, true, 2, 0.1, 0.9),    -- land, reached, no source
    (4, 2, true, 3, NULL, 0.2),   -- not yet reached by ASA
    (5, 6, true, 3, NULL, 0.9)    -- baseline slick, reached per its source
) AS slick(id, run, active, cls, gsp, arf);
INSERT INTO source (id, type, ext_id) VALUES (1, 1, 'mmsi'), (2, 2, 'rig');
INSERT INTO slick_to_source (slick, source, active, geojson_fc, geometry)
SELECT slick, source, active, '{{}}', {POLYGON}
FROM (VALUES (2, 1, true), (2, 2, true), (1, 1, true), (5, 1, true), (4, 1, false))
    AS link(slick, source, active);
"""


@pytest.fixture
def seeded(postgresql):
    """A real schema with one historical run, one dry run, and one live scene."""
    import sqlalchemy as sa

    from cerulean_cloud import database_schema

    info = postgresql.info
    url = f"postgresql://{info.user}:@{info.host}:{info.port}/{info.dbname}"
    engine = sa.create_engine(url)
    with engine.begin() as setup:
        setup.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS postgis")
        database_schema.Base.metadata.create_all(setup)
        setup.exec_driver_sql(SEED)
    engine.dispose()
    conn = diag.connect(url.replace("postgresql://", "postgresql+asyncpg://"))
    yield conn
    conn.close()


def test_queries_against_the_real_schema(seeded):
    conn = seeded
    now = diag.as_utc(diag.query(conn, "SELECT now() AS now")[0]["now"])

    runs = diag.select_run(conn, (), ())
    assert [run["historical_run"] for run in runs] == [1]  # skips the dry run
    assert runs[0]["queued"] == 5  # confirmed tasks win over the trigger count
    assert (runs[0]["eodag_count"], runs[0]["git_hash"]) == (8, "abc123")
    assert [r["trigger"] for r in diag.select_run(conn, (1, 2), ())] == [3, 1]
    by_trigger = diag.select_run(conn, (), (1,))
    assert by_trigger[0]["queued"] == 6 and by_trigger[0]["historical_run"] is None

    summary = diag.summarize_scenes(diag.scene_rows(conn, [1], now), 5, 0)
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["crashed"] == 1
    assert summary["in_progress"] == 1
    assert summary["attempts"] == 5 and summary["retried"] == 1
    assert summary["unaccounted"] == 1

    times = diag.orchestrator_times(conn, [1])
    assert times["last_end"] - times["first_start"] == timedelta(hours=7)
    assert times["last_end"] < times["last_start"]  # unfinished rows are ignored

    not_oil = diag.not_oil_ids(conn)
    assert sorted(not_oil) == [1, 2]
    params = {"not_oil": not_oil, "arf": diag.ASA_ASPECT_RATIO_CUTOFF, "triggers": [1]}
    run = diag.yield_stats(conn, "o.trigger = ANY(%(triggers)s)", params)
    assert run["slicks"] == 3  # the deactivated slick is excluded
    assert run["oil_slicks"] == 2
    assert run["scenes_with_slicks"] == 1
    assert run["asa_eligible"] == 3
    assert run["asa_reached"] == 2
    assert run["vessel_dark_skipped"] == 1
    assert run["slicks_with_source"] == 1
    assert run["sources"] == 2  # inactive links and inactive slicks are excluded
    assert run["by_type"] == {
        "VESSEL": {"matches": 1, "slicks": 1},
        "INFRA": {"matches": 1, "slicks": 1},
    }
    assert run["last_association"].tzinfo is timezone.utc

    baseline = diag.yield_stats(
        conn,
        "o.success IS TRUE AND o.inference_start_time >= %(start)s"
        " AND o.inference_start_time < %(end)s AND o.trigger IN"
        " (SELECT id FROM trigger WHERE trigger_type = 'SNS_TOPIC')",
        {**params, "start": now.date() - timedelta(days=365), "end": now.date()},
    )
    assert baseline["scenes"] == 1
    assert baseline["asa_reached"] == 1  # has a source although GSP is unset


def test_full_report_and_cli_against_the_real_schema(seeded, monkeypatch):
    report = diag.build_report(
        seeded, None, "test", diag.select_run(seeded, (), ()), OPTIONS
    )
    assert report["orchestrator"]["verdict"] == "RUNNING"
    assert report["asa"]["pending_slicks"] == 1
    assert report["benchmarks"]["baseline_scenes"] == 1
    assert "3. ASA: RUNNING" in diag.render(report)
    json.dumps(report, default=str)


def test_connection_rejects_writes(seeded):
    import psycopg2

    with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
        diag.query(
            seeded, "INSERT INTO cls (id, short_name) VALUES (99, 'X') RETURNING id"
        )
