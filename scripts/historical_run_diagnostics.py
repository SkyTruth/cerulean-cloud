"""Report progress, yields, and bottlenecks for a Cerulean historical (archive) run.

Read-only. Database numbers come from the run's `trigger` rows; queue depths,
instance counts, and DB connections come from Cloud Tasks and Cloud Monitoring
using Google application default credentials.

    python scripts/historical_run_diagnostics.py --stage test --env-file notebooks/.env
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone

import click
import psycopg2
import psycopg2.extras

STACK_SECRET_KEYS = {"test": "TEST", "staging": "STAGING", "prod20240903": "PRODUCTION"}
BACKLOG_QUEUE = "orchestrator-backlog"
SOURCE_TYPES = ("VESSEL", "INFRA", "DARK", "NATURAL")

# An attempt with no result after the orchestrator's 59-minute timeout has died.
STALE_ATTEMPT = timedelta(minutes=60)
# ASA is treated as still working if it associated a source this recently.
ASA_RECENT = timedelta(minutes=30)
# The ASA handler skips VESSEL and DARK for slicks at or below this aspect ratio.
ASA_ASPECT_RATIO_CUTOFF = 0.35
# Listing queued ASA tasks is skipped above this depth.
MAX_TASKS_LISTED = 5000
MIN_BASELINE_SCENES = 100
MONITORING_PAD = timedelta(minutes=5)

OK, WARN, FLAG = "OK", "WARN", "FLAG"


# ---------------------------------------------------------------- pure logic


def as_utc(value):
    """The database stores naive UTC; make every timestamp comparable."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def pct(part, whole):
    """Percentage, or None when the denominator is missing or zero."""
    if part is None or not whole:
        return None
    return 100.0 * part / whole


def ratio(part, whole):
    """Plain ratio, or None when the denominator is missing or zero."""
    if part is None or not whole:
        return None
    return part / whole


def scene_outcome(succeeded, in_progress, failed):
    """One outcome per scene; retries add attempts, so rows are not scenes."""
    if succeeded:
        return "succeeded"
    if in_progress:
        return "in_progress"
    if failed:
        return "failed"
    return "crashed"  # every attempt ended without recording a result


def summarize_scenes(rows, queued, queue_depth):
    """Count scene outcomes and what the queued total leaves unexplained."""
    counts = {"succeeded": 0, "in_progress": 0, "failed": 0, "crashed": 0}
    attempts = retried = 0
    for row in rows:
        counts[scene_outcome(row["succeeded"], row["in_progress"], row["failed"])] += 1
        attempts += row["attempts"]
        retried += row["attempts"] > 1
    seen = len(rows)
    unaccounted = None
    if queued is not None:
        # Scenes exit before any insert when they are dry runs, touch the
        # antimeridian, or have no ocean tiles; those never reach the database.
        unaccounted = max(queued - seen - (queue_depth or 0), 0)
    return {
        **counts,
        "queued": queued,
        "seen": seen,
        "attempts": attempts,
        "retried": retried,
        "unaccounted": unaccounted,
        "completed_pct": pct(counts["succeeded"], queued),
        "failed_pct": pct(counts["failed"] + counts["crashed"], queued or seen),
    }


def orchestrator_verdict(queue_depth, in_progress):
    """RUNNING while work is queued or in progress; IDLE when the queue is unknown."""
    if in_progress or queue_depth:
        return "RUNNING"
    return "IDLE" if queue_depth is None else "COMPLETE"


def asa_verdict(orchestrator, queued_tasks, pending_slicks, last_association, now):
    """ASA has no run table, so combine its queue with what it wrote to slicks."""
    if orchestrator == "RUNNING" or queued_tasks:
        return "RUNNING"
    if not pending_slicks:
        return "IDLE" if queued_tasks is None else "COMPLETE"
    if last_association and now - last_association < ASA_RECENT:
        return "RUNNING"
    return "STALLED"  # nothing queued, yet slicks were never reached


def series_peak(points, limit=None, near=0.95):
    """Peak of (time, value) points and the time spent near the limit."""
    if not points:
        return None
    when, peak = max(points, key=lambda point: point[1])
    result = {
        "max": peak,
        "at": when,
        "limit": limit,
        "pct_of_limit": pct(peak, limit),
    }
    if limit:
        result["minutes_near_limit"] = sum(value >= near * limit for _, value in points)
    return result


def peak_level(peak, warn=80, flag=95):
    """Grade a peak against its limit."""
    used = (peak or {}).get("pct_of_limit")
    if used is None:
        return OK
    return FLAG if used >= flag else WARN if used >= warn else OK


def yield_metrics(stats, scenes=None):
    """Per-scene and per-slick rates that can be compared between runs."""
    scenes = stats["scenes"] if scenes is None else scenes
    reached = stats["asa_reached"]
    metrics = {
        "slicks_per_scene": ratio(stats["slicks"], scenes),
        "oil_slicks_per_scene": ratio(stats["oil_slicks"], scenes),
        "pct_scenes_with_slick": pct(stats["scenes_with_slicks"], scenes),
        "sources_per_slick": ratio(stats["sources"], reached),
        "pct_slicks_with_source": pct(stats["slicks_with_source"], reached),
    }
    for name in SOURCE_TYPES:
        metrics[f"{name.lower()}_per_slick"] = ratio(
            stats["by_type"].get(name, {}).get("matches", 0), reached
        )
    return metrics


def compare_to_baseline(run, baseline, tolerance_pct):
    """Express each run metric as a percentage of its baseline value."""
    rows = []
    for name, value in run.items():
        expected = baseline.get(name)
        of_expected = pct(value, expected)
        level = OK
        if of_expected is not None and abs(of_expected - 100) > tolerance_pct:
            level = FLAG
        rows.append(
            {
                "metric": name,
                "run": value,
                "expected": expected,
                "pct_of_expected": of_expected,
                "level": level,
            }
        )
    return rows


def scene_date(task_name):
    """Acquisition date embedded in a Sentinel-1 scene ID inside a task name."""
    match = re.search(r"_(\d{8})T\d{6}_", task_name)
    return datetime.strptime(match.group(1), "%Y%m%d").date() if match else None


# ------------------------------------------------------------------ database


def connect(db_url):
    """Open a session that the database itself keeps read-only."""
    conn = psycopg2.connect(
        db_url.replace("postgresql+asyncpg://", "postgresql://", 1),
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    conn.set_session(readonly=True, autocommit=True)
    return conn


def query(conn, sql, params=None):
    """Run one statement and return its rows."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def select_run(conn, historical_run_ids, trigger_ids):
    """Resolve the run to its trigger rows, preferring the historical_run table."""
    has_history = query(conn, "SELECT to_regclass('public.historical_run') AS t")[0][
        "t"
    ]
    if has_history and not trigger_ids:
        where = "hr.id = ANY(%(ids)s)" if historical_run_ids else "NOT hr.dry_run"
        rows = query(
            conn,
            f"""
            SELECT hr.id AS historical_run, hr.trigger, hr.date_start, hr.date_end,
                   hr.dry_run, hr.eodag_count, hr.git_hash,
                   coalesce(hr.queued_task_count, t.filtered_scene_count) AS queued,
                   hr.submission_start_time AS submission_start,
                   hr.submission_end_time AS submission_end,
                   hr.submission_success, hr.submission_error
            FROM historical_run hr JOIN trigger t ON t.id = hr.trigger
            WHERE {where}
            ORDER BY hr.submission_start_time DESC
            """,
            {"ids": list(historical_run_ids)},
        )
        return rows if historical_run_ids else rows[:1]
    if historical_run_ids:
        raise click.ClickException("This database has no historical_run table")
    where = (
        "t.id = ANY(%(ids)s)" if trigger_ids else "t.trigger_type = 'HISTORICAL_RUN'"
    )
    rows = query(
        conn,
        f"""
        SELECT NULL::bigint AS historical_run, t.id AS trigger,
               NULL::date AS date_start, NULL::date AS date_end,
               NULL::boolean AS dry_run, t.scene_count AS eodag_count,
               NULL::text AS git_hash, t.filtered_scene_count AS queued, t.trigger_time AS submission_start,
               NULL::timestamp AS submission_end, NULL::boolean AS submission_success,
               NULL::text AS submission_error
        FROM trigger t WHERE {where} ORDER BY t.trigger_time DESC
        """,
        {"ids": list(trigger_ids)},
    )
    return rows if trigger_ids else rows[:1]


def scene_rows(conn, triggers, now):
    """One row per scene with what its attempts recorded."""
    return query(
        conn,
        """
        SELECT o.sentinel1_grd,
               count(*) AS attempts,
               bool_or(o.success IS TRUE) AS succeeded,
               bool_or(o.success IS FALSE) AS failed,
               bool_or(o.success IS NULL AND o.inference_start_time >= %(cutoff)s)
                   AS in_progress
        FROM orchestrator_run o
        WHERE o.trigger = ANY(%(triggers)s)
        GROUP BY o.sentinel1_grd
        """,
        {"triggers": triggers, "cutoff": (now - STALE_ATTEMPT).replace(tzinfo=None)},
    )


def orchestrator_times(conn, triggers):
    """First start and last recorded finish; unfinished rows carry no end time."""
    row = query(
        conn,
        """
        SELECT min(inference_start_time) AS first_start,
               max(inference_start_time) AS last_start,
               max(inference_end_time) FILTER (WHERE success IS NOT NULL) AS last_end
        FROM orchestrator_run WHERE trigger = ANY(%(triggers)s)
        """,
        {"triggers": triggers},
    )[0]
    return {key: as_utc(value) for key, value in row.items()}


def not_oil_ids(conn):
    """NOT_OIL and its subclasses (land, sea ice, artefact)."""
    row = query(
        conn,
        """
        WITH RECURSIVE tree AS (
            SELECT id FROM cls WHERE short_name = 'NOT_OIL'
            UNION ALL
            SELECT c.id FROM cls c JOIN tree ON c.supercls = tree.id
        )
        SELECT coalesce(array_agg(id), '{}') AS ids FROM tree
        """,
    )[0]
    return list(row["ids"])


def yield_stats(conn, scope, params):
    """Active slicks and source associations for the runs matched by `scope`.

    A slick counts as reached by ASA when its geometric slick potential is set
    (only ASA writes it) or it already has a source.
    """
    cte = f"""
        WITH runs AS (
            SELECT o.id, o.sentinel1_grd FROM orchestrator_run o WHERE {scope}
        ), sl AS (
            SELECT s.id, s.orchestrator_run, s.cls, s.create_time,
                   s.geometric_slick_potential AS gsp,
                   s.aspect_ratio_factor AS arf,
                   coalesce(s.machine_confidence, 0) > 0 AS eligible
            FROM slick s JOIN runs ON runs.id = s.orchestrator_run
            WHERE s.active
        ), src AS (
            SELECT sts.slick, source.type, sts.create_time
            FROM slick_to_source sts
            JOIN sl ON sl.id = sts.slick
            JOIN source ON source.id = sts.source
            WHERE sts.active
        ), matched AS (
            SELECT DISTINCT slick FROM src
        )
    """
    totals = query(
        conn,
        cte
        + """
        SELECT
            (SELECT count(DISTINCT sentinel1_grd) FROM runs) AS scenes,
            (SELECT count(*) FROM sl) AS slicks,
            (SELECT count(*) FROM sl WHERE cls <> ALL(%(not_oil)s)) AS oil_slicks,
            (SELECT count(DISTINCT orchestrator_run) FROM sl) AS scenes_with_slicks,
            (SELECT min(create_time) FROM sl) AS first_slick,
            (SELECT max(create_time) FROM sl) AS last_slick,
            (SELECT count(*) FROM sl WHERE eligible) AS asa_eligible,
            (SELECT count(*) FROM sl LEFT JOIN matched m ON m.slick = sl.id
              WHERE eligible AND (gsp IS NOT NULL OR m.slick IS NOT NULL))
                AS asa_reached,
            (SELECT count(*) FROM sl WHERE eligible AND arf <= %(arf)s)
                AS vessel_dark_skipped,
            (SELECT count(*) FROM matched) AS slicks_with_source,
            (SELECT count(*) FROM src) AS sources,
            (SELECT min(create_time) FROM src) AS first_association,
            (SELECT max(create_time) FROM src) AS last_association
        """,
        params,
    )[0]
    by_type = query(
        conn,
        cte
        + """
        SELECT st.short_name, count(*) AS matches, count(DISTINCT src.slick) AS slicks
        FROM src JOIN source_type st ON st.id = src.type
        GROUP BY st.short_name
        """,
        params,
    )
    stats = dict(totals)
    for key in ("first_slick", "last_slick", "first_association", "last_association"):
        stats[key] = as_utc(stats[key])
    stats["by_type"] = {
        row["short_name"]: {"matches": row["matches"], "slicks": row["slicks"]}
        for row in by_type
    }
    return stats


# ---------------------------------------------------------------------- GCP


class Gcp:
    """Thin REST access to Cloud Tasks, Cloud Run, and Cloud Monitoring."""

    def __init__(self, project, region):
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.session = AuthorizedSession(credentials)
        self.project = project
        self.location = f"projects/{project}/locations/{region}"

    def get(self, url, params=None):
        response = self.session.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()

    def pages(self, url, key, params=None, limit=None):
        """Follow nextPageToken, stopping early once `limit` items are read."""
        params, items = dict(params or {}), []
        while True:
            body = self.get(url, params)
            items += body.get(key, [])
            token = body.get("nextPageToken")
            if not token or (limit and len(items) >= limit):
                return items
            params["pageToken"] = token

    def find_queue(self, prefix):
        """Pulumi appends a random suffix, so match queues by name prefix."""
        queues = self.pages(
            f"https://cloudtasks.googleapis.com/v2/{self.location}/queues", "queues"
        )
        names = [queue["name"].rsplit("/", 1)[1] for queue in queues]
        matches = [name for name in names if name.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one queue starting with {prefix}: {matches}")
        return matches[0]

    def queue(self, queue_id):
        """Current depth, oldest task, and dispatch limit of one queue."""
        body = self.get(
            f"https://cloudtasks.googleapis.com/v2beta3/{self.location}/queues/{queue_id}",
            {"readMask": "name,state,rateLimits,stats"},
        )
        stats, limits = body.get("stats", {}), body.get("rateLimits", {})
        return {
            "queue": queue_id,
            "state": body.get("state"),
            "depth": int(stats.get("tasksCount", 0)),
            "oldest_task": stats.get("oldestEstimatedArrivalTime"),
            "executed_last_minute": int(stats.get("executedLastMinuteCount", 0)),
            "dispatching_now": int(stats.get("concurrentDispatchesCount", 0)),
            "max_concurrent_dispatches": limits.get("maxConcurrentDispatches"),
        }

    def task_names(self, queue_id):
        """Names of queued tasks, up to the listing cap."""
        tasks = self.pages(
            f"https://cloudtasks.googleapis.com/v2/{self.location}/queues/{queue_id}/tasks",
            "tasks",
            {"pageSize": 1000},
            limit=MAX_TASKS_LISTED,
        )
        return [task["name"].rsplit("/", 1)[1] for task in tasks]

    def max_instances(self, service):
        """Configured instance ceiling, or None when the platform default applies."""
        body = self.get(
            f"https://run.googleapis.com/v2/{self.location}/services/{service}"
        )
        return (
            body.get("template", {}).get("scaling", {}).get("maxInstanceCount") or None
        )

    def series(self, metric_filter, start, end, group_by):
        """Per-minute maxima, summed across the series sharing `group_by`."""
        series = self.pages(
            f"https://monitoring.googleapis.com/v3/projects/{self.project}/timeSeries",
            "timeSeries",
            {
                "filter": metric_filter,
                "interval.startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "interval.endTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "aggregation.alignmentPeriod": "60s",
                "aggregation.perSeriesAligner": "ALIGN_MAX",
                "aggregation.crossSeriesReducer": "REDUCE_SUM",
                "aggregation.groupByFields": group_by,
                "pageSize": 100000,
            },
        )
        merged = {}  # pages can split one series; keep the busiest per minute
        for item in series:
            for point in item.get("points", []):
                value = point["value"]
                number = float(value.get("int64Value", value.get("doubleValue", 0)))
                when = point["interval"]["endTime"]
                merged[when] = max(merged.get(when, 0), number)
        return sorted(merged.items())

    def instance_series(self, service, start, end):
        return self.series(
            'metric.type="run.googleapis.com/container/instance_count" '
            'AND resource.type="cloud_run_revision" '
            f'AND resource.labels.service_name="{service}"',
            start,
            end,
            "resource.label.service_name",
        )

    def db_connection_series(self, stage, start, end):
        instance = f"{self.project}:cerulean-cloud-{stage}-database-instance"
        return self.series(
            'metric.type="cloudsql.googleapis.com/database/postgresql/num_backends" '
            'AND resource.type="cloudsql_database" '
            f'AND resource.labels.database_id=starts_with("{instance}")',
            start,
            end,
            "resource.label.database_id",
        )

    def queue_depth_series(self, queue_id, start, end):
        return self.series(
            'metric.type="cloudtasks.googleapis.com/queue/depth" '
            'AND resource.type="cloud_tasks_queue" '
            f'AND resource.labels.queue_id="{queue_id}"',
            start,
            end,
            "resource.label.queue_id",
        )


def attempt(report, label, call):
    """Run one GCP lookup; a failure becomes a warning, not a crash."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - any API or auth error is reportable
        add_check(report, WARN, f"{label} unavailable: {exc}")
        return None


# ------------------------------------------------------------------- report


def add_check(report, level, message):
    report["checks"].append({"level": level, "message": message})


def build_report(conn, gcp, stage, runs, options, baseline_conn=None):
    """Collect every section into one JSON-serializable dictionary."""
    now = as_utc(query(conn, "SELECT now() AS now")[0]["now"])
    triggers = [run["trigger"] for run in runs]
    report = {"generated_at": now, "stage": stage, "checks": []}

    # 1. Run
    queued_counts = [run["queued"] for run in runs]
    queued = None if None in queued_counts else sum(queued_counts)
    # Only an EODAG search records a count; explicit scene lists leave it empty.
    eodag_counts = [run["eodag_count"] for run in runs]
    starts = [
        as_utc(run["submission_start"]) for run in runs if run["submission_start"]
    ]
    report["run"] = {
        "historical_runs": [run["historical_run"] for run in runs],
        "triggers": triggers,
        "date_start": min(
            (r["date_start"] for r in runs if r["date_start"]), default=None
        ),
        "date_end": max((r["date_end"] for r in runs if r["date_end"]), default=None),
        "eodag_count": None if None in eodag_counts else sum(eodag_counts),
        "queued": queued,
        "git_hashes": sorted({run["git_hash"] for run in runs if run["git_hash"]}),
        "submission_start": min(starts, default=None),
        "submissions": [dict(run) for run in runs],
    }
    if any(run["dry_run"] for run in runs):
        add_check(report, WARN, "Dry-run submission: the orchestrator writes nothing")
    if queued is None:
        add_check(
            report, WARN, "Queued task count unknown (submission failed or old schema)"
        )

    # 2-3. Orchestrator
    backlog = gcp and attempt(
        report, "Orchestrator queue", lambda: gcp.queue(options["queue"])
    )
    depth = backlog["depth"] if backlog else None
    scenes = summarize_scenes(scene_rows(conn, triggers, now), queued, depth)
    times = orchestrator_times(conn, triggers)
    verdict = orchestrator_verdict(depth, scenes["in_progress"])
    report["orchestrator"] = {**scenes, **times, "queue": backlog, "verdict": verdict}
    failed_pct = scenes["failed_pct"] or 0
    if failed_pct > options["failed_warn_pct"]:
        level = FLAG if failed_pct > options["failed_flag_pct"] else WARN
        add_check(report, level, f"{failed_pct:.1f}% of scenes failed or crashed")
    if scenes["unaccounted"] and verdict != "RUNNING":
        add_check(
            report,
            WARN,
            f"{scenes['unaccounted']} queued scenes never reached the database "
            "(antimeridian, no ocean tiles, dry run, or dropped tasks)",
        )

    # 4. ASA
    not_oil = not_oil_ids(conn)
    params = {"not_oil": not_oil, "arf": ASA_ASPECT_RATIO_CUTOFF, "triggers": triggers}
    stats = yield_stats(conn, "o.trigger = ANY(%(triggers)s)", params)
    asa_queue, run_tasks = None, None
    if gcp:
        asa_queue = attempt(
            report,
            "ASA queue",
            lambda: gcp.queue(
                options["asa_queue"]
                or gcp.find_queue(
                    f"cerulean-cloud-{stage}-queue-cloud-tasks-asa-analysis"
                )
            ),
        )
    if asa_queue:
        run_tasks = asa_queue["depth"]  # shared with live scenes unless narrowed below
        first, last = report["run"]["date_start"], report["run"]["date_end"]
        if first and last and 0 < asa_queue["depth"] <= MAX_TASKS_LISTED:
            names = attempt(
                report, "ASA task list", lambda: gcp.task_names(asa_queue["queue"])
            )
            if names is not None:
                dates = [scene_date(name) for name in names]
                run_tasks = sum(bool(day) and first <= day <= last for day in dates)
    pending = stats["asa_eligible"] - stats["asa_reached"]
    unmatched = stats["asa_reached"] - stats["slicks_with_source"]
    asa_state = asa_verdict(verdict, run_tasks, pending, stats["last_association"], now)
    report["asa"] = {
        **stats,
        "queue": asa_queue,
        "queued_tasks_for_run": run_tasks,
        "scenes_expecting_asa": stats["scenes_with_slicks"],
        "pending_slicks": pending,
        "reached_without_source": unmatched,
        "reached_without_source_pct": pct(unmatched, stats["asa_reached"]),
        "verdict": asa_state,
    }
    if asa_state == "STALLED":
        add_check(
            report, FLAG, f"ASA queue is empty but {pending} slicks were never reached"
        )

    # 5. Runtime
    ends = [times["last_end"], stats["last_slick"], stats["last_association"]]
    start = report["run"]["submission_start"] or times["first_start"]
    end = max((value for value in ends if value), default=None)
    report["runtime"] = {
        "start": start,
        "first_slick": stats["first_slick"],
        "last_slick": stats["last_slick"],
        "last_orchestrator_end": times["last_end"],
        "last_association": stats["last_association"],
        "end": end,
        "elapsed": str(timedelta(seconds=round((end - start).total_seconds())))
        if start and end
        else None,
        "final": verdict == "COMPLETE" and asa_state == "COMPLETE",
    }

    # 6. Benchmarks
    run_metrics = yield_metrics(stats, scenes["succeeded"])
    baseline_end = options["baseline_end"] or now.date()
    baseline_start = options["baseline_start"] or baseline_end - timedelta(days=365)
    baseline = yield_stats(
        baseline_conn or conn,
        "o.success IS TRUE AND o.inference_start_time >= %(start)s"
        " AND o.inference_start_time < %(end)s AND o.trigger IN"
        " (SELECT id FROM trigger WHERE trigger_type = 'SNS_TOPIC')",
        {**params, "start": baseline_start, "end": baseline_end},
    )
    usable = baseline["scenes"] >= MIN_BASELINE_SCENES
    rows = compare_to_baseline(
        run_metrics, yield_metrics(baseline), options["tolerance_pct"]
    )
    report["benchmarks"] = {
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "baseline_scenes": baseline["scenes"],
        "tolerance_pct": options["tolerance_pct"],
        "usable": usable,
        "metrics": rows,
    }
    if not usable:
        add_check(
            report,
            WARN,
            f"Baseline has only {baseline['scenes']} live scenes; benchmarks are "
            "not graded (use --baseline-db-url for a fuller database)",
        )
    for row in rows if usable else []:
        if row["level"] == FLAG:
            add_check(
                report,
                FLAG,
                f"{row['metric']} is {row['pct_of_expected']:.0f}% of the baseline",
            )

    # 7. Bottlenecks
    report["bottlenecks"] = {}
    # A dry run records no inference times, so fall back to its submission.
    window_start = times["first_start"] or start
    if gcp and window_start:
        # Monitoring samples land up to a minute late, so pad a finished run.
        finished = verdict == "COMPLETE" and end
        window_end = min(end + MONITORING_PAD, now) if finished else now
        window = (window_start - MONITORING_PAD, window_end)
        peaks = report["bottlenecks"]
        peaks["window"] = {"start": window[0], "end": window[1]}
        for label, suffix in (
            ("orchestrator", "cr-orchestrator"),
            ("inference", "cr-infer"),
            ("asa", "cf-asa"),
        ):
            service = f"cerulean-cloud-{stage}-{suffix}"
            limit = attempt(
                report, f"{label} instance limit", lambda: gcp.max_instances(service)
            )
            points = attempt(
                report,
                f"{label} instance count",
                lambda: gcp.instance_series(service, *window),
            )
            peaks[f"{label}_instances"] = series_peak(points, limit)
        max_connections = int(query(conn, "SHOW max_connections")[0]["max_connections"])
        points = attempt(
            report,
            "DB connections",
            lambda: gcp.db_connection_series(stage, *window),
        )
        peaks["db_connections"] = series_peak(points, max_connections)
        points = attempt(
            report,
            "Queue depth history",
            lambda: gcp.queue_depth_series(options["queue"], *window),
        )
        peaks["orchestrator_queue_depth"] = series_peak(points)
        for name, peak in peaks.items():
            level = peak_level(peak) if name != "window" else OK
            if level != OK:
                add_check(
                    report,
                    level,
                    f"{name} peaked at {peak['max']:.0f} of {peak['limit']} "
                    f"({peak['pct_of_limit']:.0f}%) at {peak['at']}",
                )
    return report


def show(value):
    """Compact text for one report value."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S UTC")
    return str(value)


def render(report):
    """Readable terminal report; the JSON output carries the same data."""
    lines = []

    def section(title):
        lines.extend(["", title, "-" * len(title)])

    def row(label, value, note=""):
        lines.append(f"  {label:<34}{show(value)}{'  ' + note if note else ''}")

    def percent(value):
        return "" if value is None else f"({value:.1f}%)"

    run, orch, asa = report["run"], report["orchestrator"], report["asa"]
    lines.append(
        f"Historical run diagnostics: {report['stage']} at {show(report['generated_at'])}"
    )

    section("1. Run")
    row("historical_run ids", run["historical_runs"])
    row("trigger ids", run["triggers"])
    row("scene dates", f"{show(run['date_start'])} to {show(run['date_end'])}")
    row("scenes found by EODAG", run["eodag_count"])
    row("tasks queued", run["queued"])
    row("git hash", ", ".join(run["git_hashes"]) or None)
    row("submission started", run["submission_start"])

    section(f"2. Orchestrator: {orch['verdict']}")
    row(
        "scenes succeeded",
        orch["succeeded"],
        percent(orch["completed_pct"]) + " of queued",
    )
    row("scenes failed", orch["failed"])
    row("scenes crashed (no result)", orch["crashed"])
    row("failed or crashed", percent(orch["failed_pct"]) or None)
    row("scenes in progress", orch["in_progress"])
    row("queued but never in database", orch["unaccounted"])
    row("attempts / scenes retried", f"{orch['attempts']} / {orch['retried']}")
    queue = orch["queue"] or {}
    row("queue depth now", queue.get("depth"), queue.get("queue", ""))
    row("oldest queued task", queue.get("oldest_task"))
    row(
        "dispatching now / limit",
        f"{show(queue.get('dispatching_now'))} / {show(queue.get('max_concurrent_dispatches'))}",
    )
    row("earliest start_time", orch["first_start"])
    row("latest end_time", orch["last_end"])

    section(f"3. ASA: {asa['verdict']}")
    queue = asa["queue"] or {}
    row("queue depth now", queue.get("depth"), queue.get("queue", ""))
    row("queued tasks for this run", asa["queued_tasks_for_run"])
    row("scenes expecting ASA", asa["scenes_expecting_asa"])
    row("active slicks", asa["slicks"], f"{asa['oil_slicks']} not NOT_OIL")
    row("slicks eligible for ASA", asa["asa_eligible"])
    row("slicks reached by ASA", asa["asa_reached"])
    row("slicks still pending", asa["pending_slicks"])
    row(
        "reached but no source",
        asa["reached_without_source"],
        percent(asa["reached_without_source_pct"]),
    )
    row("VESSEL/DARK skipped (narrow)", asa["vessel_dark_skipped"])
    row("source associations", asa["sources"])
    for name in SOURCE_TYPES:
        found = asa["by_type"].get(name, {"matches": 0, "slicks": 0})
        row(f"  {name}", found["matches"], f"on {found['slicks']} slicks")
    row("earliest association", asa["first_association"])
    row("latest association", asa["last_association"])

    runtime = report["runtime"]
    section(
        "4. Runtime" + ("" if runtime["final"] else " (provisional: still running)")
    )
    row("start", runtime["start"])
    row("first slick added", runtime["first_slick"])
    row("last slick added", runtime["last_slick"])
    row("last orchestrator end", runtime["last_orchestrator_end"])
    row("last source association", runtime["last_association"])
    row("elapsed", runtime["elapsed"])

    bench = report["benchmarks"]
    section("5. Benchmarks vs live scenes")
    row("baseline period", f"{bench['baseline_start']} to {bench['baseline_end']}")
    row("baseline scenes", bench["baseline_scenes"])
    lines.append(f"  {'metric':<28}{'run':>10}{'expected':>10}{'% of exp.':>11}")
    for item in bench["metrics"]:
        mark = "  FLAG" if item["level"] == FLAG and bench["usable"] else ""
        lines.append(
            f"  {item['metric']:<28}{show(item['run']):>10}{show(item['expected']):>10}"
            f"{show(item['pct_of_expected']):>11}{mark}"
        )

    section("6. Bottlenecks")
    peaks = report["bottlenecks"]
    if not peaks:
        lines.append("  not checked")
    for name, peak in peaks.items():
        if name == "window":
            row("window", f"{show(peak['start'])} to {show(peak['end'])}")
        elif peak is None:
            row(name, None)
        else:
            note = f"at {peak['at']}"
            if peak["limit"]:
                note = (
                    f"of {peak['limit']} ({peak['pct_of_limit']:.0f}%) {note}; "
                    f"{peak['minutes_near_limit']} min near limit"
                )
            row(f"max {name}", f"{peak['max']:.0f}", note)

    section("Flags")
    if not report["checks"]:
        lines.append("  none")
    for check in report["checks"]:
        lines.append(f"  [{check['level']}] {check['message']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------- CLI


def resolve_db_url(stage, db_url):
    """Explicit URL first, then the notebook's STACK_SECRETS, then DB_URL."""
    if db_url:
        return db_url
    if os.getenv("STACK_SECRETS"):
        key = STACK_SECRET_KEYS.get(stage, stage)
        try:
            return json.loads(os.environ["STACK_SECRETS"])[key]["DB_URL"]
        except (KeyError, TypeError, ValueError):
            raise click.ClickException(
                f"STACK_SECRETS must contain a DB_URL for {key}"
            ) from None
    for name in ("HISTORICAL_DB_URL", "DB_URL"):
        if os.getenv(name):
            return os.environ[name]
    raise click.ClickException("Provide --db-url, STACK_SECRETS, or DB_URL")


@click.command()
@click.option("--stage", default="test", show_default=True, help="Pulumi stack name.")
@click.option(
    "--env-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Load environment variables, including STACK_SECRETS, from this file.",
)
@click.option("--db-url", help="Database URL; overrides STACK_SECRETS and DB_URL.")
@click.option(
    "--historical-run-id",
    type=int,
    multiple=True,
    help="historical_run.id; repeatable.",
)
@click.option("--trigger-id", type=int, multiple=True, help="trigger.id; repeatable.")
@click.option(
    "--queue",
    default=BACKLOG_QUEUE,
    show_default=True,
    help="Orchestrator queue of the run.",
)
@click.option("--asa-queue", help="ASA queue ID; found by stack prefix when omitted.")
@click.option(
    "--baseline-start",
    type=click.DateTime(["%Y-%m-%d"]),
    help="Default: one year before the end.",
)
@click.option(
    "--baseline-end", type=click.DateTime(["%Y-%m-%d"]), help="Default: today."
)
@click.option("--baseline-db-url", help="Compute the baseline from another database.")
@click.option(
    "--tolerance-pct",
    default=50.0,
    show_default=True,
    help="Benchmark deviation that raises a flag.",
)
@click.option("--failed-warn-pct", default=1.0, show_default=True)
@click.option("--failed-flag-pct", default=5.0, show_default=True)
@click.option("--skip-gcp", is_flag=True, help="Database sections only.")
@click.option(
    "--json",
    "json_path",
    type=click.Path(dir_okay=False),
    help="Also write the report as JSON.",
)
def cli(
    stage,
    env_file,
    db_url,
    historical_run_id,
    trigger_id,
    skip_gcp,
    json_path,
    **options,
):
    """Diagnose the latest historical run, or the runs selected by ID."""
    if historical_run_id and trigger_id:
        raise click.UsageError("Use --historical-run-id or --trigger-id, not both")
    if env_file:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    for key in ("baseline_start", "baseline_end"):
        options[key] = options[key] and options[key].date()

    conn = connect(resolve_db_url(stage, db_url))
    baseline_conn = options["baseline_db_url"] and connect(options["baseline_db_url"])
    runs = select_run(conn, historical_run_id, trigger_id)
    if not runs:
        raise click.ClickException(
            "No matching historical run found. Dry runs are skipped by default; "
            "select one with --historical-run-id or --trigger-id"
        )

    gcp = None
    if not skip_gcp:
        try:
            gcp = Gcp(
                os.getenv("GCPPROJECT", "cerulean-338116"),
                os.getenv("GCPREGION", "europe-west1"),
            )
        except Exception as exc:  # noqa: BLE001 - report missing credentials plainly
            raise click.ClickException(
                f"Google credentials unavailable ({exc}); use --skip-gcp"
            ) from None

    report = build_report(conn, gcp, stage, runs, options, baseline_conn)
    click.echo(render(report))
    if json_path:
        with open(json_path, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        click.echo(f"\nJSON written to {json_path}")


if __name__ == "__main__":
    cli()
