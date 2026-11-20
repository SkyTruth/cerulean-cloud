-- Read-only companion to scripts/backfill_asa_completion_times_pre2023.sql. Run it first to
-- confirm where this week's historical backlog starts (its backlog_since value).

-- 1. Historical submissions since November, with the runs each one produced.
SELECT h.id AS historical_run, h.trigger, h.dry_run,
       h.submission_start_time::timestamp(0) AS submitted,
       h.date_start, h.date_end, h.queued_task_count,
       count(o.id) AS runs,
       min(o.inference_end_time)::timestamp(0) AS first_run_end,
       max(o.inference_end_time)::timestamp(0) AS last_run_end
FROM historical_run h
LEFT JOIN orchestrator_run o ON o.trigger = h.trigger
WHERE h.submission_start_time >= timestamp '2026-11-01'
GROUP BY h.id, h.trigger, h.dry_run, h.submission_start_time, h.date_start, h.date_end, h.queued_task_count
ORDER BY h.submission_start_time;

-- 2. Runs on pre-2023 scenes by week they finished, last 8 weeks.
SELECT date_trunc('week', o.inference_end_time)::date AS week_of,
       count(*) AS runs_on_pre2023_scenes,
       count(*) FILTER (WHERE o.asa_vessel_time IS NULL AND o.asa_infra_time IS NULL
                          AND o.asa_dark_time IS NULL) AS all_null
FROM orchestrator_run o
JOIN sentinel1_grd g ON g.id = o.sentinel1_grd
WHERE g.start_time < timestamp '2023-01-01'
  AND o.inference_end_time >= timezone('UTC', now()) - interval '8 weeks'
GROUP BY 1
ORDER BY 1;
