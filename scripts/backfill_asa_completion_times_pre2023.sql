-- Supplement to scripts/backfill_asa_completion_times.sql: runs on scenes captured before 2023
-- had ASA unless they belong to this week's backlog. Fills only columns that are still NULL
-- with 1970-01-01 ("done, time unknown"); stamps from source rows or from #224 are kept.
--
-- Skipped, and left as they are:
--   * runs that ended on or after backlog_since, or whose trigger belongs to a non-dry-run
--     historical_run submitted on or after backlog_since (this week's backlog)
--   * runs that ended after GSP shipped (2026-03-17 19:49:08) where some active slick has
--     neither GSP nor a source (ASA never reached it)
--   * same-scene re-runs that finished earlier the same UTC day (their ASA task may have
--     been deduplicated)
--   * runs with no active slick with machine_confidence > 0 (#224 never selects them)
--
-- First run scripts/backfill_asa_completion_times_pre2023_precheck.sql to confirm backlog_since.
-- Plain SQL: runs in psql (\i) or DBeaver (Manual Commit, Execute SQL Script).
--   psql:    BEGIN;  \i scripts/backfill_asa_completion_times_pre2023.sql   then COMMIT; or ROLLBACK;
--   DBeaver: Manual Commit mode, Alt+X, then the Commit or Rollback button.
-- Every "MUST BE 0" row in the final result must show 0 before you commit.
--
-- Undo after COMMIT:
--   UPDATE orchestrator_run o
--   SET asa_vessel_time = CASE WHEN r.vessel_set THEN NULL ELSE o.asa_vessel_time END,
--       asa_infra_time  = CASE WHEN r.infra_set  THEN NULL ELSE o.asa_infra_time  END,
--       asa_dark_time   = CASE WHEN r.dark_set   THEN NULL ELSE o.asa_dark_time   END
--   FROM asa_backfill_receipt_20261120_pre2023 r WHERE o.id = r.id;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30min';
SET LOCAL enable_nestloop = off;

-- The one value to edit: the start of this week's backlog (UTC).
CREATE TEMP TABLE bf_params ON COMMIT DROP AS
SELECT timestamp '2026-11-13 00:00:00' AS backlog_since;

CREATE TABLE asa_backfill_receipt_20261120_pre2023 (
    id bigint PRIMARY KEY,
    vessel_set boolean NOT NULL,
    infra_set boolean NOT NULL,
    dark_set boolean NOT NULL
);

WITH const AS (
    SELECT p.backlog_since,
           timestamp '2023-01-01'           AS old_scene_before,
           timestamp '2026-03-17 19:49:08'  AS gsp_release,
           timestamp '1970-01-01 00:00:00'  AS unknown_time
    FROM bf_params p
), backlog_trigger AS (
    SELECT DISTINCT h.trigger
    FROM historical_run h CROSS JOIN const
    WHERE NOT h.dry_run AND h.submission_start_time >= const.backlog_since
), old_run AS (
    SELECT o.id, o.sentinel1_grd, o.inference_end_time,
           o.asa_vessel_time IS NULL AS v,
           o.asa_infra_time  IS NULL AS i,
           o.asa_dark_time   IS NULL AS d
    FROM orchestrator_run o
    JOIN sentinel1_grd g ON g.id = o.sentinel1_grd
    CROSS JOIN const
    WHERE g.start_time < const.old_scene_before
      AND o.inference_end_time < const.backlog_since
      AND NOT EXISTS (SELECT 1 FROM backlog_trigger b WHERE b.trigger = o.trigger)
      AND (o.asa_vessel_time IS NULL OR o.asa_infra_time IS NULL OR o.asa_dark_time IS NULL)
), reach AS (
    SELECT s.orchestrator_run AS run,
           bool_and(s.geometric_slick_potential IS NOT NULL
                    OR EXISTS (SELECT 1 FROM slick_to_source x WHERE x.slick = s.id)) AS reached
    FROM slick s
    JOIN old_run r ON r.id = s.orchestrator_run
    WHERE s.active AND s.machine_confidence > 0
    GROUP BY 1
), target AS (
    SELECT r.id, r.v, r.i, r.d
    FROM old_run r
    JOIN reach e ON e.run = r.id
    CROSS JOIN const
    WHERE (e.reached OR r.inference_end_time < const.gsp_release)
      AND NOT EXISTS (SELECT 1 FROM orchestrator_run o2
                      WHERE o2.sentinel1_grd = r.sentinel1_grd
                        AND o2.id <> r.id
                        AND o2.inference_end_time <= r.inference_end_time
                        AND o2.inference_end_time::date = r.inference_end_time::date)
), upd AS (
    UPDATE orchestrator_run o
    -- COALESCE reads the live row, so a stamp written after this snapshot is kept.
    SET asa_vessel_time = COALESCE(o.asa_vessel_time, const.unknown_time),
        asa_infra_time  = COALESCE(o.asa_infra_time,  const.unknown_time),
        asa_dark_time   = COALESCE(o.asa_dark_time,   const.unknown_time)
    FROM target t
    CROSS JOIN const
    WHERE o.id = t.id
    RETURNING o.id, t.v, t.i, t.d
)
INSERT INTO asa_backfill_receipt_20261120_pre2023 SELECT * FROM upd;

-- Checks. Rows marked MUST BE 0 must show 0.
WITH r AS (
    SELECT r.*, o.inference_end_time, o.trigger, g.start_time
    FROM asa_backfill_receipt_20261120_pre2023 r
    JOIN orchestrator_run o USING (id)
    JOIN sentinel1_grd g ON g.id = o.sentinel1_grd
), p AS (
    SELECT backlog_since FROM bf_params
), backlog_trigger AS (
    SELECT DISTINCT h.trigger FROM historical_run h, p
    WHERE NOT h.dry_run AND h.submission_start_time >= p.backlog_since
)
SELECT '1 runs updated' AS check_name, count(*)::text AS value FROM r
UNION ALL
SELECT '2 columns set to 1970: vessel / infra / dark',
       count(*) FILTER (WHERE vessel_set) || ' / ' || count(*) FILTER (WHERE infra_set) || ' / '
       || count(*) FILTER (WHERE dark_set)
FROM r
UNION ALL
SELECT '3 MUST BE 0: scene captured 2023 or later', count(*)::text
FROM r WHERE start_time >= timestamp '2023-01-01'
UNION ALL
SELECT '4 MUST BE 0: run ended on or after backlog_since', count(*)::text
FROM r, p WHERE r.inference_end_time >= p.backlog_since
UNION ALL
SELECT '5 MUST BE 0: run from a backlog trigger', count(*)::text
FROM r WHERE r.trigger IN (SELECT trigger FROM backlog_trigger)
UNION ALL
SELECT '6 MUST BE 0: post-GSP run with a slick lacking GSP and sources', count(*)::text
FROM r
WHERE r.inference_end_time >= timestamp '2026-03-17 19:49:08'
  AND EXISTS (SELECT 1 FROM slick sl
              WHERE sl.orchestrator_run = r.id AND sl.active AND sl.machine_confidence > 0
                AND sl.geometric_slick_potential IS NULL
                AND NOT EXISTS (SELECT 1 FROM slick_to_source x WHERE x.slick = sl.id))
UNION ALL
SELECT '7 MUST BE 0: trigger 860158 run touched', count(*)::text
FROM r WHERE r.trigger = 860158
UNION ALL
SELECT '8 backlog triggers excluded', coalesce(string_agg(trigger::text, ', ' ORDER BY trigger), 'none')
FROM backlog_trigger
UNION ALL
SELECT '9 runs with an active slick still all NULL, by capture date',
       count(*) FILTER (WHERE g.start_time < timestamp '2023-01-01') || ' before 2023 / '
       || count(*) FILTER (WHERE g.start_time >= timestamp '2023-01-01') || ' 2023 or later'
FROM orchestrator_run o
JOIN sentinel1_grd g ON g.id = o.sentinel1_grd
WHERE o.asa_vessel_time IS NULL AND o.asa_infra_time IS NULL AND o.asa_dark_time IS NULL
  AND EXISTS (SELECT 1 FROM slick s WHERE s.orchestrator_run = o.id AND s.active AND s.machine_confidence > 0)
ORDER BY 1;
