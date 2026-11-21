-- Backfill orchestrator_run.asa_{vessel,infra,dark}_time for runs the per-scene ASA flow
-- handled, before the daily ASA batch deploys. asa_natural_time is not tracked and stays NULL.
-- Ran on prod (prod20240903) on 2026-11-20 and updated 322,194 runs.
--
-- Value per run and type:
--   NULL                 still owed: a VESSEL/DARK/ALL/notebook task is queued, GFW has not
--                        published the scene's AIS (VESSEL, DARK), the run is a same-day
--                        re-run without sources, or ASA never reached every slick (run skipped)
--   max(create_time)     the type's latest source association on the run's active slicks
--   1970-01-01 00:00:00  assumed done, time unknown: scene captured on or after
--                        :assume_done_since (default 2023-01-01) and no evidence either way
--   NULL                 the same case for older scenes
--
-- Operator steps (prod: project cerulean-338116, region europe-west1):
--   0. \d orchestrator_run   -- the asa_* columns must exist (Alembic does not add them)
--   1. gcloud tasks queues pause "$ASA_QUEUE" --location=europe-west1 --project=cerulean-338116
--      then wait 10 minutes (function timeout is 540 s) so running tasks finish.
--   2. LISTED_AT=$(date -u '+%Y-%m-%d %H:%M:%S')
--      gcloud tasks list --queue="$ASA_QUEUE" --location=europe-west1 --project=cerulean-338116 \
--          --format='value(name.basename())' > asa_pending.txt
--   3. From the repo root, in the same shell as step 2 (LISTED_AT is a shell variable):
--      psql <connection> -v listed_at="$LISTED_AT"      [-v assume_done_since=2023-01-01]
--        CREATE TEMP TABLE asa_pending_task (name text);
--        \copy asa_pending_task FROM 'asa_pending.txt'
--        BEGIN;
--        \i scripts/backfill_asa_completion_times.sql
--        \i scripts/backfill_asa_completion_times_checks.sql
--        \copy asa_backfill_ids TO 'asa_backfill_ids.csv' CSV HEADER
--        COMMIT;            -- promptly: the UPDATE holds row locks until then
--   4. Deploy #224, then: gcloud tasks queues resume "$ASA_QUEUE" ...
--
-- Rollback after COMMIT (before #224 writes anything): load asa_backfill_ids.csv and
--   UPDATE orchestrator_run o SET asa_vessel_time = NULL, asa_infra_time = NULL, asa_dark_time = NULL
--   FROM ids WHERE o.id = ids.id;

-- Stop at the first error and return to the prompt; then type ROLLBACK;
\set ON_ERROR_STOP on
\if :{?listed_at}
\else
    \set listed_at ''
\endif
SELECT :'listed_at' ~ '^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$' AS listed_at_ok \gset
\if :listed_at_ok
\else
    \echo 'listed_at is missing or empty. Set it to the UTC time the queue listing started, for example:'
    \echo '    \\set listed_at ''2026-11-20 14:05:00'''
    DO $$ BEGIN RAISE EXCEPTION 'listed_at is missing or empty'; END $$;
\endif
\if :{?assume_done_since}
\else
    \set assume_done_since '2023-01-01'
\endif

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30min';
-- Row estimates for grouped CTE output default to 200, which makes the planner pick
-- nested loops that compare every run with every other; hash joins keep it linear.
SET LOCAL enable_nestloop = off;

-- Refuse an empty, foreign, or post-#224 listing.
DO $$
DECLARE
    v_total int;
    v_bad   text;
BEGIN
    SELECT count(*) INTO v_total FROM asa_pending_task;
    IF v_total = 0 THEN
        RAISE EXCEPTION 'asa_pending_task is empty: list the paused ASA queue first';
    END IF;
    SELECT string_agg(DISTINCT CASE WHEN name !~ '^asa-S1[A-D]_[A-Z0-9_]+-' THEN left(name, 80)
                                    ELSE split_part(name, '-', 3) END, ', ') INTO v_bad
    FROM asa_pending_task
    WHERE name !~ '^asa-S1[A-D]_[A-Z0-9_]+-'
       OR split_part(name, '-', 3) NOT IN ('b5c7aed7cd',   -- ALL
                                           'e3165f6a20',   -- VESSEL
                                           '9fdc9ad167',   -- DARK
                                           '786f7aa884')   -- DARK-VESSEL
          AND split_part(name, '-', 3) !~ '^[0-9a-f]{32}$'; -- TriggerOnDemand notebook task
    IF v_bad IS NOT NULL THEN
        RAISE EXCEPTION 'unexpected ASA task names or flag hashes: % (INFRA dcc6fc6061 means #224 is live)', v_bad;
    END IF;
END $$;

CREATE TEMP TABLE asa_backfill_ids (
    id bigint PRIMARY KEY,
    asa_vessel_time timestamp,
    asa_infra_time timestamp,
    asa_dark_time timestamp
) ON COMMIT PRESERVE ROWS;

WITH const AS (
    SELECT date_trunc('day', timezone('UTC', now())) - interval '3 days'  AS published_through,
           timestamp '2026-03-17 19:49:08'                                AS gsp_release,  -- tag 1.2.18 prod deploy finished
           :'listed_at'::timestamp                                        AS listed_at,
           :'assume_done_since'::timestamp                                AS assume_done_since,
           timestamp '1970-01-01 00:00:00'                                AS unknown_time
), pending AS (
    SELECT split_part(name, '-', 2) AS scene_id,
           bool_or(split_part(name, '-', 3) IN ('b5c7aed7cd') OR split_part(name, '-', 3) ~ '^[0-9a-f]{32}$')
               AS infra,
           bool_or(split_part(name, '-', 3) IN ('b5c7aed7cd', 'e3165f6a20', '786f7aa884')
                   OR split_part(name, '-', 3) ~ '^[0-9a-f]{32}$') AS vessel,
           bool_or(split_part(name, '-', 3) IN ('b5c7aed7cd', '9fdc9ad167', '786f7aa884')
                   OR split_part(name, '-', 3) ~ '^[0-9a-f]{32}$') AS dark
    FROM asa_pending_task
    GROUP BY 1
), eligible AS (
    SELECT s.id, s.orchestrator_run AS run,
           s.geometric_slick_potential IS NOT NULL
               OR EXISTS (SELECT 1 FROM slick_to_source x WHERE x.slick = s.id) AS reached
    FROM slick s
    WHERE s.active AND s.machine_confidence > 0
), run AS (
    SELECT o.id,
           CASE WHEN g.start_time >= const.assume_done_since THEN const.unknown_time END AS fallback,
           g.start_time + interval '6 hours' >= const.published_through AS ais_unpublished,
           COALESCE(p.infra, false)  AS infra_pending,
           COALESCE(p.vessel, false) AS vessel_pending,
           COALESCE(p.dark, false)   AS dark_pending,
           -- Same scene re-orchestrated earlier that UTC day: this run's ASA task may have
           -- been deduplicated, and a later retry can set GSP without running INFRA.
           EXISTS (SELECT 1 FROM orchestrator_run o2
                   WHERE o2.sentinel1_grd = o.sentinel1_grd
                     AND o2.id <> o.id
                     AND o2.inference_end_time <= o.inference_end_time
                     AND o2.inference_end_time::date = o.inference_end_time::date) AS dedup_suspect
    FROM orchestrator_run o
    JOIN sentinel1_grd g ON g.id = o.sentinel1_grd
    JOIN (SELECT run, bool_and(reached) AS reached FROM eligible GROUP BY run) e ON e.run = o.id
    LEFT JOIN pending p ON p.scene_id = g.scene_id
    CROSS JOIN const
    WHERE o.inference_end_time < const.listed_at      -- its ASA task existed before the listing
      AND (e.reached OR o.inference_end_time < const.gsp_release)
      AND o.asa_vessel_time IS NULL AND o.asa_infra_time IS NULL AND o.asa_dark_time IS NULL
), seen AS (
    SELECT e.run,
           max(sts.create_time) FILTER (WHERE st.short_name = 'VESSEL') AS vessel,
           max(sts.create_time) FILTER (WHERE st.short_name = 'INFRA')  AS infra,
           max(sts.create_time) FILTER (WHERE st.short_name = 'DARK')   AS dark
    FROM eligible e
    JOIN slick_to_source sts ON sts.slick = e.id
    JOIN source src ON src.id = sts.source
    JOIN source_type st ON st.id = src.type
    GROUP BY e.run
), upd AS (
    UPDATE orchestrator_run o
    SET asa_infra_time  = CASE
            WHEN run.infra_pending THEN NULL
            WHEN run.dedup_suspect THEN seen.infra
            ELSE COALESCE(seen.infra, run.fallback) END,
        asa_vessel_time = CASE
            WHEN run.vessel_pending OR run.ais_unpublished THEN NULL
            WHEN run.dedup_suspect THEN seen.vessel
            ELSE COALESCE(seen.vessel, run.fallback) END,
        asa_dark_time   = CASE
            WHEN run.dark_pending OR run.ais_unpublished THEN NULL
            WHEN run.dedup_suspect THEN seen.dark
            ELSE COALESCE(seen.dark, run.fallback) END
    FROM run
    LEFT JOIN seen ON seen.run = run.id
    WHERE o.id = run.id
      -- Re-checked on the live row if another session stamped it after this snapshot.
      AND o.asa_vessel_time IS NULL AND o.asa_infra_time IS NULL AND o.asa_dark_time IS NULL
    RETURNING o.id, o.asa_vessel_time, o.asa_infra_time, o.asa_dark_time
)
INSERT INTO asa_backfill_ids SELECT * FROM upd;
