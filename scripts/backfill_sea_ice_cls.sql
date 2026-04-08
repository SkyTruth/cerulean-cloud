-- One-time backfill for SEA_ICE DB classifications on historical slicks.
--
-- Intent:
--   - Reclassify legacy slick rows that predate sea-ice masking.
--   - Match the branch behavior in cerulean_cloud/cloud_run_orchestrator/handler.py:
--       * leave model inference_idx untouched
--       * update slick.cls to SEA_ICE when the slick is within 1000 m of the
--         resolved MASIE mask
--       * stamp orchestrator_run.sea_ice_date with the mask date that was used
--   - Keep writes short and batchable so frontend reads keep flowing.
--
-- Prerequisites:
--   1) Apply the NOT_OIL / SEA_ICE migration first.
--   2) Load normalized MASIE polygons into public.masie_sea_ice_stage with:
--        mask_date date NOT NULL,
--        geom geometry(MultiPolygon, 4326) NOT NULL
--      Multiple rows per date are expected.
--   3) Call the procedure outside an explicit BEGIN/COMMIT block because it
--      commits each batch internally.
--
-- Notes:
--   - The queue is rebuilt from orchestrator_run rows with sea_ice_date IS NULL,
--     excluding runs already recorded in public.sea_ice_backfill_audit.
--   - The procedure is safe to run in multiple psql sessions at once because it
--     claims queue rows with FOR UPDATE SKIP LOCKED.
--   - By default the procedure stops if it reaches a scene_date that has no
--     loaded mask on or before that date. That is safer than silently auditing
--     partial coverage.
--
-- Example per-date raw import pattern if you do not already have a consolidated
-- stage table:
--   1) Import one NOAA zip into a scratch table with ogr2ogr, reprojected to
--      EPSG:4326 and promoted to multipolygon.
--   2) Append it into public.masie_sea_ice_stage with the desired mask_date.
--
-- Example append after importing a scratch table public.masie_sea_ice_raw_20260401:
-- INSERT INTO public.masie_sea_ice_stage (mask_date, geom)
-- SELECT
--     DATE '2026-04-01',
--     ST_Multi(
--         ST_CollectionExtract(
--             ST_MakeValid(geom),
--             3
--         )
--     )::geometry(MultiPolygon, 4326) AS geom
-- FROM public.masie_sea_ice_raw_20260401
-- WHERE geom IS NOT NULL
--   AND NOT ST_IsEmpty(geom);


-- -----------------------------------------------------------------------------
-- 0) Helper tables
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.masie_sea_ice_stage (
    mask_date date NOT NULL,
    geom geometry(MultiPolygon, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS masie_sea_ice_stage_mask_date_idx
    ON public.masie_sea_ice_stage (mask_date);

CREATE INDEX IF NOT EXISTS masie_sea_ice_stage_geom_gix
    ON public.masie_sea_ice_stage
    USING gist (geom);

CREATE TABLE IF NOT EXISTS public.sea_ice_backfill_audit (
    orchestrator_run_id bigint PRIMARY KEY,
    scene_date date NOT NULL,
    applied_mask_date date,
    updated_slick_count integer NOT NULL,
    processed_at timestamptz NOT NULL DEFAULT now(),
    note text
);

CREATE INDEX IF NOT EXISTS sea_ice_backfill_audit_scene_date_idx
    ON public.sea_ice_backfill_audit (scene_date);


-- -----------------------------------------------------------------------------
-- 1) Validation
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF to_regprocedure('public.get_slick_subclses(bigint)') IS NULL THEN
        RAISE EXCEPTION
            'Missing public.get_slick_subclses(bigint). Apply exposed-function migrations before running this backfill.';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM public.cls
        WHERE short_name = 'NOT_OIL'
    ) THEN
        RAISE EXCEPTION
            'Missing cls.short_name = NOT_OIL. Apply the sea-ice taxonomy migration before running this backfill.';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM public.cls
        WHERE short_name = 'SEA_ICE'
    ) THEN
        RAISE EXCEPTION
            'Missing cls.short_name = SEA_ICE. Apply the sea-ice taxonomy migration before running this backfill.';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'orchestrator_run'
          AND column_name = 'sea_ice_date'
    ) THEN
        RAISE EXCEPTION
            'Missing public.orchestrator_run.sea_ice_date. Apply the sea-ice taxonomy migration before running this backfill.';
    END IF;
END $$;


-- -----------------------------------------------------------------------------
-- 2) Rebuild the queue of candidate runs
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS public.sea_ice_backfill_queue;

CREATE UNLOGGED TABLE public.sea_ice_backfill_queue AS
WITH not_oil_root AS (
    SELECT id
    FROM public.cls
    WHERE short_name = 'NOT_OIL'
),
not_oil_clses AS (
    SELECT c.id
    FROM not_oil_root r
    CROSS JOIN LATERAL public.get_slick_subclses(r.id) AS c
)
SELECT
    o.id AS orchestrator_run_id,
    sg.start_time::date AS scene_date
FROM public.orchestrator_run o
JOIN public.sentinel1_grd sg
  ON sg.id = o.sentinel1_grd
WHERE o.success IS TRUE
  AND o.sea_ice_date IS NULL
  AND NOT EXISTS (
      SELECT 1
      FROM public.sea_ice_backfill_audit a
      WHERE a.orchestrator_run_id = o.id
  )
  AND (
      ST_YMax(sg.geometry::geometry) >= 50
      OR ST_YMin(sg.geometry::geometry) <= -50
  )
  AND EXISTS (
      SELECT 1
      FROM public.slick s
      WHERE s.orchestrator_run = o.id
        AND s.active
        AND NOT EXISTS (
            SELECT 1
            FROM not_oil_clses noc
            WHERE noc.id = s.cls
        )
  );

ALTER TABLE public.sea_ice_backfill_queue
    ADD PRIMARY KEY (orchestrator_run_id);

CREATE INDEX sea_ice_backfill_queue_scene_date_idx
    ON public.sea_ice_backfill_queue (scene_date);

ANALYZE public.sea_ice_backfill_queue;


-- -----------------------------------------------------------------------------
-- 3) Worker procedure
-- -----------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE public.backfill_sea_ice_slick_cls_from_queue(
    p_run_batch_size integer DEFAULT 50,
    p_sleep_sec double precision DEFAULT 0.05,
    p_buffer_m double precision DEFAULT 1000,
    p_stop_on_missing_mask boolean DEFAULT TRUE
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_scene_date date;
    v_mask_date date;
    v_not_oil_root_id integer;
    v_sea_ice_cls_id integer;
    v_claimed_runs integer;
    v_updated_slicks integer;
BEGIN
    SELECT id
    INTO v_not_oil_root_id
    FROM public.cls
    WHERE short_name = 'NOT_OIL';

    SELECT id
    INTO v_sea_ice_cls_id
    FROM public.cls
    WHERE short_name = 'SEA_ICE';

    IF v_not_oil_root_id IS NULL OR v_sea_ice_cls_id IS NULL THEN
        RAISE EXCEPTION
            'Missing NOT_OIL or SEA_ICE cls rows. Apply the sea-ice taxonomy migration before running this procedure.';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM public.masie_sea_ice_stage) THEN
        RAISE EXCEPTION
            'public.masie_sea_ice_stage is empty. Load MASIE masks before calling the backfill procedure.';
    END IF;

    LOOP
        v_scene_date := NULL;
        v_mask_date := NULL;
        v_claimed_runs := 0;
        v_updated_slicks := 0;

        SELECT q.scene_date
        INTO v_scene_date
        FROM public.sea_ice_backfill_queue q
        ORDER BY q.scene_date, q.orchestrator_run_id
        LIMIT 1;

        EXIT WHEN v_scene_date IS NULL;

        SELECT MAX(mask_date)
        INTO v_mask_date
        FROM public.masie_sea_ice_stage
        WHERE mask_date <= v_scene_date;

        IF v_mask_date IS NULL AND p_stop_on_missing_mask THEN
            RAISE EXCEPTION
                'No mask loaded in public.masie_sea_ice_stage on or before scene_date %.',
                v_scene_date;
        END IF;

        WITH claimed AS (
            DELETE FROM public.sea_ice_backfill_queue q
            USING (
                SELECT orchestrator_run_id, scene_date
                FROM public.sea_ice_backfill_queue
                WHERE scene_date = v_scene_date
                ORDER BY orchestrator_run_id
                LIMIT p_run_batch_size
                FOR UPDATE SKIP LOCKED
            ) pick
            WHERE q.orchestrator_run_id = pick.orchestrator_run_id
            RETURNING q.orchestrator_run_id, q.scene_date
        ),
        not_oil_clses AS (
            SELECT c.id
            FROM public.get_slick_subclses(v_not_oil_root_id) AS c
        ),
        candidate_slicks AS (
            SELECT
                s.id,
                s.orchestrator_run
            FROM public.slick s
            JOIN claimed c
              ON c.orchestrator_run_id = s.orchestrator_run
            WHERE s.active
              AND NOT EXISTS (
                  SELECT 1
                  FROM not_oil_clses noc
                  WHERE noc.id = s.cls
              )
        ),
        matched_slicks AS (
            SELECT DISTINCT
                cs.id,
                cs.orchestrator_run
            FROM candidate_slicks cs
            JOIN public.slick s
              ON s.id = cs.id
            JOIN public.masie_sea_ice_stage sm
              ON sm.mask_date = v_mask_date
             AND ST_DWithin(s.geometry, sm.geom::geography, p_buffer_m)
        ),
        updated AS (
            UPDATE public.slick s
            SET cls = v_sea_ice_cls_id
            FROM matched_slicks ms
            WHERE s.id = ms.id
              AND s.cls IS DISTINCT FROM v_sea_ice_cls_id
            RETURNING ms.orchestrator_run
        ),
        updated_counts AS (
            SELECT
                orchestrator_run,
                COUNT(*)::integer AS updated_slick_count
            FROM updated
            GROUP BY orchestrator_run
        ),
        marked_runs AS (
            UPDATE public.orchestrator_run o
            SET sea_ice_date = v_mask_date
            FROM claimed c
            WHERE o.id = c.orchestrator_run_id
              AND v_mask_date IS NOT NULL
              AND o.sea_ice_date IS DISTINCT FROM v_mask_date
            RETURNING o.id
        ),
        audit_rows AS (
            INSERT INTO public.sea_ice_backfill_audit (
                orchestrator_run_id,
                scene_date,
                applied_mask_date,
                updated_slick_count,
                processed_at,
                note
            )
            SELECT
                c.orchestrator_run_id,
                c.scene_date,
                v_mask_date,
                COALESCE(uc.updated_slick_count, 0),
                now(),
                CASE
                    WHEN v_mask_date IS NULL THEN 'NO_MASK_LOADED_ON_OR_BEFORE_SCENE_DATE'
                    ELSE NULL
                END
            FROM claimed c
            LEFT JOIN updated_counts uc
              ON uc.orchestrator_run = c.orchestrator_run_id
            ON CONFLICT (orchestrator_run_id) DO UPDATE
            SET scene_date = EXCLUDED.scene_date,
                applied_mask_date = EXCLUDED.applied_mask_date,
                updated_slick_count = EXCLUDED.updated_slick_count,
                processed_at = EXCLUDED.processed_at,
                note = EXCLUDED.note
            RETURNING 1
        )
        SELECT
            COALESCE((SELECT COUNT(*) FROM audit_rows), 0),
            COALESCE((SELECT SUM(updated_slick_count) FROM updated_counts), 0)
        INTO v_claimed_runs, v_updated_slicks;

        IF v_claimed_runs = 0 THEN
            COMMIT;
            CONTINUE;
        END IF;

        RAISE NOTICE
            'Processed scene_date %, mask_date %, claimed runs %, updated slicks %',
            v_scene_date,
            v_mask_date,
            v_claimed_runs,
            v_updated_slicks;

        COMMIT;
        PERFORM pg_sleep(p_sleep_sec);
    END LOOP;
END;
$$;


-- -----------------------------------------------------------------------------
-- 4) Helpful inspection queries
-- -----------------------------------------------------------------------------
SELECT COUNT(*) AS queued_runs
FROM public.sea_ice_backfill_queue;

SELECT COUNT(DISTINCT scene_date) AS queued_scene_dates
FROM public.sea_ice_backfill_queue;

SELECT
    scene_date,
    COUNT(*) AS queued_run_count
FROM public.sea_ice_backfill_queue
GROUP BY scene_date
ORDER BY scene_date
LIMIT 25;

SELECT
    q.scene_date,
    COUNT(*) AS queued_run_count
FROM public.sea_ice_backfill_queue q
LEFT JOIN LATERAL (
    SELECT MAX(mask_date) AS applied_mask_date
    FROM public.masie_sea_ice_stage sm
    WHERE sm.mask_date <= q.scene_date
) sm ON TRUE
WHERE sm.applied_mask_date IS NULL
GROUP BY q.scene_date
ORDER BY q.scene_date;

SELECT
    applied_mask_date,
    COUNT(*) AS processed_runs,
    COALESCE(SUM(updated_slick_count), 0) AS updated_slicks
FROM public.sea_ice_backfill_audit
GROUP BY applied_mask_date
ORDER BY applied_mask_date DESC NULLS LAST
LIMIT 25;


-- -----------------------------------------------------------------------------
-- 5) Example worker calls
-- -----------------------------------------------------------------------------
-- Run one worker:
-- CALL public.backfill_sea_ice_slick_cls_from_queue(50, 0.05, 1000, TRUE);
--
-- Run multiple workers in parallel:
--   open 2-4 psql sessions and run the same CALL in each one
--
-- Rebuild the queue after loading more masks:
--   \i scripts/backfill_sea_ice_cls.sql
--
-- Optional cleanup after review:
-- DROP TABLE IF EXISTS public.sea_ice_backfill_queue;
-- DROP PROCEDURE IF EXISTS public.backfill_sea_ice_slick_cls_from_queue(integer, double precision, double precision, boolean);
-- DROP TABLE IF EXISTS public.sea_ice_backfill_audit;
-- DROP TABLE IF EXISTS public.masie_sea_ice_stage;
