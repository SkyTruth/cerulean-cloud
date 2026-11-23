-- Checks for scripts/backfill_asa_completion_times.sql. Run inside the backfill transaction,
-- before COMMIT. Rows marked MUST BE 0 must be 0.
WITH stamped AS (
    SELECT b.*, o.inference_end_time, o.trigger, g.scene_id, g.start_time
    FROM asa_backfill_ids b
    JOIN orchestrator_run o USING (id)
    JOIN sentinel1_grd g ON g.id = o.sentinel1_grd
), pending AS (
    SELECT split_part(name, '-', 2) AS scene_id, split_part(name, '-', 3) AS h FROM asa_pending_task
), cutoff AS (
    SELECT timezone('UTC', now()) AS now_utc,
           -- published_through() in cerulean_cloud/cloud_function_asa/main.py
           date_trunc('day', timezone('UTC', now()) - interval '12 hours') - interval '2 days'
               AS published_through
)
SELECT '1 rows updated' AS check_name, count(*)::text AS value FROM stamped
UNION ALL
SELECT '2a vessel / infra / dark set from source rows',
       count(*) FILTER (WHERE asa_vessel_time > '1970-01-01') || ' / ' ||
       count(*) FILTER (WHERE asa_infra_time  > '1970-01-01') || ' / ' ||
       count(*) FILTER (WHERE asa_dark_time   > '1970-01-01')
FROM stamped
UNION ALL
SELECT '2b vessel / infra / dark set to 1970-01-01 (done, time unknown)',
       count(*) FILTER (WHERE asa_vessel_time = '1970-01-01') || ' / ' ||
       count(*) FILTER (WHERE asa_infra_time  = '1970-01-01') || ' / ' ||
       count(*) FILTER (WHERE asa_dark_time   = '1970-01-01')
FROM stamped
UNION ALL
SELECT '2c MUST BE 0: 1970 stamp on a scene captured before assume_done_since', count(*)::text
FROM stamped
WHERE start_time < :'assume_done_since'::timestamp
  AND '1970-01-01' IN (asa_vessel_time, asa_infra_time, asa_dark_time)
UNION ALL
SELECT '3 MUST BE 0: stamped run ended at or after listed_at', count(*)::text
FROM stamped
WHERE inference_end_time >= :'listed_at'::timestamp
  AND (asa_vessel_time IS NOT NULL OR asa_infra_time IS NOT NULL OR asa_dark_time IS NOT NULL)
UNION ALL
SELECT '4 MUST BE 0: VESSEL set with a VESSEL/ALL task pending', count(*)::text
FROM stamped s
WHERE s.asa_vessel_time IS NOT NULL
  AND EXISTS (SELECT 1 FROM pending p WHERE p.scene_id = s.scene_id
                AND (p.h IN ('b5c7aed7cd', 'e3165f6a20', '786f7aa884') OR p.h ~ '^[0-9a-f]{32}$'))
UNION ALL
SELECT '5 MUST BE 0: DARK set with a DARK/ALL task pending', count(*)::text
FROM stamped s
WHERE s.asa_dark_time IS NOT NULL
  AND EXISTS (SELECT 1 FROM pending p WHERE p.scene_id = s.scene_id
                AND (p.h IN ('b5c7aed7cd', '9fdc9ad167', '786f7aa884') OR p.h ~ '^[0-9a-f]{32}$'))
UNION ALL
SELECT '6 MUST BE 0: VESSEL/DARK set on a scene GFW has not published', count(*)::text
FROM stamped s CROSS JOIN cutoff c
WHERE (s.asa_vessel_time IS NOT NULL OR s.asa_dark_time IS NOT NULL)
  AND s.start_time + interval '6 hours' >= c.published_through
UNION ALL
SELECT '7 MUST BE 0: stamped post-GSP run with a slick lacking GSP and sources', count(*)::text
FROM stamped s
WHERE s.inference_end_time >= timestamp '2026-03-17 19:49:08'
  AND (s.asa_vessel_time IS NOT NULL OR s.asa_infra_time IS NOT NULL OR s.asa_dark_time IS NOT NULL)
  AND EXISTS (SELECT 1 FROM slick sl
              WHERE sl.orchestrator_run = s.id AND sl.active AND sl.machine_confidence > 0
                AND sl.geometric_slick_potential IS NULL
                AND NOT EXISTS (SELECT 1 FROM slick_to_source x WHERE x.slick = sl.id))
UNION ALL
SELECT '8 runs #224''s first daily batch would select', count(*)::text
FROM orchestrator_run o
JOIN sentinel1_grd g ON g.id = o.sentinel1_grd
CROSS JOIN cutoff c
WHERE g.start_time >= c.now_utc - interval '32 days'
  AND g.start_time < c.published_through - interval '6 hours'
  AND (o.asa_vessel_time IS NULL OR o.asa_infra_time IS NULL OR o.asa_dark_time IS NULL)
  AND EXISTS (SELECT 1 FROM slick s WHERE s.orchestrator_run = o.id AND s.active AND s.machine_confidence > 0)
UNION ALL
SELECT '9 runs with an active slick still all NULL, by capture age',
       count(*) FILTER (WHERE g.start_time >= c.now_utc - interval '32 days') || ' within 32 d / '
       || count(*) FILTER (WHERE g.start_time < c.now_utc - interval '32 days') || ' older'
FROM orchestrator_run o
JOIN sentinel1_grd g ON g.id = o.sentinel1_grd
CROSS JOIN cutoff c
WHERE o.asa_vessel_time IS NULL AND o.asa_infra_time IS NULL AND o.asa_dark_time IS NULL
  AND EXISTS (SELECT 1 FROM slick s WHERE s.orchestrator_run = o.id AND s.active AND s.machine_confidence > 0)
ORDER BY 1;
