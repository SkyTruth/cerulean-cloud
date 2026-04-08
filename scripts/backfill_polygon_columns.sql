DROP TABLE IF EXISTS slick_geom_backfill_queue;

CREATE UNLOGGED TABLE slick_geom_backfill_queue AS
SELECT s.id
FROM slick s
WHERE s.slick_timestamp >= TIMESTAMPTZ '2000-01-01 00:00:00+00'
  AND s.slick_timestamp <  TIMESTAMPTZ '2027-03-08 00:00:00+00'
  AND s.geometry_count IS NULL
  AND s.largest_area IS NULL
  AND s.median_area IS NULL;

ALTER TABLE slick_geom_backfill_queue
    ADD PRIMARY KEY (id);

ANALYZE slick_geom_backfill_queue;


CREATE OR REPLACE PROCEDURE backfill_slick_geom_fields_from_queue(
    p_batch_size integer DEFAULT 5000,
    p_sleep_sec  double precision DEFAULT 0.02
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_rows integer;
BEGIN
    LOOP
        WITH claimed AS (
            DELETE FROM slick_geom_backfill_queue q
            USING (
                SELECT id
                FROM slick_geom_backfill_queue
                ORDER BY id
                LIMIT p_batch_size
                FOR UPDATE SKIP LOCKED
            ) pick
            WHERE q.id = pick.id
            RETURNING q.id
        ),
        computed AS (
            SELECT
                s.id,
                ST_NumGeometries(s.geometry::geometry) AS geometry_count,
                stats.largest_area,
                stats.median_area
            FROM slick s
            JOIN claimed c ON c.id = s.id
            CROSS JOIN LATERAL (
                SELECT
                    MAX(part.area_m2) AS largest_area,
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY part.area_m2) AS median_area
                FROM (
                    SELECT ST_Area((d.geom)::geography) AS area_m2
                    FROM ST_Dump(s.geometry::geometry) AS d
                ) part
            ) stats
        ),
        updated AS (
            UPDATE slick s
            SET
                geometry_count = c.geometry_count,
                largest_area   = c.largest_area,
                median_area    = c.median_area
            FROM computed c
            WHERE s.id = c.id
            RETURNING 1
        )
        SELECT count(*) INTO v_rows
        FROM updated;

        RAISE NOTICE 'Updated % rows this batch', v_rows;

        COMMIT;

        EXIT WHEN v_rows = 0;

        PERFORM pg_sleep(p_sleep_sec);
    END LOOP;
END;
$$;


select count(*) from slick_geom_backfill_queue

-- CALL backfill_slick_geom_fields_from_queue(10000, 0.02); -- use in PSQL tool
-- drop table slick_geom_backfill_queue;
-- drop procedure backfill_slick_geom_fields_from_queue;