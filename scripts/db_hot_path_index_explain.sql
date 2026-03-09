\pset footer off
\timing on

SET statement_timeout = '5min';

\echo === Latest HITL per slick ===
EXPLAIN (ANALYZE, BUFFERS)
WITH sample_slicks AS (
    SELECT id
    FROM slick
    WHERE active
      AND cls <> 1
    ORDER BY create_time DESC
    LIMIT 200
)
SELECT
    s.id,
    hs.cls,
    hs.update_time
FROM sample_slicks AS s
LEFT JOIN LATERAL (
    SELECT
        hs.cls,
        hs.update_time
    FROM hitl_slick AS hs
    WHERE hs.slick = s.id
    ORDER BY hs.update_time DESC
    LIMIT 1
) AS hs
ON TRUE;

\echo === Latest HITL per user and slick ===
EXPLAIN (ANALYZE, BUFFERS)
WITH sample_pairs AS (
    SELECT
        hs."user",
        hs.slick
    FROM hitl_slick AS hs
    GROUP BY hs."user", hs.slick
    ORDER BY max(hs.update_time) DESC
    LIMIT 200
)
SELECT
    p."user",
    p.slick,
    hs.id,
    hs.update_time
FROM sample_pairs AS p
JOIN LATERAL (
    SELECT
        hs.id,
        hs.update_time
    FROM hitl_slick AS hs
    WHERE hs."user" = p."user"
      AND hs.slick = p.slick
    ORDER BY hs.update_time DESC
    LIMIT 1
) AS hs
ON TRUE;

\echo === Active per-source lookup ===
EXPLAIN (ANALYZE, BUFFERS)
WITH sample_source AS (
    SELECT source
    FROM slick_to_source
    WHERE active
    GROUP BY source
    ORDER BY count(*) DESC
    LIMIT 1
)
SELECT
    sts.slick,
    sts.source,
    sts.rank
FROM slick_to_source AS sts
JOIN sample_source AS ss
  ON ss.source = sts.source
WHERE sts.active
  AND sts.rank <= 3
ORDER BY sts.rank, sts.slick DESC
LIMIT 200;

\echo === Active per-slick best score lookup ===
EXPLAIN (ANALYZE, BUFFERS)
WITH sample_slick AS (
    SELECT slick
    FROM slick_to_source
    WHERE active
    GROUP BY slick
    ORDER BY count(*) DESC
    LIMIT 1
)
SELECT
    sts.source,
    sts.collated_score
FROM slick_to_source AS sts
JOIN sample_slick AS ss
  ON ss.slick = sts.slick
WHERE sts.active
ORDER BY sts.collated_score DESC NULLS LAST
LIMIT 10;

\echo === Active per-slick rank lookup ===
EXPLAIN (ANALYZE, BUFFERS)
WITH sample_slick AS (
    SELECT slick
    FROM slick_to_source
    WHERE active
    GROUP BY slick
    ORDER BY count(*) DESC
    LIMIT 1
)
SELECT
    sts.source,
    sts.rank
FROM slick_to_source AS sts
JOIN sample_slick AS ss
  ON ss.slick = sts.slick
WHERE sts.active
ORDER BY sts.rank
LIMIT 10;

\echo === HITL request inbox by user ===
EXPLAIN (ANALYZE, BUFFERS)
WITH sample_user AS (
    SELECT "user"
    FROM hitl_request
    GROUP BY "user"
    ORDER BY count(*) DESC
    LIMIT 1
)
SELECT
    hr.id,
    hr.slick,
    hr.date_requested
FROM hitl_request AS hr
JOIN sample_user AS su
  ON su."user" = hr."user"
ORDER BY hr.date_requested DESC
LIMIT 200;

\echo === Vessel lookup by flag ===
EXPLAIN (ANALYZE, BUFFERS)
WITH sample_flag AS (
    SELECT flag
    FROM source_vessel
    WHERE flag IS NOT NULL
    GROUP BY flag
    ORDER BY count(*) DESC
    LIMIT 1
)
SELECT
    sv.source_id,
    sv.flag
FROM source_vessel AS sv
JOIN sample_flag AS sf
  ON sf.flag = sv.flag
ORDER BY sv.source_id
LIMIT 200;
