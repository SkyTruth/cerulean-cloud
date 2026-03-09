\pset footer off

\echo === Report metadata ===
SELECT now() AT TIME ZONE 'UTC' AS captured_at_utc;

\echo === Hot table stats ===
SELECT
    relname AS table_name,
    n_live_tup,
    n_dead_tup,
    round(
        100 * n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0),
        2
    ) AS dead_tuple_pct,
    n_mod_since_analyze,
    last_autovacuum,
    last_autoanalyze,
    autovacuum_count,
    autoanalyze_count
FROM pg_stat_user_tables
WHERE relname IN (
    'slick',
    'slick_to_source',
    'slick_to_aoi',
    'orchestrator_run',
    'sentinel1_grd'
)
ORDER BY n_dead_tup DESC, relname;

\echo === Top statements by total execution time ===
SELECT
    queryid,
    calls,
    round(total_exec_time::numeric, 2) AS total_exec_time_ms,
    round(mean_exec_time::numeric, 2) AS mean_exec_time_ms,
    rows,
    left(regexp_replace(query, '\s+', ' ', 'g'), 240) AS query_sample
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 25;

\echo === Top PL function timings ===
SELECT
    schemaname,
    funcname,
    calls,
    round(total_time::numeric, 2) AS total_time_ms,
    round(self_time::numeric, 2) AS self_time_ms
FROM pg_stat_user_functions
ORDER BY total_time DESC
LIMIT 25;
