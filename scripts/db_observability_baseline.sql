\pset footer off

\echo === Baseline metadata ===
SELECT
    now() AT TIME ZONE 'UTC' AS captured_at_utc,
    current_database() AS database_name,
    current_user AS database_user;

\echo === Extension status ===
SELECT extname, extversion
FROM pg_extension
WHERE extname = 'pg_stat_statements';

\echo === Instance and session settings ===
SELECT name, setting, unit, context
FROM pg_settings
WHERE name IN (
    'autovacuum',
    'autovacuum_analyze_scale_factor',
    'autovacuum_analyze_threshold',
    'autovacuum_vacuum_insert_scale_factor',
    'autovacuum_vacuum_insert_threshold',
    'autovacuum_vacuum_scale_factor',
    'autovacuum_vacuum_threshold',
    'pg_stat_statements.save',
    'pg_stat_statements.track',
    'track_functions'
)
ORDER BY name;

\echo === Hot table reloptions ===
SELECT
    c.relname AS table_name,
    COALESCE(array_to_string(c.reloptions, ', '), '(defaults)') AS reloptions
FROM pg_class AS c
JOIN pg_namespace AS n
  ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
  AND c.relname IN (
      'slick',
      'slick_to_source',
      'slick_to_aoi',
      'orchestrator_run',
      'sentinel1_grd'
  )
ORDER BY c.relname;

\echo === Hot table vacuum and analyze stats ===
SELECT
    schemaname,
    relname AS table_name,
    n_live_tup,
    n_dead_tup,
    n_mod_since_analyze,
    last_autovacuum,
    last_autoanalyze,
    vacuum_count,
    autovacuum_count,
    analyze_count,
    autoanalyze_count
FROM pg_stat_user_tables
WHERE relname IN (
    'slick',
    'slick_to_source',
    'slick_to_aoi',
    'orchestrator_run',
    'sentinel1_grd'
)
ORDER BY relname;

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
