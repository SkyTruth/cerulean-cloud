# Database observability

This repo now manages the database-side prerequisites for statement and PL/pgSQL observability:

- Cloud SQL instance flags:
  - `pg_stat_statements.track = all`
  - `pg_stat_statements.save = on`
  - `track_functions = pl`
- Alembic migration:
  - `CREATE EXTENSION IF NOT EXISTS pg_stat_statements`
  - tighter autovacuum/analyze reloptions on the hot tables
  - `ANALYZE` on `slick`, `slick_to_source`, `slick_to_aoi`, `orchestrator_run`, and `sentinel1_grd`

## Hot table reloptions

The migration applies the same reloptions to these hot tables:

- `autovacuum_enabled = true`
- `toast.autovacuum_enabled = true`
- `autovacuum_vacuum_scale_factor = 0.02`
- `autovacuum_vacuum_threshold = 500`
- `autovacuum_vacuum_insert_scale_factor = 0.02`
- `autovacuum_vacuum_insert_threshold = 500`
- `autovacuum_analyze_scale_factor = 0.01`
- `autovacuum_analyze_threshold = 250`

These thresholds are intentionally more aggressive than the PostgreSQL defaults so row-count and dead-tuple estimates stay fresh enough for plan comparisons on the busiest tables.

## Rollout notes

- Do not run `pg_stat_statements_reset()` before taking a baseline.
- Cloud SQL applies instance flag changes at the instance level and may require a restart before `track_functions` and `pg_stat_statements` settings are active.
- If you patch Cloud SQL flags outside Pulumi, include the full flag set in the request. Cloud SQL overwrites unspecified flags back to their defaults.
- If `pg_stat_statements` already exists in the target database, take a baseline snapshot before applying additional tuning changes.
- If `pg_stat_statements` is being enabled for the first time, take the first post-rollout snapshot and let it accumulate representative workload before comparing plans.
- If the Alembic role cannot create extensions in Cloud SQL, run `CREATE EXTENSION IF NOT EXISTS pg_stat_statements;` once as a sufficiently privileged database user, then rerun migrations.

## Baseline capture

Use the Cloud SQL proxy connection flow from the [README](../README.md#connecting), then run:

```sh
mkdir -p reports/db_observability
psql "$DB_URL" -v ON_ERROR_STOP=1 -f scripts/db_observability_baseline.sql \
  | tee "reports/db_observability/baseline_$(date +%Y%m%d_%H%M%S).txt"
```

For a daily or on-demand report:

```sh
psql "$DB_URL" -v ON_ERROR_STOP=1 -f scripts/db_observability_report.sql \
  | tee "reports/db_observability/report_$(date +%Y%m%d_%H%M%S).txt"
```

## Must-pass checks

- `SELECT extname FROM pg_extension WHERE extname = 'pg_stat_statements';`
- `SHOW track_functions;`
- `SELECT relname, n_mod_since_analyze, last_autoanalyze, last_autovacuum FROM pg_stat_user_tables WHERE relname IN ('slick', 'slick_to_source', 'slick_to_aoi', 'orchestrator_run', 'sentinel1_grd');`
- `SELECT queryid, calls, total_exec_time, mean_exec_time, query FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 25;`
