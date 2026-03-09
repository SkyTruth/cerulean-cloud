# Hot-path index pack

This index pack targets only the read paths we already know are hot:

- Latest HITL lookup per slick from `slick_plus`
- User-specific latest HITL lookup
- Active per-source lookup in `get_slicks_by_source`
- Active per-slick ranking and best-score lookups in `slick_to_source`
- HITL request inbox by user and newest request
- Vessel flag lookup by flag and `source_id`

## Indexes

The migration adds these indexes with `CREATE INDEX CONCURRENTLY`:

- `idx_hitl_slick_slick_update_desc` on `hitl_slick (slick, update_time DESC)`
- `idx_hitl_slick_user_slick_update_desc` on `hitl_slick ("user", slick, update_time DESC)`
- `idx_sts_active_source_rank_slick` on `slick_to_source (source, rank, slick) WHERE active`
- `idx_sts_active_slick_score_desc` on `slick_to_source (slick, collated_score DESC) WHERE active`
- `idx_sts_active_slick_rank` on `slick_to_source (slick, rank) WHERE active`
- `idx_hitl_request_user_date_desc` on `hitl_request ("user", date_requested DESC)`
- `idx_source_vessel_flag_source` on `source_vessel (flag, source_id)`

## Evidence

Observed in repo:

- `slick_plus` fetches latest HITL with `ORDER BY hs.update_time DESC LIMIT 1`
- `get_slicks_by_source` filters on `sts.source`, `sts.active`, and `sts.rank`
- `slick_to_source` re-ranking and score ordering paths use `WHERE active` plus `ORDER BY collated_score DESC` or `ORDER BY rank`

Observed in user-provided live patterns:

- user-specific latest HITL
- HITL request newest-first per user
- vessel flag lookup by `flag` and `source_id`

## Capture before/after plans

Run the same plan script before and after applying the migration:

```sh
mkdir -p reports/db_observability
psql "$DB_URL" -v ON_ERROR_STOP=1 -f scripts/db_hot_path_index_explain.sql \
  | tee "reports/db_observability/hot_path_before_$(date +%Y%m%d_%H%M%S).txt"
```

Apply the migration, then rerun:

```sh
psql "$DB_URL" -v ON_ERROR_STOP=1 -f scripts/db_hot_path_index_explain.sql \
  | tee "reports/db_observability/hot_path_after_$(date +%Y%m%d_%H%M%S).txt"
```

## What to compare

- Access path changes: look for seq scans or wide bitmap scans turning into index scans or index-only scans on the new indexes.
- Latency: compare `Execution Time` for the same sample-backed query.
- Buffer usage: look for fewer shared reads/hits on the bad paths.
- Planner cost: compare total cost, but treat runtime and buffers as the stronger signal.

If the before/after plans look unchanged, first verify that:

- the migration completed successfully,
- the new indexes are valid in `pg_indexes`,
- table stats are current enough for the planner to use them.
