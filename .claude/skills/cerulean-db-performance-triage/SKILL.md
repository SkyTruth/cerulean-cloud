---
name: cerulean-db-performance-triage
description: Diagnose Cerulean database and map-query performance issues across the backend and UI query layers. Use when reviewing Postgres/PostGIS performance, asking what SQL to run on production, interpreting EXPLAIN plans, ranking fixes by user-visible latency, or designing before/after benchmark queries for slick, source, AOI, HITL, and map-serving paths.
---

# Cerulean DB Performance Triage

Use this skill for Cerulean-specific database performance work. It is intentionally narrow: Postgres/PostGIS schema, query, and read-model issues in the Cerulean backend and UI SQL layers.

## Use when

- The user asks why Cerulean queries are slow.
- The user wants a repo review focused on database performance.
- The user wants exact SQL to run on production for diagnosis.
- The user provides `EXPLAIN` output and wants help interpreting it.
- The user wants fixes ranked by user-visible speed impact.
- The user wants before/after benchmark queries for a proposed optimization.

## Do not use when

- The task is generic Postgres tuning for a different repo.
- The task is implementation-only and the diagnosis is already settled.
- The task is primarily frontend rendering performance with no database angle.
- The task is mostly infra sizing or Cloud SQL operations with no Cerulean query analysis.

## Scope split

- Backend repo responsibilities usually include:
  - schema and migrations
  - indexes
  - triggers and write paths
  - read-model tables and refresh jobs
  - queue/backfill/cutover logic
- UI or frontend repo responsibilities usually include:
  - SQL query builders
  - request-path filtering and pagination
  - tile/list/detail endpoint query shapes
  - response-size guardrails

If only one repo is available, still separate recommendations into backend vs frontend responsibilities.

## Core rules

1. Separate `Observed`, `Inferred`, and `Assumed` claims.
2. Prefer live evidence over static code inspection when the two disagree.
3. Rank fixes by user-visible latency first, not by elegance.
4. Distinguish query-shape problems from missing-index problems.
5. Treat benchmark queries as either:
   - real current-path queries, or
   - explicit `after-style proxy` queries when the future table/path does not exist yet.
6. Do not recommend partitioning or deep type migrations before query shape and read models are fixed.

## Cerulean-specific hotspots

Check these first.

- `slick_plus` and `get_slicks_by_*`
  - Common failure mode: enrich most of the corpus, then filter late.
- `utils/db/slick-query-builder.ts`
  - Common failure mode: build source/HITL candidates before spatial narrowing.
- Source profile and ranking queries
  - Common failure mode: live corpus-wide ranking or `DISTINCT ON` hiding row multiplication.
- HITL review queries
  - Common failure mode: derive "latest state" from `hitl_slick` history on hot paths.
- `slick_to_source`
  - Common failure mode: hot association table is too wide because it stores heavy payload columns.
- `slick`
  - Common failure mode: broad scans over wide rows with multiple geometry variants and large derived payloads.

## Cerulean anti-patterns

Flag these explicitly when present.

- Enrich the world, then filter.
- Aggregate AOIs or sources per slick on the request path for large result sets.
- Use raw `hitl_slick` history to answer "current HITL state" in hot reads.
- Compute source ranking live across the corpus.
- Serve large interactive maps as bulk GeoJSON instead of MVT/generalized geometry.
- Recommend indexes as the primary fix when plans are dominated by full-corpus joins, aggregation, or disk-spilling sorts.

## Standard evidence pack

If the user does not already have live evidence, ask for or provide SQL for:

1. Extension and instrumentation status
   - `pg_stat_statements`
   - `track_functions`
2. Hot-table stats
   - `pg_stat_user_tables`
3. Index inventory and usage
   - `pg_stat_user_indexes`
4. Fanout checks
   - `slick_to_aoi`, `aoi_chunks`, source-match counts
5. Representative IDs
   - busy `source`, `aoi`, `scene_id`, `slick_id`
6. `EXPLAIN (ANALYZE, BUFFERS, VERBOSE)` for:
   - source path
   - AOI path
   - tile/list path
   - any `slick_plus` or `get_slicks_by_*` path

If production risk is a concern, start with `EXPLAIN (BUFFERS, VERBOSE)`.

## Workflow

1. Map the request to the relevant Cerulean query family.
   - `slick_plus` or exposed functions
   - map query builder
   - source profile/ranking
   - HITL/latest-state
   - write-path or trigger cost

2. Inspect the repo code before concluding.
   - backend: migrations, ORM/database client, trigger code, API functions
   - frontend/UI: SQL query builders and endpoint SQL

3. Build a fact table.
   - `Observed`: exact plan nodes, timings, rows, loops, spill, buffers, table sizes
   - `Inferred`: likely dominant causes
   - `Assumed`: anything missing from the evidence

4. Read plans with a user-latency lens.
   Focus on:
   - rows touched before `LIMIT`
   - repeated nested-loop probes
   - sort or hash spill to disk
   - late filters after expensive joins/aggregates
   - row multiplication from tag/HITL joins
   - full-corpus work for small result sets

5. Group fixes into buckets.
   - immediate: indexes, predicates, simple query rewrites
   - medium: filter-first rewrites, current-state tables
   - structural: read models, payload split-outs, async rebuild workflows

6. Rerank by likely user-visible impact.
   Typical order:
   - replace request-time global enrichment
   - filter spatially first
   - precompute source/ranking summaries
   - shrink hot tables
   - add targeted indexes
   - cleanup and long-term architecture work

7. Produce benchmark pairs.
   For each important path, provide:
   - `before`: current production query
   - `after-style proxy`: a query that approximates the future design without requiring the future tables yet
   - a short note explaining what the proxy does and does not prove

## Common Cerulean fix buckets

Use these as patterns, not defaults.

- Current-state HITL read model
  - Goal: stop reading `hitl_slick` history on hot paths.
- `slick_read_model`
  - Goal: replace `slick_plus`-style live enrichment for interactive reads.
- Filter-first map queries
  - Goal: spatially narrow `slick` first, then enrich only matching rows.
- Source profile summary tables
  - Goal: stop recomputing ranking and best-slick-per-scene live.
- Payload split-out
  - Goal: move heavy detail columns out of `slick` and `slick_to_source`.

## Guardrails for recommendations

- Be realistic about scale.
  - Example: a current-state HITL table may simplify logic a lot, but it will not dominate UX if only a tiny fraction of slicks have HITL rows.
- Call out when static review was corrected by live data.
- If a query is already fast, say so and do not pad the roadmap.
- If a proposed fix mainly helps operability or correctness, do not oversell it as a UX improvement.
- If a benchmark query is a proxy, label it clearly.

## Output contract

Unless the user asks otherwise, return:

1. `Findings`
   - ordered by severity or user impact
   - include file/query references where possible
2. `Observed / Inferred / Assumed`
3. `Recommended steps`
   - immediate vs structural
   - backend vs frontend split
4. `Benchmark queries`
   - before and after-style proxy pairs
5. `What this will likely change for users`
   - realistic impact, not aspirational impact

## Good outcomes

This skill is working well when it:

- identifies the dominant cost from live evidence, not intuition
- avoids overfitting to one missing index
- translates database behavior into user-visible latency impact
- produces a roadmap that can be delegated to separate backend and frontend workstreams
- gives the user runnable SQL to confirm the diagnosis before implementation
