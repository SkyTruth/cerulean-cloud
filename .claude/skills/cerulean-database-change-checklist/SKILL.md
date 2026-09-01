---
name: cerulean-database-change-checklist
description: Review Cerulean database changes that touch Alembic migrations, seed data, translation seed CSV follow-through, generated ORM schema, TiPG exposure, or frontend-consumed DB-backed vocabulary. Use when adding or changing tables, views, constraints, indexes, lookup data, localization tables, translatable vocabulary rows, or when deciding what else in the repo must change after a schema update.
---

# Cerulean Database Change Checklist

Use this skill for Cerulean-specific database changes. Focus on the repo follow-through that is easy to miss after a migration appears "done".

## Use when

- Add or modify Alembic migrations.
- Add or change tables, columns, constraints, indexes, or views.
- Add or change controlled-vocabulary or localization data.
- Review whether a schema change also needs ORM, TiPG, or frontend SQL follow-up.
- Review staged schema-related changes for correctness and repo consistency.

## Do not use when

- The task is only frontend copy or UI translation with no database dependency.
- The task is generic Postgres advice for another repo.
- The task changes application logic against an existing schema without changing DB structure or seed data.

## Core rules

1. Treat Alembic as the source of truth for schema evolution.
2. Do not hand-edit `cerulean_cloud/database_schema.py` except as part of an intentional regeneration.
3. Do not assume deploy regenerates ORM code. In this repo, deploy runs Alembic, not `sqlacodegen`.
4. Never seed child rows using numeric IDs exported from a different database.
5. Resolve lookup rows by stable natural keys such as `short_name` or locale `code`.
6. Review TiPG exposure explicitly for every new table or view.
7. Keep `short_name` and other logic keys untranslated.
8. When changing translatable vocabulary rows, update `docs/vocabulary_translations.csv` or state explicitly why no seed delta is needed.

## Repo-specific checkpoints

Check these paths explicitly before closing a DB task:

- `alembic/versions/`
- `cerulean_cloud/database_schema.py`
- `cerulean_cloud/database_client.py`
- `docs/vocabulary_translations.csv`
- `stack/cloud_run_tipg.py`
- any frontend SQL inventory or frontend query files if display fields are affected

Search for `# EditTheDatabase` and confirm whether each marker needs action for the current change.

## Workflow

1. Map the database change.
   - Identify schema objects being added or changed.
   - Identify seed or reference data affected.
   - Identify backend readers and writers.
   - Identify frontend-visible DB-backed text fields.

2. Check migration portability.
   - Assume target stacks may have only schema-guaranteed seed data.
   - If translations or child rows depend on lookup data, seed only rows that are guaranteed to exist or resolve rows by natural key at runtime.
   - If environment-specific rows are optional, either skip them cleanly or add a separate follow-up data migration.

3. Check translation-seed follow-through.
   - If the change adds, renames, or rewrites rows in translatable vocabulary tables such as `cls`, `tag`, `aoi_type`, `source_type`, `frequency`, `permission`, or `layer`, update `docs/vocabulary_translations.csv` in the archetypal structure introduced by the translation branch.
   - Use stable natural keys in `context_key` such as `short_name`; do not leave orphaned rows behind after renames.
   - Add one CSV row per translatable field and cover every seeded locale unless the canonical source field is intentionally empty.
   - Keep `source_checksum` aligned across all field rows for the same entity.
   - Load `.claude/skills/cerulean-vocabulary-translation-seed/SKILL.md` when the CSV is touched or a DB vocabulary row might need translation.

4. Check ORM expectations.
   - If runtime code needs ORM classes for new tables, regenerate `cerulean_cloud/database_schema.py` locally and review the diff carefully.
   - If runtime code does not yet query the new tables, say explicitly that ORM regeneration is optional.

5. Check TiPG exposure.
   - Decide whether the new table or view should be publicly readable.
   - Update `RESTRICTED_COLLECTIONS` when needed.
   - Add `TIPG_TABLE_CONFIG` geometry or datetime settings only if the table needs them.

6. Check frontend-visible vocabulary.
   - If the frontend reads DB-backed labels or descriptions, plan the query changes.
   - Prefer `LEFT JOIN ..._i18n` plus `COALESCE(translated, base_value)`.
   - Keep stable keys like `short_name` unchanged.

7. Review generated-schema diffs.
   - Confirm new classes and constraints are backed by actual migrations.
   - Treat unrelated regenerated churn as suspect until explained.
   - Call out any staged ORM change with no matching migration support.

## Common Cerulean failure modes

- A migration succeeds only on one database because it assumes exported IDs are portable.
- A localization seed assumes reference rows exist in every stack.
- A vocabulary row is added or renamed without updating `docs/vocabulary_translations.csv`.
- A translation CSV row uses a stale `context_key`, so the seed loader silently skips it.
- A new table is unintentionally exposed through TiPG.
- A staged `database_schema.py` regeneration includes unrelated schema changes and slips through review.
- A backend or frontend consumer keeps selecting English base columns after i18n tables exist.

## Must-pass checks

- Migration syntax checks pass.
- The migration can run on a database containing only schema-guaranteed seed data.
- No seed step depends on foreign numeric IDs from another environment.
- Translation seed follow-through is either implemented or explicitly not applicable.
- Affected translatable entities have no orphaned pre-rename `context_key` rows left in the CSV.
- TiPG exposure has been reviewed explicitly.
- ORM regeneration expectations are stated clearly.
- Frontend follow-up is either implemented or explicitly called out.

## Output contract

Unless the user asks otherwise, return:

1. `Findings`
   - bugs, regressions, portability risks, and follow-up gaps first
2. `Repo follow-through`
   - ORM, TiPG, frontend SQL, and `# EditTheDatabase` markers
3. `Validation`
   - what was checked and what was not run
