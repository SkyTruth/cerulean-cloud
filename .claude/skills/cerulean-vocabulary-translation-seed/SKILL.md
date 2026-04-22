---
name: cerulean-vocabulary-translation-seed
description: Maintain Cerulean's vocabulary translation seed CSV when DB-backed vocabulary rows are added, renamed, or rewritten. Use when changing translatable rows in `cls`, `tag`, `aoi_type`, `source_type`, `frequency`, `permission`, or `layer`, when adding locales, or when a migration changes human-facing vocabulary that should match the archetypal translation-branch CSV structure.
---

# Cerulean Vocabulary Translation Seed

Use this skill when a Cerulean database change adds or changes DB-backed vocabulary that should be translated through `docs/vocabulary_translations_es_fr_pt_id.csv`.

## Use when

- Add new rows to `cls`, `tag`, `aoi_type`, `source_type`, `frequency`, `permission`, or `layer`.
- Rename a vocabulary `short_name` or other stable natural key used by the translation seed.
- Change a human-facing `long_name`, `description`, `citation`, or `notes` value for a translatable vocabulary row.
- Add or remove translation locales.
- Review whether a DB vocabulary change has complete translation follow-through.

## Do not use when

- The task is frontend-only copy or `dom_i18n_dictionary` work with no DB vocabulary seed changes.
- The task changes freeform user content such as AOI names, ship names, or HITL notes.
- The task touches a database table that is not part of the translation seed archetype and has no plan to join `..._i18n` tables.

Negative examples:

- "Translate a React button label from English to Spanish."
- "Localize a user's saved AOI name."

## Authoritative files

- `alembic/versions/d6c7b48d9f11_add_vocabulary_i18n_tables.py`
- `docs/vocabulary_translations_es_fr_pt_id.csv`

Read those files before editing the CSV.

## Core rules

1. Treat `docs/vocabulary_translations_es_fr_pt_id.csv` as the archetype for structure and coverage.
2. Keep the locale metadata comment lines and CSV header order unchanged unless the locale roster itself changes.
3. Use `context_key` as the stable lookup key. In practice this is the natural key such as `short_name`, not the numeric database ID.
4. Update or remove old rows when a key is renamed. Do not leave stale `context_key` rows behind.
5. Add one CSV row per translatable field, not one row per entity.
6. Fill every seeded locale column for every non-empty human-facing source string. Do not leave a locale blank just because translation is inconvenient.
7. If a value is intentionally identical across languages, repeat it explicitly instead of leaving the cell blank.
8. Only leave translation cells blank when the canonical source field is intentionally empty.
9. Keep one `source_checksum` value across all field rows for the same entity.
10. Compute `source_checksum` as the MD5 of the English source field values joined by `||` in the field order defined below.
11. Treat `entity_id` as informational. Keep it aligned with the canonical row when known, but do not rely on it for portability.

## Field order for `source_checksum`

- `cls`: `long_name||description`
- `tag`: `long_name||description||citation`
- `aoi_type`: `long_name||citation`
- `source_type`: `long_name||citation`
- `frequency`: `long_name`
- `permission`: `long_name`
- `layer`: `long_name||notes||citation`

Every row for a given entity should reuse the same checksum derived from that ordered English payload.

## Workflow

1. Map the affected vocabulary entities.
   - Identify each changed row and its stable key such as `short_name`.
   - Identify which translatable fields changed.
   - Identify whether the change is an add, rename, or text rewrite.

2. Confirm the archetype.
   - Read `TRANSLATION_CONFIG` in `alembic/versions/d6c7b48d9f11_add_vocabulary_i18n_tables.py`.
   - Confirm the entity is one of the seeded translation types and note its translatable fields.

3. Update the CSV rows.
   - Preserve the existing metadata lines and header.
   - For each affected entity, ensure there is exactly one CSV row per translatable field.
   - Keep rows grouped with neighboring rows of the same entity type and key so the file remains readable.
   - If the change renames a key, rewrite the existing rows to the new `context_key` rather than appending duplicates under the new name.

4. Translate thoroughly.
   - Update all seeded locales currently represented in the header: `es`, `fr`, `pt`, `pt-br`, `id`, `sw`.
   - Translate human-facing text, not logic keys.
   - Prefer high-quality natural translations over literal word swaps.
   - Keep product names, organization names, URLs, and citations appropriately preserved when they should remain unchanged.

5. Recompute `source_checksum`.
   - Build the ordered English payload for the entity using the field order above.
   - Join with `||`.
   - Compute the MD5 of that exact UTF-8 string.
   - Reuse the checksum on every field row for that entity.

6. Check rename hazards.
   - Search for the old `context_key` in the CSV and remove or rewrite it as part of the same change.
   - If the DB migration renames a seeded row, ensure the CSV now matches the new key in the same commit.

7. Validate the result.
   - Confirm each affected `(entity_type, context_key)` has exactly one row for each required field.
   - Confirm no duplicate `(entity_type, context_key, field_name)` rows exist.
   - Confirm no old pre-rename keys remain.
   - Confirm all seeded locale columns are populated for non-empty source strings.

## Common failure modes

- A new vocabulary row lands in the DB but never gets translation rows.
- A class rename such as `BACKGROUND -> NOT_OIL` leaves stale CSV rows under the old key.
- A new entity gets only `long_name` translated while `description`, `citation`, or `notes` are missed.
- `source_checksum` is copied from another entity or recomputed inconsistently across rows.
- A translation is left blank even though the source field is non-empty.

## Output contract

Unless the user asks otherwise, return:

1. `Findings`
   - missing translation coverage, stale keys, checksum issues, and locale gaps first
2. `CSV follow-through`
   - affected entities, fields, locale coverage, and rename handling
3. `Validation`
   - what was checked and what was not run
