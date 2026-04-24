# Cerulean Cloud Agent Instructions

This repository uses `AGENTS.md` as the canonical shared instruction file for AI coding agents. `CLAUDE.md` imports this file for Claude Code.

Repo-local skills live under `.claude/skills`. Treat that directory as the canonical checked-in skill catalog. `.agents/skills` is a symlink mirror to the same catalog for Codex-native discovery. Do not add a second live copy under a bare repo-root `skills/` directory.

## Repo-local skills

Before doing substantial work, inspect the frontmatter of each repo-local `SKILL.md` under `.claude/skills/*/SKILL.md` and load the relevant skill body for any task that matches its `name` or `description`.

Current repo-local skills:

- `.claude/skills/cerulean-cloud-test-env/SKILL.md`
- `.claude/skills/cerulean-database-change-checklist/SKILL.md`
- `.claude/skills/cerulean-db-performance-triage/SKILL.md`
- `.claude/skills/cerulean-published-geoasset-debugging/SKILL.md`
- `.claude/skills/cerulean-slick-classification-boundary/SKILL.md`
- `.claude/skills/cerulean-ui-frontend-sql/SKILL.md`
- `.claude/skills/cerulean-vocabulary-translation-seed/SKILL.md`
- `.claude/skills/skytruth-intent-engineering/SKILL.md`

## Required behavior

- Treat skills under `.claude/skills` as repo-local, not global.
- Prefer an applicable repo-local skill over a similar global skill.
- If multiple repo-local skills apply, use the minimal set that cleanly covers the task.
- Do not ignore an applicable repo-local skill just because it is not surfaced in the session's built-in available-skills list.
- When a task is ambiguous across multiple local skills, inspect the candidate `SKILL.md` files before editing code.
- Keep shared `SKILL.md` files in the portable Agent Skills subset: `name`, `description`, concise Markdown instructions, and optional bundled `references/`, `scripts/`, or `assets/`.
- Put Codex-specific skill metadata in `agents/openai.yaml`.
- Avoid Claude-only skill extensions in shared skills unless the portability tradeoff is intentional and documented in the skill.
- Keep shared skills generic for any maintainer of this checkout. Do not hard-code developer-specific absolute paths, usernames, home directories, shell profiles, machine-local environment names, or setup assumptions; derive paths from the active checkout and use repo-relative examples.

## High-priority triggers

- Use `.claude/skills/cerulean-cloud-test-env/SKILL.md` before running `pytest`, targeted pytest slices, or other test commands in this repo.
- Use `.claude/skills/cerulean-slick-classification-boundary/SKILL.md` for slick classification logic, orchestrator class overrides, land/sea-ice masking classification, or not-oil taxonomy changes.
- Use `.claude/skills/cerulean-published-geoasset-debugging/SKILL.md` when a generated geospatial asset, CRS transform, or published geometry output looks visually wrong or inconsistent with expectations.
- Use `.claude/skills/cerulean-db-performance-triage/SKILL.md` for Cerulean database or map-query performance analysis.
- Use `.claude/skills/cerulean-database-change-checklist/SKILL.md` for migrations, schema changes, lookup data, or DB-backed vocabulary changes.
- Use `.claude/skills/cerulean-vocabulary-translation-seed/SKILL.md` when adding or renaming rows in translatable vocabulary tables or updating `docs/vocabulary_translations.csv`.
- Use `.claude/skills/cerulean-ui-frontend-sql/SKILL.md` when answering frontend SQL questions without the `cerulean-ui` repo open.
- Use `.claude/skills/skytruth-intent-engineering/SKILL.md` for ambiguous prompts, multi-step SkyTruth workflows, publishable artifacts, operational risk, measurement/mapping work, or decisions that need explicit outcomes, constraints, stop rules, and validation.

## Scope

These instructions apply to the entire repository tree rooted here.
