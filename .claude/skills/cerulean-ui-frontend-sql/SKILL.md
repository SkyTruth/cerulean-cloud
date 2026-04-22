---
name: cerulean-ui-frontend-sql
description: Reference the current Cerulean UI/frontend SQL inventory. Use when asked about cerulean-ui endpoint SQL, frontend query shapes, shared query-builder fragments, source profile SQL, slick detail/search SQL, HITL request SQL, config SQL, or when another conversation needs the preserved frontend SQL snapshot for lookup.
---

# Cerulean UI Frontend SQL

Use this skill when a conversation needs the current frontend SQL query inventory but the `cerulean-ui` repo is not open in the workspace.

## Use when

- The user asks which SQL powers a known frontend endpoint or query-builder file.
- The user wants the current query text for source, slick, HITL, config, auth cleanup, or i18n frontend paths.
- The user wants to compare a backend change against the frontend's current query shape.
- The user wants a quick answer about tables, joins, filters, or selected fields used by the frontend.

## Do not use when

- The task is generic Cerulean database performance diagnosis. Use `cerulean-db-performance-triage`.
- The actual `cerulean-ui` repo is open and should be read directly instead of relying on this snapshot.
- The task is about backend write paths, migrations, or schema changes with no frontend query lookup need.

## Workflow

1. Read [references/frontend-sql-inventory.md](references/frontend-sql-inventory.md).
2. Match the user's endpoint or file path to the corresponding section.
3. State whether the SQL comes from direct endpoint SQL or a shared query-builder fragment.
4. Treat the inventory as a stored snapshot. If live accuracy matters and the UI repo is available, verify against live code.
5. If the user asks for fixes or impact analysis, separate frontend query-shape changes from backend/index/schema changes.
6. Preserve the stored SQL verbatim when quoting or comparing it. Label any behavior beyond the text as inferred.

## Notes

- This inventory is a stored snapshot added on 2026-03-11 from a user-provided frontend SQL dump.
- The reference file intentionally groups queries by frontend file path so future conversations can jump straight to the relevant endpoint.
