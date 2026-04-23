---
name: cerulean-ui-frontend-sql
description: Reference or refresh the current Cerulean UI/frontend SQL inventory. Use when asked about cerulean-ui endpoint SQL, frontend query shapes, shared query-builder fragments, source profile SQL, slick detail/search SQL, HITL request SQL, config SQL, updating this stored snapshot from the live UI repo, or when another conversation needs the preserved frontend SQL snapshot for lookup.
---

# Cerulean UI Frontend SQL

Use this skill when a conversation needs the current frontend SQL query inventory but the `cerulean-ui` repo is not open in the workspace.

## Use when

- The user asks which SQL powers a known frontend endpoint or query-builder file.
- The user wants the current query text for source, slick, HITL, config, auth cleanup, or i18n frontend paths.
- The user wants to compare a backend change against the frontend's current query shape.
- The user wants a quick answer about tables, joins, filters, or selected fields used by the frontend.
- The user asks to refresh, update, or regenerate this skill's stored frontend SQL inventory from `cerulean-ui`.

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

## Updating This Skill

When asked to refresh or update this skill from `cerulean-ui`, update this skill folder rather than creating a new skill or a second reference copy.

1. Locate the active backend checkout with `git rev-parse --show-toplevel`.
2. Locate the UI checkout from a user-provided path or a sibling path such as `../cerulean-ui`; do not hard-code developer-specific absolute paths in the skill.
3. Record the UI commit with `git -C "$UI_REPO" log -1 --format='%H %cs %s'`.
4. Check whether query-bearing UI paths are dirty:

```bash
git -C "$UI_REPO" status --short -- pages/api utils/db libs/db
```

If those paths are dirty, state that in the refreshed inventory header and be explicit about whether the snapshot includes working-tree changes or only committed code.

5. Discover live SQL surfaces with a focused search:

```bash
rg -l 'pgSql`|pgQuery\(' "$UI_REPO/pages/api" "$UI_REPO/utils/db" "$UI_REPO/libs/db" -g '*.ts' -g '*.tsx'
```

6. Compare the discovered files with the inventory index. Add new query files, remove sections for paths that no longer exist, and keep endpoint sections grouped by frontend file path.
7. Normalize SQL from `pgSql` tagged templates and `pgQuery` calls into readable SQL. Use placeholders such as `:parameter`, `<identifier>`, and `/* optional */` for dynamic template pieces.
8. For routes that call a shared builder, point to the builder section instead of duplicating the whole query. For routes with no Cerulean database SQL, say so briefly.
9. Update the snapshot header in [references/frontend-sql-inventory.md](references/frontend-sql-inventory.md) with the refresh date, UI commit, and any relevant dirty-state note.
10. Update the note below with the latest refresh date.
11. Validate Markdown and whitespace with:

```bash
git diff --check -- .claude/skills/cerulean-ui-frontend-sql
rg -c '^```' .claude/skills/cerulean-ui-frontend-sql/references/frontend-sql-inventory.md
```

The fence count should be even. Regenerate `agents/openai.yaml` only if the skill's trigger/metadata meaning changes.

## Notes

- This inventory is a stored snapshot added on 2026-03-11 and refreshed on 2026-04-22 from the live `cerulean-ui` checkout.
- The reference file intentionally groups queries by frontend file path so future conversations can jump straight to the relevant endpoint.
