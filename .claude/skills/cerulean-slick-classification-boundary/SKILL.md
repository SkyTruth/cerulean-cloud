---
name: cerulean-slick-classification-boundary
description: Keep Cerulean model taxonomy, DB slick taxonomy, triggers, and not-oil subtree handling aligned while favoring parsimony over speculative generality.
---

# Cerulean Slick Classification Boundary

Use this skill when changing Cerulean classification logic that touches any of:
- `cls` table rows or subclasses
- `slick.cls` or `hitl_cls`
- `model.cls_map`
- `inference_idx`
- insert triggers that derive `slick.cls`
- not-oil exclusion logic in SQL or Python
- orchestrator class overrides for land, sea ice, or similar non-oil masks

Do not use this skill for unrelated work such as networking, UI polish, dependency updates, or generic geospatial helpers.

## Intent

Keep the boundary between model taxonomy and DB taxonomy explicit:

- Model taxonomy:
  - lives in `model.cls_map`
  - controls `inference_idx`
  - may legitimately continue to use `BACKGROUND`
- DB taxonomy:
  - lives in `cls`, `slick.cls`, and `hitl_cls`
  - may rename or refine the not-oil subtree for product/HITL use
  - should support subclasses such as `LAND`, `SEA_ICE`, `ARTEFACT`

The default stance is parsimony:
- prefer the narrowest change that satisfies the requirement
- do not add caches, wrappers, compatibility shims, or data backfills unless there is observed evidence they are needed

## When To Use

Use this skill when the task involves one or more of:
- adding or renaming slick classes
- changing how `slick.cls` is populated
- introducing new not-oil subclasses
- reconciling `BACKGROUND` in model outputs with DB-side `NOT_OIL`
- updating SQL filters that mean “exclude not-oil”
- modifying orchestrator overrides for land/sea-ice/non-oil cases
- adding constraints such as `slick.cls NOT NULL`

## When Not To Use

Do not use this skill when:
- the task is only about model quality, thresholds, or training outputs with no DB taxonomy change
- the task is only about sea-ice file discovery, caching, or GCS plumbing and does not alter class semantics
- the task is only about frontend presentation of already-classified data

Negative examples:
- “Tune UNet thresholds to reduce vessel false positives.”
- “Fix Titiler timeout handling for large scenes.”
- “Add a new map layer to the frontend.”
- “Refactor centerline generation for performance.”

## Required Workflow

1. Identify the layer boundary before editing.
- Ask: is this change model-facing, DB-facing, or both?
- Default assumption: user requests about product/HITL classes are DB-facing unless they explicitly mention retraining or changing `cls_map`.

2. Inspect these components before changing anything.
- The `cls` taxonomy migration path.
- The insert trigger that derives `slick.cls`.
- The orchestrator override path for explicit DB class overrides.
- Any SQL/Python filters that mean “exclude not-oil.”
- Any use of `model.cls_map` or `background_class_idx`.

3. Preserve model taxonomy unless explicitly told otherwise.
- Do not rename `BACKGROUND` in `model.cls_map` unless the user explicitly asks for a model-taxonomy change.
- If DB class `1` is renamed from `BACKGROUND` to `NOT_OIL`, translate `BACKGROUND -> NOT_OIL` at the DB-mapping layer rather than rewriting model taxonomy by default.

4. Prefer existing recursive DB mechanisms over hard-coded subclass lists.
- If the meaning is “anything under class 1 / Not Oil,” use the existing recursive helper where practical:
  - `public.get_slick_subclses(1)`
- Only keep hard-coded lists when the code path truly needs named direct lookups rather than subtree semantics.

5. Treat explicit DB overrides as first-class.
- If the orchestrator detects land, sea ice, or another explicit not-oil subclass, persist that as `slick.cls`.
- Preserve explicit overrides in insert-trigger logic rather than overwriting them from `inference_idx`.

6. Do not invent defensive data fixes without evidence.
- If proposing a backfill, null cleanup, or compatibility shim, tie it to observed database state or a proven migration hazard.
- If there is no evidence, omit the extra path.

7. Enforce invariants in the database when they are true invariants.
- If `slick.cls` must always be present, prefer `NOT NULL` over relying on convention alone.

8. If the user asks for a plan first, stop and give the plan before editing.

## Repo-Specific Rules

- `inference_idx` is model output state, not the product taxonomy.
- `slick.cls` is the DB-side slick classification.
- `hitl_cls` is the human-facing override taxonomy.
- Do not conflate `BACKGROUND` in `cls_map` with the DB short name unless you have explicitly verified they are intentionally identical.
- When simplifying, remove one-off helpers and caches that do not buy anything in a one-scene-per-run path.

## Output Checklist

Before finalizing a classification/taxonomy change, verify:
- model taxonomy stayed untouched unless explicitly requested
- DB taxonomy change is localized to the right layer
- insert-trigger behavior is still correct
- explicit DB overrides still win
- “exclude not-oil” logic covers the full subtree cleanly
- no speculative compatibility code remains
- compile checks and diff hygiene pass
