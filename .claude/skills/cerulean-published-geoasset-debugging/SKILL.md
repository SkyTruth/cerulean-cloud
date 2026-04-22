---
name: cerulean-published-geoasset-debugging
description: Debug Cerulean generated geospatial assets when the published output looks wrong, stale, or inconsistent with local expectations. Covers geometry generation, CRS/publication artifacts, deployment drift, and right-sized validation.
---

# Cerulean Published Geoasset Debugging

Use this skill when a Cerulean-generated geospatial asset looks wrong after publication or deployment, especially when the failure could be in geometry generation, CRS transformation, output serialization, or stale runtime deployment.

## Use when

- A generated GeoJSON, GCS asset, or map-served vector output looks visually wrong.
- Local code appears correct but deployed output still looks stale or broken.
- A geospatial pipeline needs debugging across source processing, final publication, and deployment.
- Validity checks pass but the rendered output is still obviously wrong.
- You need to add confidence-building tests without permanently making the suite heavy.

## Do not use when

- The task is a pure database schema or migration review.
- The task is only frontend styling or map UI behavior with no generated asset issue.
- The task is generic Cloud Run deployment work unrelated to a geospatial output.
- The task is purely classification taxonomy or business logic with no geometry/output path.

Negative examples:
- “Rename DB classes for HITL.”
- “Tune a Postgres query.”
- “Change leaflet styling.”
- “Fix GitHub Actions caching.”

## Intent

Separate four failure classes cleanly:

- source-data or generation bug
- geometry/publication bug
- deployment drift or stale runtime
- overgrown validation/test surface

The default stance is:
- inspect the artifact itself
- measure the geometry, do not rely on visual intuition alone
- compare local and deployed behavior explicitly
- keep permanent tests small after the root cause is understood

## Core rules

1. Treat the generated artifact as primary evidence.
2. Distinguish invalid geometry from visually incorrect geometry.
3. Do not assume the deployed service is running the latest code.
4. Prefer one targeted real-source reproduction over many heavy live tests.
5. After proving the root cause, keep only the smallest durable regression coverage.
6. When publication to geographic coordinates is involved, treat that step as a likely failure boundary.
7. Prefer repo-consistent deployment forcing patterns over bespoke fixes unless standardization is intentional.

## Workflow

1. Confirm the symptom.
- Inspect the actual output file or served asset.
- Record feature count, bounds, CRS, and a few suspicious geometries.
- Quantify suspicious behavior such as extreme spans or large coordinate jumps.

2. Split the problem boundary.
- Is the geometry already wrong before final publication?
- Does it become wrong only after reprojection or serialization?
- Is local output correct while deployed output is stale?
- Is the issue visual only, or also reflected in measured geometry structure?

3. Check deployment freshness.
- Compare the relevant service/template to other working services in the repo.
- Look for repo-standard revision forcing such as commit-hash environment changes.
- Do not keep debugging runtime output as a code bug if the service has not actually updated.

4. Reproduce at the smallest informative layer.
- Start with deterministic local/unit coverage around the suspected transformation step.
- If needed, run one real-source reproduction against the exact problematic upstream artifact.
- Use that reproduction to identify the real failure boundary.

5. Fix at the right layer.
- Generation-stage problems: source filtering, polygonization, dissolve, simplify, etc.
- Publication-stage problems: reprojection, dateline handling, densification, serialization.
- Deployment-stage problems: image/template update semantics, stale config, revision forcing.

6. Scale testing back down.
- Keep one or two fast deterministic regressions.
- Remove heavy live/source-backed tests unless they are uniquely valuable.
- Preserve at least one test for the actual failure class, not just incidental implementation details.

## Output contract

Unless the user asks otherwise, return:

1. Symptom summary
2. Failure boundary
   - generation vs publication vs deployment
3. Fix
4. Validation
   - fast tests kept
   - any one-off live/source-backed verification
5. Remaining risk or deploy follow-up

## Good outcomes

This skill is working well when it:
- finds whether the break is in code or in deploy state
- uses direct artifact inspection rather than vague map screenshots alone
- fixes the smallest responsible layer
- avoids bloating the permanent test suite
- leaves behind a compact, reusable regression
