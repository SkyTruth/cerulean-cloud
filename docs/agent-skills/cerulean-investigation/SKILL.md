---
name: cerulean-investigation
description: Investigate potential marine oil slicks in SkyTruth's Cerulean data with reproducible queries and evidence-limited conclusions, by running curl against the public Cerulean API (api.cerulean.skytruth.org) or by computer use on the Cerulean map (cerulean.skytruth.org). Use for slick searches by area, period, vessel or offshore structure, single-detection lookups, exports and comparisons.
---

# Cerulean environmental investigation

Answer environmental questions with Cerulean data that others can inspect and reproduce. You work for your own organization, not SkyTruth: keep Cerulean data separate from your analysis, and never imply that SkyTruth reviewed your conclusions. SkyTruth's terms limit Cerulean use to environmental conservation and license the data CC BY-SA 4.0; credit SkyTruth Cerulean and disclose your modifications ([citation format](references/methodology.md#terms-and-attribution)).

Service details here match the live API and map as of 2026-10-26 UTC. Where the live service differs, follow it and record the difference.

## Two routes

| Route | Use it for | Guide |
|---|---|---|
| `curl` against the public API, `https://api.cerulean.skytruth.org` | counts, tables, exports, anything that must be reproducible | [api.md](references/api.md) |
| Computer use on the map, `https://cerulean.skytruth.org` | imagery, the map's exact default filters, selecting a named EEZ or MPA, what the map shows for a vessel | [computer-use.md](references/computer-use.md) |

Run `curl` from a shell; web-fetch tools have failed to open the API's JSON endpoints. Use no other route: not the map application's internal `/api/...` endpoints, and not scripts that make their own network requests. Process saved responses locally with python3 or jq if needed.

## API quick start

```sh
API=https://api.cerulean.skytruth.org/collections
curl -sS --get "$API/public.slick_plus/items" \
  --data-urlencode 'bbox=11,36,20,43' \
  --data-urlencode 'datetime=2025-05-01T00:00:00Z/2025-05-08T23:59:59Z' \
  --data-urlencode 'filter=machine_confidence >= 0.7 AND max_source_collated_score >= 0' \
  --data-urlencode 'limit=10000' \
  -o slicks.geojson -w 'HTTP %{http_code}\n'
```

This returned 5 detections (3935019, 4110801, 4111397, 4112995, 4112997), the same 5 the map shows for that box and week. A result is complete only when `numberMatched` equals `numberReturned`. [api.md](references/api.md) has recipes for one detection, a vessel or structure, and an EEZ, IHO sea area or MPA.

## Evidence rules

**SkyTruth's Cerulean maintainers warn that the vast majority of unfiltered detections are false positives.** Treat this as a reason to filter and investigate, not as a measured error rate. Never subtract 99%, multiply counts by 0.01, or invent a precision for filtered results.

- Every record is a **potential oil slick detection**: a dark radar pattern that a model outlined. It does not establish oil, volume, source, responsibility or a distinct event.
- **Ignore `cls`**, the CNN's class label (Vessel, Natural, Infrastructure and so on). It is unreliable: do not report it, filter on it or reason from it. Use the latest human review class (`hitl_cls`, `hitl_cls_name`) where one exists; otherwise call the detection unreviewed.
- Human review interprets the same imagery and is not independent confirmation. Say "human-classified apparent oil".
- No score is a probability: 0.8 is not 80%. `machine_confidence` stays in the quality screen only because the map applies it; never rank or interpret by it. Screen sources with the collated score (`max_source_collated_score`, the map's "Slick-source match"), and use the geometric score (`slick_plus.slick_confidence`) only when no collated score exists. **`source_plus.slick_confidence` is the machine score, not the geometric score.**
- A candidate source is an investigation lead. Missing AIS does not show concealment, and no candidate does not show no source. **Intent cannot be determined from space; make no claims of legality or illegality.**
- Detection counts are not spill counts, polygon area is not oil volume, and overlapping polygons cannot be summed as unique area. No detections does not show a clean area. Source rankings are not rankings of polluters.

These rules apply to titles, table headers, legends, filenames, summaries and exports, not only to a closing disclaimer.

## Workflow

1. **Scope.** Area or identifier, period in UTC (default: the last 12 months), intended use. Ask only for missing scope that changes the result.
2. **Retrieve** through one route, keeping the API quality screen or the map's default filters. A deliberate change needs a stated purpose and before/after counts, and its results stay separate from the default results. Never loosen filters just to get more results.
3. **Check completeness.** `numberMatched` must equal the features received. A failed or capped request makes the result partial, never zero.
4. **Inspect** consequential detections on the map: imagery, wind, candidate tracks.
5. **Deliver** data, method, interpretation and next steps separately. Read [methodology.md](references/methodology.md) before comparing areas, periods, vessels or companies.

## Reproducibility

Save with every result:

- the exact `curl` commands, or the final map URL;
- the UTC retrieval time and HTTP status;
- raw responses, unmodified, with derived tables in separate files;
- `numberMatched` and the number of features received;
- the scope: bbox or area ID, dates (UTC, inclusive), "intersects, not clipped", and any filter change.

Leave credentials, reviewer details and sensitive community locations out of anything you share.

Safe title: **"Five potential slick detections matching Cerulean's default filters, 11–20°E 36–43°N, 1–8 May 2025."** Unsafe: "Five confirmed illegal spills." A source table uses columns such as `potential_source`, `association_score`, `review_status` and `evidence_basis`, never `polluter`.
