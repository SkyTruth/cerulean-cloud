# Possible origins and counts by source type

Use this guide both to supply an origin field for a detection and to interpret requests such as "How many dark vessel slicks were in this area last year?" These are derived estimates, not extra API fields or confirmed sources. Keep the [evidence rules](../SKILL.md#evidence-rules) and the retrieval quality screen.

## Estimate a detection's possible origin

### 1. Apply the latest human classification first

Use `slick_plus.hitl_cls` and `hitl_cls_name`, never the machine class `cls`.

| Latest human class | Possible origin |
|---|---|
| Vessel, including old, recent and coincident (`hitl_cls` 5–8) | `Vessel` |
| Infrastructure (4) | `Infrastructure` |
| Natural (3) | `Natural` |
| Not Oil or a not-oil subclass (1, 10–12 in the current vocabulary) | Exclude from oil-detection displays and counts |
| No review, Anthropogenic (2), or Ambiguous (9) | Proceed to candidate-based estimation |

Retain the review class and its meaning when estimating from candidates: a source estimate does not turn an ambiguous human oil classification into human-classified apparent oil. A human Vessel classification alone cannot establish whether the source is `DARK` or AIS-associated `VESSEL`.

### 2. Build the set of qualifying source types

Retrieve **all available** `source_plus` rows for the slick whose own `source_collated_score > 0`, then build the distinct set `S` of their `source_type` values. Use all ranks and types; do not restrict this set to the map's top three, the highest-scoring candidate, a particular source ID or the requested type. Strictly positive excludes zero, negative and null scores.

Neither `max_source_collated_score` nor the `source_type_*_ids` arrays tell you which types have positive scores. Neither `machine_confidence` nor either collection's `slick_confidence` determines origin.

Check completeness before assigning a set. A failed, capped or incomplete candidate request cannot establish a sole type or an empty set. Deduplicate candidate rows by `(slick_id, source_id)` and retain raw types. `source_plus` exposes active associations, but does not expose source verification/rejection fields and can include reviewer-rejected associations. Disclose that limit for API-only estimates; inspect consequential associations on the map.

### 3. Assign the automated label

Use **Possible origin** as the field heading. Keep human review and source-analysis status alongside the label.

| Candidate evidence, after the human override above | Label |
|---|---|
| Only `INFRA` has positive-scoring candidates, and AIS/dark analysis is known to be pending | `Infrastructure (Pending)` |
| Otherwise, one distinct type: `VESSEL`, `INFRA`, `DARK` or `NATURAL` | `Vessel`, `Infrastructure`, `Dark` or `Natural`, respectively |
| More than one distinct type | `Uncertain` |
| No positive-scoring candidates, and source analysis is known to be pending | `Pending` |
| No positive-scoring candidates after analysis has run, or processing status is unavailable | `Unknown` |

An infrastructure-only positive match before AIS/dark analysis gives `Infrastructure (Pending)`. Once that analysis has run, use `Infrastructure` if infrastructure remains the only positive-scoring type. With no positive match and analysis known to be pending, use `Pending`. Any candidate-based label can change as later analysis adds or changes candidates. A lower-scoring positive candidate of a different type is enough to change a sole-type estimate to `Uncertain`.

`Unreviewed` describes human review, not source-analysis progress. The public collections do not expose a reliable per-analyzer completion flag. Do not infer completion or pending status from age, missing candidates, or a 72-hour deadline. If status is unavailable, say so alongside the estimate; a known pending stage makes an existing type estimate provisional.

Keep `DARK` and `VESSEL` distinct for dark-vessel questions. If the user explicitly wants a combined physical `Vessel` category, merge them **before** counting distinct types, preserve the raw set and disclose the grouping. Never infer DARK from a missing MMSI, an ID prefix alone or a human Vessel classification; use `source_type = DARK`.

## Answer questions in the other direction

For a requested source type `T`, retrieve the area/period population from `slick_plus`, then retrieve all positive-scoring candidate types for those slick IDs using the [API recipe](api.md#recipes). Use the slick's acquisition timestamp for the period. Filtering the candidate request to `source_type = T` would hide competitors and falsely make every result look like sole-type support.

Apply human precedence, and distinguish these groups:

- **Human-classified origin:** a specific latest human class determines the origin. Report its basis separately from automated estimates. For DARK, there is no corresponding human slick class; a reviewed Vessel can have DARK candidate support, but Vessel review alone is insufficient. Describe that support as a subgroup of reviewed vessels when present.
- **Sole-type support:** no specific human origin overrides the estimate, and `S = {T}`. Several positive candidates of type `T` still give one type. Describe this as an automated possible origin supported by a single type.
- **Mixed-type support:** no specific human origin overrides the estimate, `T` is in `S`, and `S` contains another type. The possible origin is `Uncertain`; report these detections separately from sole-type estimates.

If candidate types conflict with a specific human origin, preserve the human origin. Describe the conflicting candidates separately if relevant; do not count them as an automated origin of the requested type. Exclude not-oil reviews throughout. Human origin, sole-type and mixed-type groups must not double-count the same detection in a combined total.

For "dark vessel slicks", report the number of unique detections with DARK-only candidate support and the additional number with DARK among mixed types. Give any reviewed-vessel subgroup separately, splitting its DARK support into sole-type and mixed-type support too; its displayed human origin remains `Vessel`. Include remaining `Unknown`/`Pending` detections when relevant to the requested population, and disclose when processing status could not be established. A detection with two DARK candidates counts once, and a mixed detection counts once in the DARK mixed group even if it has several DARK candidates. Mixed-type groups for different requested types can overlap; do not sum them as unique detections.

Safe answer pattern (illustrative counts): "12 potential slick detections had DARK as their only positive-scoring candidate type; 7 additional detections had DARK plus another positive-scoring type, so their origins remain uncertain." Explain that sole-type support is more consistent among the available candidates, not a measured probability or confirmation. Do not label either group "confirmed dark-vessel spills" or imply that missing AIS demonstrates concealment.

### Runnable example

Run the two `curl` requests in the [API recipe](api.md#recipes), with `API` and `CERULEAN_UTM_SOURCE` set as in the quick start, and check both HTTP statuses. Then run the code below in the same directory. It uses only Python's standard library and saved responses; it makes no network requests. It writes derived labels and auditable DARK support groups to `origin-estimates.json`, leaving both raw files unchanged.

This small-batch example requires a complete response in each input file and stops if a response is partial. For a larger area/period population, retrieve and reconcile every page and ID batch as described in the API guide before applying the same grouping logic. An API-only example cannot determine processing status, so it records status as unavailable. An integration can populate `analysis_status_by_slick` from an explicit processing signal: `pending` while AIS/dark analysis is pending, or `run` once it has run. A known pending status produces `Infrastructure (Pending)` for infrastructure-only support, or `Pending` for no positive support.

```sh
python3 - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

def complete_rows(filename, key):
    data = json.loads(Path(filename).read_text())
    rows = [feature["properties"] for feature in data["features"]]
    if len(rows) != data["numberMatched"] or len({key(row) for row in rows}) != len(rows):
        raise SystemExit(f"Incomplete or duplicate response: {filename}")
    return rows

slicks = complete_rows("origin-slicks.geojson", lambda row: row["id"])
sources = complete_rows("origin-candidates.geojson", lambda row: (row["slick_id"], row["source_id"]))
slick_ids = {row["id"] for row in slicks}
if any(row["slick_id"] not in slick_ids for row in sources):
    raise SystemExit("Candidate response includes slicks outside the retrieved population")

types = defaultdict(set)
for row in sources:
    score = row["source_collated_score"]
    if score is not None and score > 0:
        types[row["slick_id"]].add(row["source_type"])

human_origins = {3: "Natural", 4: "Infrastructure", **dict.fromkeys((5, 6, 7, 8), "Vessel")}
type_labels = {"VESSEL": "Vessel", "INFRA": "Infrastructure", "DARK": "Dark", "NATURAL": "Natural"}
groups = {name: [] for name in (
    "dark_only", "dark_mixed", "reviewed_vessel_dark_only",
    "reviewed_vessel_dark_mixed", "conflicting_human_origin",
)}
origins = []
analysis_status_by_slick = {}  # Populate only from an explicit processing signal.
for slick in slicks:
    review = slick["hitl_cls"]
    if review in (1, 10, 11, 12):  # Current not-oil classes; excluded throughout.
        continue
    source_types = types[slick["id"]]
    status = analysis_status_by_slick.get(slick["id"], "unavailable")
    human = human_origins.get(review)
    if human:
        origin, basis = human, "human_review"
    elif source_types == {"INFRA"} and status == "pending":
        origin, basis = "Infrastructure (Pending)", "source_candidates"
    elif len(source_types) == 1:
        origin, basis = type_labels[next(iter(source_types))], "source_candidates"
    else:
        origin = "Uncertain" if source_types else ("Pending" if status == "pending" else "Unknown")
        basis = "source_candidates" if source_types else "no_support"
    origins.append({
        "slick_id": slick["id"], "possible_origin": origin, "basis": basis,
        "hitl_cls": review, "hitl_cls_name": slick["hitl_cls_name"],
        "positive_source_types": sorted(source_types), "source_analysis_status": status,
    })
    if "DARK" in source_types:
        group = "dark_only" if source_types == {"DARK"} else "dark_mixed"
        if human == "Vessel":
            group = "reviewed_vessel_" + group
        elif human:
            group = "conflicting_human_origin"
        groups[group].append(slick["id"])

counts = {name: len(ids) for name, ids in groups.items()}
result = {"origins": origins, "dark_support_counts": counts, "dark_support_slick_ids": groups}
Path("origin-estimates.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(counts, indent=2))
PY
```

`dark_only` and `dark_mixed` contain detections without a specific human origin override. Reviewed-vessel support and conflicting human origins stay in separate groups, so they cannot silently inflate an automated DARK-origin count. The illustrative IDs in the API recipe need not have DARK support; zero DARK counts are valid only after complete retrieval. Change the area/period population to answer the user's actual question.

### Worked cases

The examples below are hypothetical; all candidate-based rows have no specific human origin override, and candidate retrieval is complete.

| Human review / processing context | Candidate types and collated scores | Possible origin | DARK question |
|---|---|---|---|
| No specific origin | DARK 0.6; DARK 0.2 | `Dark` | One detection with sole-type support |
| No specific origin | DARK 0.6; VESSEL 0.1 | `Uncertain` | One detection with mixed-type support |
| No specific origin | DARK 0.6; INFRA 0 | `Dark` | One detection with sole-type support; zero does not qualify |
| No specific origin | DARK −0.2; INFRA 0.4 | `Infrastructure` | Excluded from DARK support groups |
| Human Infrastructure | DARK 0.8 | `Infrastructure` | Human override; conflicting candidate kept separate |
| Human Vessel | DARK 0.6; DARK 0.2 | `Vessel` | Reviewed-vessel subgroup with sole-type DARK support |
| Human Vessel | DARK 0.6; VESSEL 0.1 | `Vessel` | Reviewed-vessel subgroup with mixed-type DARK support |
| Human Vessel | No positive candidates | `Vessel` | No DARK evidence |
| Human Not Oil | DARK 0.8 | Excluded | Excluded |
| AIS/dark analysis known pending | INFRA 0.4 | `Infrastructure (Pending)` | Provisional infrastructure estimate |
| Source analysis known pending | No positive candidates | `Pending` | Excluded from DARK support groups |
| Source analysis has run | No positive candidates | `Unknown` | Excluded from DARK support groups |
| Processing status unavailable | No positive candidates | `Unknown` | Processing status unavailable |

## Refresh and reproduce

Store the label, its human or automated basis, latest human class, positive source-type set, candidate IDs/scores, processing status if known, and retrieval time with derived results. Retain the scope, filters, raw responses and completeness counts required by the main skill.

Recompute when human review or candidate associations change. When refreshing a slick, reconcile the current full candidate set so removed or deactivated associations do not remain in the estimate. `source_plus.create_time` is association creation time, not a reliable last-modified watermark: existing scores, ranks and active status can change without advancing it, and human reviews can change without a new source row. A source-creation-only ingest cannot guarantee current origin labels; disclose its freshness limits or refresh affected detections through an available change signal or periodic reconciliation.
