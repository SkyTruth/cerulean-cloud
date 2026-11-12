# Cerulean API with curl

This guide covers queries, downloads and interpretation for anyone using the API. Automated agents must also follow the [investigation skill](../SKILL.md), including its agentic attribution and evidence rules.

`https://api.cerulean.skytruth.org` is a TiPG 1.0.1 (OGC API Features) service. Anonymous `GET` requests work, with no authentication and no published rate limit. If a request returns 401 or 403, stop and report it; do not look for keys.

Use `curl -sS --get URL --data-urlencode 'name=value' ...` so filters are encoded correctly, save each response with `-o`, and print the status with `-w 'HTTP %{http_code}\n'`. Responses are GeoJSON by default; add `--data-urlencode 'f=csv'` for a spreadsheet. Its geometry column is EWKT (`SRID=4326;MULTIPOLYGON(...)`); drop the `SRID=4326;` prefix for tools that expect WKT.

## Quick start

Run this in a terminal to download potential slick detections in a Mediterranean bounding box for 1–8 May 2025. The filter approximates the map's default [quality screen](#quality-screen); it does not confirm oil or a source.

```sh
API=https://api.cerulean.skytruth.org/collections
CERULEAN_UTM_MEDIUM='api-guide'
CERULEAN_UTM_SOURCE='unidentified-api-user'  # Replace with your organization or handle.
curl -sS --get "$API/public.slick_plus/items" \
  --data-urlencode "utm_medium=$CERULEAN_UTM_MEDIUM" \
  --data-urlencode "utm_source=$CERULEAN_UTM_SOURCE" \
  --data-urlencode 'bbox=11,36,20,43' \
  --data-urlencode 'datetime=2025-05-01T00:00:00Z/2025-05-08T23:59:59Z' \
  --data-urlencode 'filter=machine_confidence >= 0.7 AND max_source_collated_score >= 0' \
  --data-urlencode 'limit=10000' \
  -o slicks.geojson -w 'HTTP %{http_code}\n'
```

For human use, the examples use `utm_medium=api-guide`. **Agents must set `CERULEAN_UTM_MEDIUM='agentic'` and choose `CERULEAN_UTM_SOURCE` under the [skill's attribution rules](../SKILL.md#identify-agentic-use-and-the-user).** Keep both fields as separate URL parameters on every request, page and retry, never inside a CQL filter. They identify usage, not authentication or candidate pollution sources. Bare API URLs below are templates; add the parameters when using them.

Check the HTTP status before opening `slicks.geojson`. A download is complete only when the unique detections received across all pages equal `numberMatched`; see [query rules](#query-rules). On 2026-11-06 this example returned 5 detections, matching the map for the same area and period. Consult [interpretation guidance](methodology.md) before drawing conclusions from the results.

## Collections

`/collections` lists them. On 2026-10-26:

| Collection | Contents |
|---|---|
| `public.slick_plus` | One row per detection: `id`, `slick_timestamp` (UTC, no `Z`), `s1_scene_id`, `area` (m²), `length` (m), `hitl_cls` and `hitl_cls_name` (latest human review), `machine_confidence`, `slick_confidence` (geometric score), `max_source_collated_score`, `source_type_1_ids`, `source_type_2_ids` and `source_type_3_ids` (vessel MMSIs, structure IDs, dark-vessel IDs), `aoi_type_1_ids`, `aoi_type_2_ids` and `aoi_type_3_ids` (EEZ, IHO and MPA `aoi_id`s), `slick_url`, geometry. It omits detections that the model or the latest reviewer labeled not oil. Ignore its `cls` column. |
| `public.source_plus` | One row per detection–candidate pair: `slick_id`, `source_type`, `mmsi_or_structure_id`, `source_rank`, `source_collated_score`, `create_time` (when the match was computed, not the image time), `source_url`. Its `slick_confidence` is the machine score. It has no review field and no class filter, so its rows include associations a reviewer rejected and detections that `slick_plus` omits. |
| `public.cls`, `public.source_type`, `public.aoi_type` | Vocabularies, listed below. |
| `public.aoi_eez`, `public.aoi_iho`, `public.aoi_mpa` | Area IDs: `aoi_id` with Marine Regions `mrgid` (EEZ, IHO), `sovereigns` (EEZ) or WDPA `wdpaid` (MPA). No geometry. |
| `public.aoi` | Area outlines: `id` (the `aoi_id`), `type` (AOI type), `name` and full geometry. Fetch one with `$API/public.aoi/items/{aoi_id}`. |
| `public.sentinel1_grd` | Scene inventory with `scene_id`, acquisition `start_time`/`end_time`, `absolute_orbit_number` and footprint `geometry`; use for [coverage context](#coverage-context). A listed scene was not necessarily successfully processed throughout its footprint. |
| `public.get_slicks_by_*` functions | Avoid. They omit `slick_confidence`, fail without every argument, and took 36–39 s for one EEZ. |

- **Human review classes (`hitl_cls`):** 2 Anthropogenic, 3 Natural, 4 Infrastructure, 5 Vessel, 6 Vessel old, 7 Vessel recent and 8 Vessel coincident are human-classified apparent oil; 9 is Ambiguous; 1 Not Oil, 10 Land, 11 Sea Ice and 12 Artefact are not oil and are absent from `slick_plus`.
- **Source types:** 1 VESSEL (AIS), 2 INFRA, 3 DARK (vessel without AIS), 4 NATURAL.
- **AOI types:** 1 EEZ, 2 IHO, 3 MPA.

## Query rules

- **bbox:** `west,south,east,north` in degrees. It matches polygons that intersect the box, and polygons are not clipped. Split boxes that cross 180°.
- **datetime:** `start/end` in UTC with times, inclusive, applied to `slick_timestamp` on `slick_plus` and to acquisition `start_time` on `sentinel1_grd`. Dates without times return HTTP 500.
- **filter:** CQL2 text. `=`, `>=`, `IN`, `AND`, `OR`, `IS NULL` and `S_INTERSECTS(geometry, POLYGON((lon lat, ...)))` work. `IS NOT NULL` returns the null rows; use a comparison such as `>= -1000` instead. Array columns cannot be filtered: filter the `*_ids` arrays after download.
- **limit:** default 10, maximum 50,000. If `numberReturned` is below `numberMatched`, raise `limit` or page with `offset` and `sortby=id` until the unique IDs received equal `numberMatched`.
- **ids:** at most 900 IDs per `ids=`; 950 returned HTTP 500. For more, use `filter=id IN (...)`.
- **Smaller responses:** `properties=id,slick_timestamp,...` keeps only the named fields, and `bbox-only=true` replaces each geometry with its bounding box.
- **Scope every request** by time, bbox or IDs. Unfiltered requests took 20–60 s or timed out.
- **Errors:** retry 429, 502, 503 and 504 a few times with growing waits. Correct a query that returns 400, 422 or 500 instead of repeating it. An error page, malformed JSON or a missing count is never zero detections.

## Quality screen

`machine_confidence >= 0.7 AND max_source_collated_score >= 0` approximates the map's default filters: at least one candidate source with a collated score of 0 or more, plus the map's legacy 0.7 threshold. The API cannot see three map rules: the map counts only candidates ranked 1–3, admits reviewer-verified candidates at any rank or score, and, in vessel and structure searches, drops reviewer-rejected ones. For 11–20°E 36–43°N, 1–8 May 2025 both returned the same 5 detections; worldwide for those dates the map showed 110 and the API 109. Use the map when results must match it exactly.

Recent detections can wait days for source matching. For a requested review of recent data, run a separate query with `machine_confidence >= 0.7 AND max_source_collated_score IS NULL AND (slick_confidence >= 0.8 OR slick_confidence IS NULL)` (plus your bbox and dates), which applies the map's rules for detections without a candidate, and label its results provisional.

This screen defines the retrieval population; [possible origin estimation](origin.md) is a separate step. Its candidate threshold is strictly `source_collated_score > 0`, applied to each row, and it considers all available ranks and types. A screen match at score zero does not establish a possible origin. The default screen omits no-source detections: include the separate provisional population when the requested scope needs those detections, recording the filter change and counts.

## Recipes

Set `API`, `CERULEAN_UTM_MEDIUM` and `CERULEAN_UTM_SOURCE` as in the [quick start](#quick-start).

**Area and period.** Use the quick start above. For a polygon, add `AND S_INTERSECTS(geometry, POLYGON((14 36, 15 36, 15 37, 14 37, 14 36)))` to the filter, as closed rings of `lon lat` pairs. For 1–8 May 2025 this polygon returned 4111397.

**One detection.**

```sh
curl -sS --get --data-urlencode "utm_medium=$CERULEAN_UTM_MEDIUM" --data-urlencode "utm_source=$CERULEAN_UTM_SOURCE" "$API/public.slick_plus/items/4111397" -o slick-4111397.geojson
curl -sS --get --data-urlencode "utm_medium=$CERULEAN_UTM_MEDIUM" --data-urlencode "utm_source=$CERULEAN_UTM_SOURCE" "$API/public.source_plus/items" --data-urlencode 'slick_id=4111397' \
  --data-urlencode 'bbox-only=true' --data-urlencode 'limit=100' -o sources-4111397.geojson
```

Detection 4111397 has latest human class 8 (Vessel, coincident) and best collated score 0.368. Its geometric score of 1.0 plays no part, because a collated score exists. Its 8 candidates are vessels: rank 1 is MMSI 215371000 at 0.368, and the other 7 score below 0. Its `slick_url` is `https://cerulean.skytruth.org/slicks/4111397?ref=api&slick_id=4111397`.

**Possible origin or counts by source type.** Retrieve the detections for the area and acquisition period first, including `id`, `hitl_cls` and `hitl_cls_name`. The first request below is a worked lookup of two detections; for an area/period question, replace it with that population query. Then fetch positive-scoring candidates for batches of those IDs, without a source-type or rank filter:

```sh
curl -sS --get "$API/public.slick_plus/items" \
  --data-urlencode "utm_medium=$CERULEAN_UTM_MEDIUM" --data-urlencode "utm_source=$CERULEAN_UTM_SOURCE" \
  --data-urlencode 'ids=4111397,5318254' \
  --data-urlencode 'properties=id,slick_timestamp,hitl_cls,hitl_cls_name' \
  --data-urlencode 'bbox-only=true' --data-urlencode 'limit=10000' \
  --data-urlencode 'f=geojson' \
  -o origin-slicks.geojson -w 'HTTP %{http_code}\n'

curl -sS --get "$API/public.source_plus/items" \
  --data-urlencode "utm_medium=$CERULEAN_UTM_MEDIUM" --data-urlencode "utm_source=$CERULEAN_UTM_SOURCE" \
  --data-urlencode 'filter=slick_id IN (4111397, 5318254) AND source_collated_score > 0' \
  --data-urlencode 'properties=slick_id,source_id,source_type,mmsi_or_structure_id,source_rank,source_collated_score,create_time' \
  --data-urlencode 'sortby=slick_id,source_id' \
  --data-urlencode 'bbox-only=true' --data-urlencode 'limit=10000' \
  --data-urlencode 'f=geojson' \
  -o origin-candidates.geojson -w 'HTTP %{http_code}\n'
```

Replace the example IDs with the retrieved IDs. Page each batch using the same filter and sort order until the unique `(slick_id, source_id)` pairs received equal `numberMatched`; source responses have no queryable `id` column. Group locally by `slick_id` and follow the [local processing example](origin.md#runnable-example) for human precedence, origin labels and separate DARK-only/mixed-type counts. The period comes from the slick's acquisition time, not source `create_time`. `bbox-only=true` returns bounding-box geometry, rather than removing geometry. Batch query and offset pagination verified on 2026-11-12 UTC.

**Vessel or structure.**

```sh
curl -sS --get --data-urlencode "utm_medium=$CERULEAN_UTM_MEDIUM" --data-urlencode "utm_source=$CERULEAN_UTM_SOURCE" "$API/public.source_plus/items" --data-urlencode 'mmsi_or_structure_id=215371000' \
  --data-urlencode 'properties=slick_id,source_type,source_rank,source_collated_score' \
  --data-urlencode 'bbox-only=true' --data-urlencode 'limit=10000' -o vessel-215371000.geojson
```

Keep rows with `source_rank <= 3` and `source_collated_score >= 0`. Then fetch those detections with `ids=`, `filter=machine_confidence >= 0.7` and `properties=id,slick_timestamp,hitl_cls_name` to get their image dates and to drop IDs that are missing from `slick_plus` or below the map's 0.7 threshold.

- For MMSI 215371000 this gave 32 rows and 5 screened detections, 2 of them since 2024-01-01: 3517386 and 4111397.
- The map's vessel view shows only 4111397 for that period, which is consistent with a reviewer rejecting the 3517386 association. The API cannot show rejections.
- Report screened rows as candidate associations, and check the vessel on the map before relying on any of them.

**EEZ, IHO sea area or MPA.**

```sh
curl -sS --get --data-urlencode "utm_medium=$CERULEAN_UTM_MEDIUM" --data-urlencode "utm_source=$CERULEAN_UTM_SOURCE" "$API/public.aoi_eez/items" --data-urlencode 'mrgid=5682' -o aoi.geojson   # Italy: aoi_id 224
```

Look up an IHO sea area in `public.aoi_iho` by `mrgid`, or an MPA in `public.aoi_mpa` by `wdpaid`. Query `slick_plus` for the period with the screen and `properties=id,aoi_type_N_ids`, where N is 1 for an EEZ, 2 for an IHO sea area or 3 for an MPA, adding a rough bbox around the area if you know one, to shrink the download. Keep the features whose `aoi_type_N_ids` contains the `aoi_id`. Italy's EEZ, 1–8 May 2025: 6 detections, the same 6 the map shows with that EEZ selected. Membership is Cerulean's precomputed intersection.

## Coverage context

Use `curl` against `$API/public.sentinel1_grd/items` with the same bbox (for a named area, the bounding box of its `public.aoi` outline) and UTC `datetime` interval, `datetime-column=start_time` (the collection's default, kept explicit), `f=geojson`, `sortby=id`, `limit=100`, and both UTM fields. This selects by acquisition start, not ingestion time. Fetch all pages, deduplicate `scene_id`, and retain `start_time`, `end_time`, `absolute_orbit_number` and full geometry. Include scenes with no detected slicks; apply no slick quality filters.

Intersect footprints with the AOI outline (the bbox, your polygon, or the `public.aoi` geometry), union the intersections, and divide their geodesic or equal-area projected area by AOI area, for example with shapely and pyproj: the **fraction intersected at least once**. Do not use `bbox-only=true` or sum overlapping scene areas. Without a geometry library, report the scenes, acquisition dates and gaps, and state that the fraction was not computed. Report acquisition dates and gaps, including the period's ends; inspect intersections by date when timing matters. Overlapping slices from one pass are not separate visits.

Label this **Cerulean inventory footprint coverage**, subject to the [processing and interpretation limits](methodology.md#coverage-context). Query syntax and full-geometry return verified on 2026-11-06.

## After retrieval

Count and list IDs locally, for example:

```sh
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["numberMatched"], d["numberReturned"], sorted(f["id"] for f in d["features"]))' slicks.geojson
```

Inspect consequential detections on the map, and cite each detection's `slick_url` after merging the two UTM fields; preserve the original value in the raw response.
