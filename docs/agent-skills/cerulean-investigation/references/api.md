# Cerulean API with curl

`https://api.cerulean.skytruth.org` is a TiPG 1.0.1 (OGC API Features) service. Anonymous `GET` requests work, with no authentication and no published rate limit. If a request returns 401 or 403, stop and report it; do not look for keys.

Use `curl -sS --get URL --data-urlencode 'name=value' ...` so filters are encoded correctly, save each response with `-o`, and print the status with `-w 'HTTP %{http_code}\n'`. Responses are GeoJSON by default; add `--data-urlencode 'f=csv'` for a spreadsheet with WKT geometry.

## Collections

`/collections` lists them. On 2026-10-26:

| Collection | Contents |
|---|---|
| `public.slick_plus` | One row per detection: `id`, `slick_timestamp` (UTC, no `Z`), `s1_scene_id`, `area` (m²), `length` (m), `hitl_cls` and `hitl_cls_name` (latest human review), `machine_confidence`, `slick_confidence` (geometric score), `max_source_collated_score`, `source_type_1_ids`, `source_type_2_ids` and `source_type_3_ids` (vessel MMSIs, structure IDs, dark-vessel IDs), `aoi_type_1_ids`, `aoi_type_2_ids` and `aoi_type_3_ids` (EEZ, IHO and MPA `aoi_id`s), `slick_url`, geometry. It omits detections that the model or the latest reviewer labeled not oil. Ignore its `cls` column. |
| `public.source_plus` | One row per detection–candidate pair: `slick_id`, `source_type`, `mmsi_or_structure_id`, `source_rank`, `source_collated_score`, `create_time` (when the match was computed, not the image time), `source_url`. Its `slick_confidence` is the machine score. It has no review field and no class filter, so its rows include associations a reviewer rejected and detections that `slick_plus` omits. |
| `public.cls`, `public.source_type`, `public.aoi_type` | Vocabularies, listed below. |
| `public.aoi_eez`, `public.aoi_iho`, `public.aoi_mpa` | Area IDs: `aoi_id` with Marine Regions `mrgid` (EEZ, IHO), `sovereigns` (EEZ) or WDPA `wdpaid` (MPA). No geometry. |
| `public.sentinel1_grd` | Sentinel-1 scene inventory. A listed scene was not necessarily processed. |
| `public.get_slicks_by_*` functions | Avoid. They omit `slick_confidence`, fail without every argument, and took 36–39 s for one EEZ. |

- **Human review classes (`hitl_cls`):** 2 Anthropogenic, 3 Natural, 4 Infrastructure, 5 Vessel, 6 Vessel old, 7 Vessel recent and 8 Vessel coincident are human-classified apparent oil; 9 is Ambiguous; 1 Not Oil, 10 Land, 11 Sea Ice and 12 Artefact are not oil and are absent from `slick_plus`.
- **Source types:** 1 VESSEL (AIS), 2 INFRA, 3 DARK (vessel without AIS), 4 NATURAL.
- **AOI types:** 1 EEZ, 2 IHO, 3 MPA.

## Query rules

- **bbox:** `west,south,east,north` in degrees. It matches polygons that intersect the box, and polygons are not clipped. Split boxes that cross 180°.
- **datetime:** `start/end` in UTC with times, inclusive, applied to `slick_timestamp`. Dates without times return HTTP 500.
- **filter:** CQL2 text. `=`, `>=`, `IN`, `AND`, `OR`, `IS NULL` and `S_INTERSECTS(geometry, POLYGON((lon lat, ...)))` work. `IS NOT NULL` returns the null rows; use a comparison such as `>= -1000` instead. Array columns cannot be filtered: filter the `*_ids` arrays after download.
- **limit:** default 10, maximum 50,000. Compare `numberMatched` with `numberReturned`; if they differ, raise `limit` or page with `offset` and `sortby=id`.
- **ids:** at most 900 IDs per `ids=`; 950 returned HTTP 500. For more, use `filter=id IN (...)`.
- **Smaller responses:** `properties=id,slick_timestamp,...` keeps only the named fields, and `bbox-only=true` replaces each geometry with its bounding box.
- **Scope every request** by time, bbox or IDs. Unfiltered requests took 20–60 s or timed out.
- **Errors:** retry 429, 502, 503 and 504 a few times with growing waits. Correct a query that returns 400, 422 or 500 instead of repeating it. An error page, malformed JSON or a missing count is never zero detections.

## Quality screen

`machine_confidence >= 0.7 AND max_source_collated_score >= 0` approximates the map's default filters: at least one candidate source with a collated score of 0 or more, plus the map's legacy 0.7 threshold. The API cannot see three map rules: the map counts only candidates ranked 1–3, admits reviewer-verified candidates at any rank or score, and drops reviewer-rejected ones. For 11–20°E 36–43°N, 1–8 May 2025 both returned the same 5 detections; worldwide for those dates the map showed 110 and the API 109. Use the map when results must match it exactly.

Recent detections can wait days for source matching. For a requested review of recent data, run a separate query with `max_source_collated_score IS NULL AND slick_confidence >= 0.8` (plus your bbox and dates) and label its results provisional.

## Recipes

Set `API=https://api.cerulean.skytruth.org/collections`.

**Area and period.** Use the quick start in SKILL.md. For a polygon, add `AND S_INTERSECTS(geometry, POLYGON((14 36, 15 36, 15 37, 14 37, 14 36)))` to the filter, as closed rings of `lon lat` pairs. For 1–8 May 2025 this polygon returned 4111397.

**One detection.**

```sh
curl -sS "$API/public.slick_plus/items/4111397" -o slick-4111397.geojson
curl -sS --get "$API/public.source_plus/items" --data-urlencode 'slick_id=4111397' \
  --data-urlencode 'bbox-only=true' --data-urlencode 'limit=100' -o sources-4111397.geojson
```

Detection 4111397 has latest human class 8 (Vessel, coincident) and best collated score 0.368. Its geometric score of 1.0 plays no part, because a collated score exists. Its 8 candidates are vessels: rank 1 is MMSI 215371000 at 0.368, and the other 7 score below 0. Its map page is `https://cerulean.skytruth.org/slicks/4111397`.

**Vessel or structure.**

```sh
curl -sS --get "$API/public.source_plus/items" --data-urlencode 'mmsi_or_structure_id=215371000' \
  --data-urlencode 'properties=slick_id,source_type,source_rank,source_collated_score' \
  --data-urlencode 'bbox-only=true' --data-urlencode 'limit=10000' -o vessel-215371000.geojson
```

Keep rows with `source_rank <= 3` and `source_collated_score >= 0`. Then fetch those detections with `ids=` and `properties=id,slick_timestamp,hitl_cls_name` to get their image dates and to drop IDs missing from `slick_plus`.

- For MMSI 215371000 this gave 32 rows and 5 screened detections, 2 of them since 2024-01-01: 3517386 and 4111397.
- The map's vessel view shows only 4111397 for that period, which is consistent with a reviewer rejecting the 3517386 association. The API cannot show rejections.
- Report screened rows as candidate associations, and check the vessel on the map before relying on any of them.

**EEZ, IHO sea area or MPA.**

```sh
curl -sS --get "$API/public.aoi_eez/items" --data-urlencode 'mrgid=5682' -o aoi.geojson   # Italy: aoi_id 224
```

Query `slick_plus` for the period with the screen and `properties=id,aoi_type_1_ids`, adding a rough bbox around the area if you know one, to shrink the download. Keep the features whose `aoi_type_1_ids` (EEZ), `aoi_type_2_ids` (IHO) or `aoi_type_3_ids` (MPA) contains the `aoi_id`. Italy's EEZ, 1–8 May 2025: 6 detections, the same 6 the map shows with that EEZ selected. Membership is Cerulean's precomputed intersection.

## After retrieval

Count and list IDs locally, for example:

```sh
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["numberMatched"], d["numberReturned"], sorted(f["id"] for f in d["features"]))' slicks.geojson
```

Inspect consequential detections on the map, and cite each detection's `slick_url`.
