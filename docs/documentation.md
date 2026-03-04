# Cerulean Progress Tracker REST API Documentation

## TL;DR
Cerulean exposes an **OGC API Features**-compatible REST API at `https://api.cerulean.skytruth.org/` for programmatic access to **oil slick detections** and **potential source attribution** data (vessels + offshore infrastructure + dark vessels), plus supporting tables (AOIs, classes). Most interactions are **`GET /collections/{collectionId}/items`** with OGC-style query parameters like **`bbox`**, **`datetime`**, **`limit`/`offset`**, **`sortby`**, and **CQL-2 `filter`**. Responses are typically **GeoJSON FeatureCollections**.

---

## Overview

For most users, we recommend using the **Cerulean web application**:  
https://cerulean.skytruth.org/

For users who want to directly access and download oil slick detection data, Cerulean provides programmatic free access to an **OGC compliant API**:  
https://api.cerulean.skytruth.org/

Currently, oil slick detections can be downloaded in addition to data used for potential source identification of vessels and offshore oil platform locations (**excluding AIS tracks, which are only accessible via the UI**). API queries can be made programmatically (e.g. `curl` / `requests`) for direct data access and download. You can also execute API queries within a browser by pasting an API URL into the address bar to view results (including a paginated map) or download data directly.

Cerulean’s API is powered by:
- **tipg** (OGC API Features server): https://developmentseed.org/tipg/
- **CQL-2** filtering: https://cran.r-project.org/web/packages/rstac/vignettes/rstac-02-cql2.html

---

## ⚠️ Upcoming Breaking Changes ⚠️

No breaking changes are announced in this document. If your integration is long-lived, you should:
- pin exact query semantics in tests (e.g., response schema expectations, field availability),
- and track the “standard API docs” for updates: https://api.cerulean.skytruth.org/

---

## Base URL

```bash
https://api.cerulean.skytruth.org/
```

The most common pattern is:

```text
GET /collections/{collectionId}/items
```

Example (slick detections):

```text
GET /collections/public.slick_plus/items
```

---

## Structure

Cerulean follows the **OGC API Features** model:

- **Collections**: named datasets exposed under `/collections`
- **Items**: features/rows within a collection exposed under `/collections/{collectionId}/items`

Common query concepts:

- **Spatial filter**: `bbox=minLon,minLat,maxLon,maxLat`
- **Temporal filter**: `datetime=start/end` (UTC; format `YYYY-MM-DDTHH:MM:SSZ`)
- **Pagination**:
  - `limit` (default 10; max 9999)
  - `offset` (start row)
- **Property filtering**:
  - `filter=...` using **CQL-2**
  - some collections also accept direct equality filters as query params, e.g. `?id=...`, `?s1_scene_id=...`, `?mrgid=...`
- **Sorting**: `sortby=field` or `sortby=-field` (descending)
- **Output format**: `f=geojson` (default) or `f=csv`

---

## Citation

To cite Cerulean API output:

> SkyTruth Cerulean API. (n.d.). Query: [Brief description of your query]. Retrieved [Month DD, YYYY] from https://api.cerulean.skytruth.org/

If you use the UI as well, include `slick_url` / `source_url` links when relevant.

---

## Requests

### Headers
- `Content-Type: application/json;` (safe default)

### Common Query Parameters

| Parameter | Type | Meaning |
|---|---|---|
| `bbox` | `minLon,minLat,maxLon,maxLat` | Spatial bounding box |
| `datetime` | `start/end` | ISO-8601 UTC range (inclusive semantics depend on server) |
| `limit` | int | Max items returned (default 10; max 9999) |
| `offset` | int | Skip N items (pagination) |
| `filter` | string | CQL-2 filter expression |
| `sortby` | string | Sort key; prefix `-` for descending |
| `f` | string | Output format (e.g., `geojson`, `csv`) |

**Note on spaces in `filter`:** CQL-2 filters include spaces and operators like `GTE`, `GT`, `EQ`, `IS NULL`. If you are using `curl` or a browser URL, ensure the query is URL-encoded where needed.

---

## Responses

### Success (typical)

Most endpoints return a **GeoJSON FeatureCollection**. In addition to `features`, responses commonly include OGC pagination metadata fields such as:

- `numberMatched`: total features matching your filters (independent of `limit`)
- `numberReturned`: number of features returned in this page
- `features`: list of GeoJSON Features

Example (shape only):

```json
{
  "type": "FeatureCollection",
  "numberMatched": 1234,
  "numberReturned": 100,
  "features": [
    {
      "type": "Feature",
      "id": 3582918,
      "geometry": { "...": "..." },
      "properties": { "...": "..." }
    }
  ]
}
```

### Errors

Errors are returned as JSON with an HTTP 4xx/5xx status. The exact shape may vary by endpoint/server version; treat it as an opaque payload and rely on HTTP status + message.

---

## Resources

### Queryable Resources Table of Contents

Primary collections you’ll likely query:

- [Slick Detections (`public.slick_plus`)](#slick-detections-publicslick_plus)
- [Slick-Source Attribution (`public.source_plus`)](#slick-source-attribution-publicsource_plus)
- [Areas of Interest (`public.aoi`)](#areas-of-interest-publicaoi)
- [EEZ AOIs (`public.aoi_eez`)](#eez-aois-publicaoi_eez)
- [IHO AOIs (`public.aoi_iho`)](#iho-aois-publicaoi_iho)
- [MPA AOIs (`public.aoi_mpa`)](#mpa-aois-publicaoi_mpa)
- [Human Review Classes (`public.cls`)](#human-review-classes-publiccls)
- [Slicks by AOI (`public.get_slicks_by_aoi`)](#slicks-by-aoi-publicget_slicks_by_aoi)

---

## Slick Detections (`public.slick_plus`)

### Description
Oil slick detection geometries and associated metadata, including model confidence and optional human-in-the-loop review labels, plus links back to the Cerulean UI.

### Endpoints
- `GET /collections/public.slick_plus/items`

> Some lookups are shown as query-parameter equality filters (e.g. `?id=...`, `?s1_scene_id=...`, `?hitl_cls=...`) in examples below.

### Fields (key subset)
<details>
  <summary>Slick detection fields (as surfaced in examples)</summary>

| Field | Type | Notes |
|---|---|---|
| `geometry` | MultiPolygon | Slick footprint |
| `id` | int | Unique slick id |
| `slick_timestamp` | datetime | When slick was detected (UTC) |
| `s1_scene_id` | string | Sentinel-1 scene identifier |
| `area` | number | Area estimate (m²) |
| `length` | number | Length estimate (m) |
| `perimeter` | number | Perimeter estimate (m) |
| `cls` | int | Model class (not ground truth) |
| `machine_confidence` | number | Model confidence score |
| `hitl_cls` | int | Human-reviewed class id (nullable) |
| `hitl_cls_name` | string | Human-reviewed label (nullable) |
| `slick_url` | string | Link to open slick in UI |
| `aoi_type_1_ids` | array | Intersecting EEZ AOI ids |
| `aoi_type_2_ids` | array | Intersecting IHO AOI ids |
| `aoi_type_3_ids` | array | Intersecting MPA AOI ids |
| `source_type_1_ids` | array | Nearby vessels (nullable) |
| `source_type_2_ids` | array | Nearby infrastructure (nullable) |
| `source_type_3_ids` | array | Nearby dark vessels (nullable) |
| `max_source_collated_score` | number | Highest slick-source match score |

</details>

### Notes on interpretation
- Not all geometries are true oil detections. Validate using Sentinel-1 imagery; the UI provides convenient verification.
- `max_source_collated_score` is a proxy for slick-to-source attribution confidence (not ground truth). A common heuristic is to prefer `> 0`.

### Example Requests

#### Example 1: Return slicks within a bounding box

**cURL**
```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.slick_plus/items?limit=100&bbox=10.9,42.3,19.7,36.1' \
  -H 'Content-Type: application/json;'
```

**Python**
```python
import requests

url = (
  "https://api.cerulean.skytruth.org/collections/public.slick_plus/items"
  "?limit=100"
  "&bbox=10.9,42.3,19.7,36.1"
)

resp = requests.get(url, headers={"Content-Type": "application/json;"})
resp.raise_for_status()
data = resp.json()

print("Total Matched:", data.get("numberMatched"))
print("Total Returned:", data.get("numberReturned"))
```

#### Example 2: Filter by datetime + high-grade results

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.slick_plus/items?limit=100&bbox=10.9,42.3,19.7,36.1&datetime=2024-01-01T00:00:00Z/2025-01-01T00:00:00Z&filter=max_source_collated_score%20%3E%200&sortby=-max_source_collated_score' \
  -H 'Content-Type: application/json;'
```

#### Example 3: Other basic filtering (CQL-2 operators)

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.slick_plus/items?limit=100&bbox=10.9,42.3,19.7,36.1&datetime=2024-01-01T00:00:00Z/2025-01-01T00:00:00Z&sortby=slick_timestamp&filter=max_source_collated_score%20GTE%200.0%20AND%20area%20GT%205000000' \
  -H 'Content-Type: application/json;'
```

#### Example 4: Filter by source presence (vessel nearby)

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.slick_plus/items?limit=100&bbox=10.9,42.3,19.7,36.1&datetime=2024-01-01T00:00:00Z/2025-01-01T00:00:00Z&sortby=slick_timestamp&filter=max_source_collated_score%20GTE%200.0%20AND%20area%20GT%205000000%20AND%20NOT%20source_type_1_ids%20IS%20NULL' \
  -H 'Content-Type: application/json;'
```

#### Example 5: Download data (GeoJSON default; CSV via `f=csv`)

GeoJSON:
```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.slick_plus/items?limit=100&bbox=10.9,42.3,19.7,36.1&sortby=slick_timestamp&datetime=2024-01-01T00:00:00Z/2025-01-01T00:00:00Z&filter=max_source_collated_score%20GTE%200.0%20AND%20area%20GT%205000000%20AND%20NOT%20source_type_1_ids%20IS%20NULL' \
  -o example_5.geojson
```

CSV:
```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.slick_plus/items?limit=100&bbox=10.9,42.3,19.7,36.1&sortby=slick_timestamp&datetime=2024-01-01T00:00:00Z/2025-01-01T00:00:00Z&filter=max_source_collated_score%20GTE%200.0%20AND%20area%20GT%205000000%20AND%20NOT%20source_type_1_ids%20IS%20NULL&f=csv' \
  -o example_5.csv
```

#### Example 6: Return a specific slick by its ID

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.slick_plus/items?id=3582918' \
  -H 'Content-Type: application/json;'
```

#### Example 7: Return all slicks detected in a specific Sentinel-1 scene

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.slick_plus/items?s1_scene_id=S1A_IW_GRDH_1SDV_20240113T165611_20240113T165636_052091_064BBD_918F' \
  -H 'Content-Type: application/json;'
```

#### Example 11: Obtaining human reviewed slicks (`hitl_cls`)

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.slick_plus/items?bbox=10.9,42.3,19.7,36.1&limit=10&hitl_cls=8' \
  -H 'Content-Type: application/json;'
```

---

## Slick-Source Attribution (`public.source_plus`)

### Description
Potential polluters / sources associated with slicks, including source type (VESSEL / INFRA / DARK / NATURAL), relative attribution scores, and links back to the UI.

### Endpoints
- `GET /collections/public.source_plus/items`

### Fields (key subset)
<details>
  <summary>Source attribution fields (as surfaced in examples)</summary>

| Field | Type | Notes |
|---|---|---|
| `geometry` | geometry | Spatial feature associated with the record |
| `id` | int | Unique record id |
| `mmsi_or_structure_id` | int | Vessel MMSI or infrastructure structure_id |
| `source_type` | string | e.g. `VESSEL`, `INFRA`, `DARK`, `NATURAL` |
| `source_collated_score` | number | Typically in `[-5, +5]`; heuristic: prefer `> 0` |
| `source_rank` | int | 1 = best candidate, 2 = next, ... |
| `slick_confidence` | number | Detection confidence proxy |
| `git_tag` | string | Version tag (compatibility across scoring versions) |
| `slick_url` | string | Link to slick in UI |
| `source_url` | string | Link to source in UI |

</details>

### Example Requests

#### Example 9: Iterate through sources (by MMSI / structure_id)

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.source_plus/items?mmsi_or_structure_id=372519000&filter=source_collated_score%20GT%200.0%20AND%20source_rank%20EQ%201' \
  -H 'Content-Type: application/json;'
```

#### Example 10: Locate dark vessel polluters in a bbox + datetime window

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.source_plus/items?bbox=10.9,42.3,19.7,36.1&datetime=2025-05-01T00:00:00Z/2025-05-09T00:00:00Z&source_type=DARK&filter=source_collated_score%20GT%200.5&sortby=-source_collated_score' \
  -H 'Content-Type: application/json;'
```

### Notes on `git_tag`
The higher the `git_tag`, the more robust results can be expected to be. A practical starting point from the examples is `git_tag >= 1.1.0`, based on breaking changes introduced after `1.0.11`.

---

## Areas of Interest (`public.aoi`)

### Description
Canonical AOI registry. Each AOI has a unique id and a `type` which indicates whether it is an EEZ, IHO Sea Area, or MPA.

### Endpoints
- `GET /collections/public.aoi/items`

### Example: search for an AOI by name (case-insensitive match)

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.aoi/items?filter=LOWER(name)%20LIKE%20%27%25greek%25%27' \
  -H 'Content-Type: application/json;'
```

The AOI `id` returned here is the identifier used for AOI-based slick filtering.

---

## EEZ AOIs (`public.aoi_eez`)

### Description
EEZ-specific AOI table. Useful for looking up an AOI id using **MRGID**.

- MRGID reference: https://www.marineregions.org/eezsearch.php

### Endpoints
- `GET /collections/public.aoi_eez/items`

### Example: lookup Greek EEZ AOI id by MRGID

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.aoi_eez/items?mrgid=5679' \
  -H 'Content-Type: application/json;'
```

---

## IHO AOIs (`public.aoi_iho`)

### Description
IHO Sea Area-specific AOI table (similar pattern to EEZ/MPA AOIs).

### Endpoints
- `GET /collections/public.aoi_iho/items`

---

## MPA AOIs (`public.aoi_mpa`)

### Description
MPA-specific AOI table. Useful for looking up AOIs by **WDPAID**.

- WDPA reference: https://www.protectedplanet.net/en/thematic-areas/wdpa?tab=WDPA

### Endpoints
- `GET /collections/public.aoi_mpa/items`

---

## Human Review Classes (`public.cls`)

### Description
Reference table for human-in-the-loop (HITL) classification labels. Includes a `supercls` hierarchy (e.g., `Vessel` and `Infrastructure` may roll up to `Anthropogenic`).

### Endpoints
- `GET /collections/public.cls/items`

### Example: fetch class definitions

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.cls/items' \
  -H 'Content-Type: application/json;'
```

---

## Slicks by AOI (`public.get_slicks_by_aoi`)

### Description
Convenience collection for returning slicks associated with a specific AOI id (EEZ / IHO / MPA), with a configurable collation threshold.

### Endpoints
- `GET /collections/public.get_slicks_by_aoi/items`

### Example 8: return slicks for a specific AOI id

```bash
curl --globoff \
  'https://api.cerulean.skytruth.org/collections/public.get_slicks_by_aoi/items?limit=100&aoi_id=12345&collation_threshold=0' \
  -H 'Content-Type: application/json;'
```

---

## Conclusion

This document reorganizes the “Cerulean API” notebook content into a reference-style REST API guide. It covers common query patterns and examples, but it is not exhaustive. For full interactive documentation and any endpoint-specific details, see:

https://api.cerulean.skytruth.org/
