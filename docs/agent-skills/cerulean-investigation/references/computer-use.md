# Cerulean map with computer use

Open `https://cerulean.skytruth.org`. Searching needs no account. SkyTruth's guides cover [search and filter](https://skytruth.org/cerulean/guide-search-and-filter), [slick details](https://skytruth.org/cerulean/guide-slick-details) and [source profiles](https://skytruth.org/cerulean/guide-source-profile). Labels below match the English interface on 2026-10-25. Where they differ, use the current labels and accessible names; take a fresh screenshot after each action, and never reuse saved coordinates.

## Default filters

The map opens with these URL parameters. Keep them unless a deliberate, recorded change is part of the question, and save the final URL.

| URL parameter | Default | Meaning |
|---|---|---|
| `date_range` | the last 12 months | UTC dates, inclusive |
| `sources` | `1_2_3` | At least one qualifying candidate: vessel, infrastructure or dark vessel. Adding `0` ("No potential sources identified nearby") also admits detections without one. |
| `source_limit` | `3` | Candidates ranked 1–3 qualify; reviewer-verified candidates qualify at any rank. |
| `source_score` | `0_Infinity` | Collated score of 0 or more, unless the candidate is reviewer-verified. |
| `slick_confidence` | `0.8` | Geometric score threshold, applied only to detections without a qualifying candidate. |
| `min_confidence` | `0.7` | Legacy model threshold; never interpret it. |
| `classes` | `2_3_4_5_6_7_8` | The CNN class filter. Leave it at the default and never interpret the class. |
| `hitl_classes` | `2_3_4_5_6_7_8_9` | Latest human class, or no review; excludes human not-oil. |
| `hitl_review` | `0_1` | Reviewed and unreviewed detections. |
| `area` | `0_Infinity` | Polygon size in km². |

The map reads URL bbox values as integers (`bbox=11_36_20_43`). `lat`, `lon` and `zoom` move the camera and do not filter.

[This example search](https://cerulean.skytruth.org/?area=0_Infinity&classes=2_3_4_5_6_7_8&date_range=2025-05-01_2025-05-08&hitl_classes=2_3_4_5_6_7_8_9&hitl_review=0_1&min_confidence=0.7&slick_confidence=0.8&source_limit=3&source_score=0_Infinity&sources=1_2_3&bbox=11_36_20_43) (11–20°E 36–43°N, 1–8 May 2025) shows "5 Slicks | 7 km²". That summary is a record count and a rounded sum of polygon areas, not spills or contaminated area.

## Tasks

| Task | Steps | Check |
|---|---|---|
| Set dates | SLICK FILTERS → date range; set both ends; close the picker | URL `date_range` and the panel agree, and the count updates |
| Select an area | AREAS OF INTEREST → enable MPAs or EEZs → click the area; choose among overlaps if asked | A named filter chip appears and the count recomputes. Showing a layer alone does not filter. |
| Draw or upload an area | AREAS OF INTEREST → Draw or upload an AOI; uploads take WGS84 GeoJSON (under 5 MB per the guide) | The outline shows and the count recomputes. Save the geometry yourself; browser storage is not shareable. |
| Vessel or structure | Search for Source by ID, Name, Flag or Tag → choose the matching MMSI or structure ID | A chip appears and results update. This view drops associations a reviewer rejected. |
| Inspect a detection | Click its outline, or open its `slick_url` | Slick details show time, length, area, location, S1 scene ID and potential sources |
| Imagery and context | Detail icons: S1 satellite imagery, slick outline, other slicks, wind, ocean current, zoom to slick or scene | Wait for tiles to load; a failed tile is not clear water |
| Candidate sources | POTENTIAL SOURCES → expand a card; View source profile | Identifier, scores and review status. A source profile applies its own filters, so do not copy its totals into a filtered report. |
| Review status | Symbol beside the heading or card | Unreviewed, verified, ambiguous or not oil; slick and source reviews are separate |
| Share | Copy the URL after the filters settle, then reopen it | Dates, area and filters survive |

## Exports

- **Download slick polygon as GeoJSON** exports one detection.
- **Download a screenshot** keeps no filter record; save the URL with it.
- **Generate incident report** produces text; review its wording before use.

For many records, use the API.

Do not create accounts, sign in, request verification or submit anything unless the user asks, and do not look for staff-only controls. If the interface differs from this guide, record the difference and continue through the API where possible.
