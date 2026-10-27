# Interpreting Cerulean data

Public sources: [methods](https://skytruth.org/cerulean/methods), [FAQ](https://skytruth.org/faq#cerulean), [slick details guide](https://skytruth.org/cerulean/guide-slick-details).

## What a detection is

Cerulean runs a model over Sentinel-1 radar images and outlines dark, smooth patches of sea. Oil smooths the surface; so do calm water, wind shadows, natural films, algae and internal waves. A missing detection can mean no image, unsuitable wind, a processing gap, a model miss or a filter.

Source matching is a second model. It compares each detection with AIS vessel tracks from 8 hours before to 6 hours after the image, with nearby infrastructure, and with radar-detected vessels that broadcast no AIS ("dark"). The collated score ranks candidates; it is neither a probability nor an observation of a discharge. AIS can be missing for many reasons, so a missing track does not show that a crew switched it off.

## Evidence ladder

| Evidence | Supports | Does not support alone |
|---|---|---|
| Radar image and polygon | A dark surface pattern at a place and time | Oil, volume, source, one distinct event |
| Model class `cls` | Nothing; ignore it | — |
| `machine_confidence` | The map's 0.7 threshold | Any ranking or interpretation |
| Geometric score | Screening detections that have no source match | Oil, source, probability |
| Collated source score | Ranking candidate sources for review | Responsibility, probability |
| Latest human class | "Human-classified apparent oil", ambiguous, or not oil | Independent confirmation; the reviewer saw the same imagery |
| Reviewer-verified source | "Association judged likely by a reviewer" | Responsibility, intent, legality |
| Independent evidence: other imagery, field sample, incident report | The specific claim it covers | Other claims |

## Recent data

The FAQ describes imagery arriving 6–12 hours after acquisition and vessel data up to about 72 hours later. Source matches can take longer: two examples were computed 10 and 28 days after their images. Recent detections without a source match are provisional. Keep them separate, record image and retrieval times, and recheck later. Never promise completion within 72 hours.

## Before comparing areas, periods, vessels or companies

1. **Population:** detection records, reviewed detections, candidate associations, or grouped events?
2. **Coverage:** how often each area was imaged (`public.sentinel1_grd`). Over the 12 months to 2026-10-26, the inventory held 260 scenes over a point in the central North Sea and 0 over one in Brazil's Campos Basin.
3. **Conditions:** wind, sea state, season, processing gaps.
4. **Dependence:** repeated or overlapping detections of one event are not independent.
5. **Selection:** the same filters, dates and review effort. Source matching favors AIS-broadcasting vessels and mapped infrastructure.
6. **Method versions:** model and matching code (`git_tag` in `source_plus`) can differ between records.
7. **Denominator:** detections per observed pass, not per calendar time.

Without these, report a descriptive sample, not rates, trends or rankings of polluters.

Area is in m² in the API and km² on the map. A sum of polygon areas counts overlapping coverage more than once; it measures neither unique area nor volume.

## Wording

| Avoid | Use |
|---|---|
| Confirmed oil spills | Potential slick detections; "human-classified apparent oil" where reviewed |
| An 80% probability of oil | Score 0.8; no probability calibration is published |
| Vessel X dumped oil illegally | A model associated this detection with vessel X; cause, responsibility and legality are unresolved |
| Five spills | Five detection records from N scenes; how they relate as events is unknown |
| The area is clean | No detections matched this query at retrieval time |
| Worst polluters | Candidate associations prioritized for review |

Useful next evidence: imagery from other sensors, field observations, operator or authority incident reports, expert review. You may draft a verification request; sending it, or contacting anyone, needs the user's authorization.

## Terms and attribution

SkyTruth's [terms of service](https://skytruth.org/terms-of-service), effective 2026-05-04, limit use to environmental conservation and license data, maps and images CC BY-SA 4.0: credit SkyTruth, link the license, and disclose your modifications. AIS-derived content may carry Global Fishing Watch terms.

Citation: SkyTruth Cerulean, [year], [map](https://cerulean.skytruth.org/) / [API](https://api.cerulean.skytruth.org/), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); accessed [UTC date]. Filtering and analysis by [your organization]; modifications: [description]. SkyTruth has not reviewed these conclusions.
