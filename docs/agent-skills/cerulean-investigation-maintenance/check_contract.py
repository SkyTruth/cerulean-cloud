#!/usr/bin/env python3
"""Re-check the live Cerulean API behavior that the cerulean-investigation skill documents.

Run it before shipping a new zip and after deploys that change TiPG collections or the
slick views. It prints one line per check and exits 1 on any difference; update the skill
and expected_contract.json together. The map checks at the end are manual. Every request
carries the skill's two attribution parameters, which also checks that the API accepts them.
"""

import csv
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.cerulean.skytruth.org/collections"
SCREEN = "machine_confidence >= 0.7 AND max_source_collated_score >= 0"
EXPECTED = json.loads(
    (Path(__file__).resolve().parent / "expected_contract.json").read_text()
)
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "cerulean-investigation-contract-check",
}
UTM = {"utm_medium": "agentic", "utm_source": "skytruth-contract-check"}
CHECKS = []


def check(name):
    def register(function):
        CHECKS.append((name, function))
        return function

    return register


def fetch(url, **params):
    url += "?" + urllib.parse.urlencode({**UTM, **params})
    request = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read()
    finally:
        time.sleep(1)


def get(url, **params):
    return json.loads(fetch(url, **params))


def status(url, **params):
    try:
        get(url, **params)
        return 200
    except urllib.error.HTTPError as error:
        return error.code


def rows(collection, **params):
    # Without f=geojson, Accept: application/json returns a plain list for some collections.
    data = get(f"{API}/{collection}/items", f="geojson", **params)
    return [
        dict(f["properties"], id=f["properties"].get("id", f["id"]))
        for f in data["features"]
    ]


def example_window(**params):
    example = EXPECTED["example"]
    window = {
        "datetime": f"{example['start']}T00:00:00Z/{example['end']}T23:59:59Z",
        "limit": 10000,
        "bbox-only": "true",
    }
    window.update(params)
    return window


@check("collections")
def collections():
    listed = get(API, f="json")["collections"]
    return EXPECTED["collections"], sorted(item["id"] for item in listed)


@check("slick_plus queryables")
def slick_queryables():
    schema = get(f"{API}/public.slick_plus/queryables")
    return EXPECTED["slick_plus_queryables"], sorted(schema["properties"])


@check("source_plus queryables (still no review fields)")
def source_queryables():
    schema = get(f"{API}/public.source_plus/queryables")
    return EXPECTED["source_plus_queryables"], sorted(schema["properties"])


@check("class, source and AOI vocabularies")
def vocabularies():
    actual = {
        "classes": {
            str(r["id"]): [r["long_name"], r["supercls"]]
            for r in rows("public.cls", limit=100)
        },
        "source_types": {
            str(r["id"]): r["short_name"] for r in rows("public.source_type", limit=20)
        },
        "aoi_types": {
            str(r["id"]): r["short_name"] for r in rows("public.aoi_type", limit=20)
        },
    }
    return {key: EXPECTED[key] for key in actual}, actual


@check("items limit default and maximum")
def limit():
    for path, spec in get("https://api.cerulean.skytruth.org/api")["paths"].items():
        if path.endswith("/items"):
            schema = next(
                p["schema"] for p in spec["get"]["parameters"] if p["name"] == "limit"
            )
            return EXPECTED["items_limit"], {
                "default": schema["default"],
                "maximum": schema["maximum"],
            }


@check("worked example 4111397 and its candidates")
def detail():
    example = EXPECTED["detail"]
    slick = get(f"{API}/public.slick_plus/items/{example['id']}", f="geojson")[
        "properties"
    ]
    candidates = sorted(
        rows(
            "public.source_plus",
            slick_id=example["id"],
            limit=100,
            **{"bbox-only": "true"},
        ),
        key=lambda r: r["source_rank"],
    )
    aliased = {round(r["slick_confidence"], 9) for r in candidates} == {
        round(slick["machine_confidence"], 9)
    }
    expected = (
        example["hitl_cls"],
        example["rank1_mmsi"],
        example["candidates"],
        True,
        example["slick_url"],
    )
    return expected, (
        slick["hitl_cls"],
        candidates[0]["mmsi_or_structure_id"],
        len(candidates),
        aliased,
        slick["slick_url"],
    )


@check("quick-start screen in the example box")
def screen_example():
    example = EXPECTED["example"]
    found = rows(
        "public.slick_plus",
        **example_window(
            bbox=",".join(map(str, example["bbox"])), filter=SCREEN, properties="id"
        ),
    )
    return example["ids"], sorted(r["id"] for r in found)


@check("polygon example")
def polygon_example():
    polygon = EXPECTED["polygon"]
    spatial = f"{SCREEN} AND S_INTERSECTS(geometry, {polygon['wkt']})"
    found = rows("public.slick_plus", **example_window(filter=spatial, properties="id"))
    return polygon["ids"], sorted(r["id"] for r in found)


@check("IS NOT NULL still returns the null rows")
def is_not_null():
    bbox = ",".join(map(str, EXPECTED["example"]["bbox"]))

    def ids(expression):
        params = example_window(bbox=bbox, filter=expression, properties="id")
        return {r["id"] for r in rows("public.slick_plus", **params)}

    null = ids("max_source_collated_score IS NULL")
    not_null = ids("max_source_collated_score IS NOT NULL")
    workaround = ids("max_source_collated_score >= -1000")
    # Null rows exist, IS NOT NULL returns exactly them, and the documented
    # workaround returns only scored rows.
    return (True, True, True), (
        bool(null),
        not_null == null,
        bool(workaround) and not workaround & null,
    )


@check("datetime without times returns HTTP 500")
def date_only():
    return 500, status(
        f"{API}/public.slick_plus/items",
        datetime="2025-05-01/2025-05-08",
        properties="id",
        limit=1,
        **{"bbox-only": "true"},
    )


@check("CSV columns and EWKT geometry")
def csv_geometry():
    expected = EXPECTED["csv"]
    text = fetch(
        f"{API}/public.slick_plus/items",
        ids=EXPECTED["detail"]["id"],
        properties="id",
        f="csv",
    ).decode()
    reader = csv.DictReader(io.StringIO(text))
    geometry = next(reader)["geometry"]
    return (expected["columns"], True), (
        reader.fieldnames,
        geometry.startswith(expected["geometry_prefix"]),
    )


@check("coverage scene query in the example box")
def coverage():
    example = EXPECTED["example"]
    data = get(
        f"{API}/public.sentinel1_grd/items",
        bbox=",".join(map(str, example["bbox"])),
        datetime=f"{example['start']}T00:00:00Z/{example['end']}T23:59:59Z",
        f="geojson",
        sortby="id",
        limit=100,
        **{"datetime-column": "start_time"},
    )
    features = data["features"]
    return EXPECTED["coverage"], {
        "matched": data["numberMatched"],
        "scenes": len({f["properties"]["scene_id"] for f in features}),
        "properties": sorted({key for f in features for key in f["properties"]}),
        "geometry_types": sorted({f["geometry"]["type"] for f in features}),
        # A bounding box has 5 ring vertices; full footprints have more.
        "full_geometry": any(
            len(f["geometry"]["coordinates"][0]) > 5
            for f in features
            if f["geometry"]["type"] == "Polygon"
        ),
    }


@check("ids= fails at 950 IDs, works at 900")
def ids_limit():
    def ids(n):
        return ",".join(str(4111000 + i) for i in range(n))

    url = f"{API}/public.slick_plus/items"
    common = {"properties": "id", "bbox-only": "true", "limit": 1000}
    return (200, 500), (
        status(url, ids=ids(900), **common),
        status(url, ids=ids(950), **common),
    )


@check("AOI lookup and membership filter")
def aoi():
    expected = EXPECTED["aoi"]
    found = rows("public.aoi_eez", mrgid=expected["mrgid"])
    aoi_id = found[0]["aoi_id"] if found else None
    slicks = rows(
        "public.slick_plus",
        **example_window(filter=SCREEN, properties="id,aoi_type_1_ids"),
    )
    members = sorted(
        r["id"] for r in slicks if aoi_id in (r.get("aoi_type_1_ids") or [])
    )
    return (expected["aoi_id"], expected["ids"]), (aoi_id, members)


@check("vessel candidate rows passing rank and score screen")
def vessel():
    v = EXPECTED["vessel"]
    candidates = rows(
        "public.source_plus",
        mmsi_or_structure_id=v["mmsi"],
        properties="slick_id,source_rank,source_collated_score",
        limit=10000,
        **{"bbox-only": "true"},
    )
    screened = sorted(
        {
            r["slick_id"]
            for r in candidates
            if r["source_rank"] <= 3 and r["source_collated_score"] >= 0
        }
    )
    return (v["rows"], v["screened_ids"]), (len(candidates), screened)


def main():
    failures = 0
    for name, function in CHECKS:
        try:
            expected, actual = function()
        except Exception as error:  # Report every check, even after one fails.
            failures += 1
            print(f"ERROR {name}: {error}")
            continue
        if expected == actual:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"DIFF  {name}: expected {expected!r}, got {actual!r}")
    for line in EXPECTED["manual_map_checks"]:
        print(f"MANUAL {line}")
    matched = len(CHECKS) - failures
    print(
        f"{matched}/{len(CHECKS)} checks match expected_contract.json ({EXPECTED['checked_at']})."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
