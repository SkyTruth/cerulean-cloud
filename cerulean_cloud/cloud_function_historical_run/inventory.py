"""List Sentinel-1 scene IDs present in the upstream S3 archive."""

from collections.abc import Collection
from datetime import date, timedelta
from xml.etree import ElementTree

import httpx


def list_sentinel1_scene_ids(
    start_date: date,
    end_date: date,
    *,
    modes: Collection[str] | None = None,
    polarizations: Collection[str] | None = None,
) -> list[str]:
    """List GRD scene IDs for inclusive UTC acquisition dates.

    By default, include every mode and polarization. For IW/DV only, pass
    modes=("IW",) and polarizations=("DV",). This lists folder names without
    applying an ocean filter or downloading imagery. Listing errors propagate
    so a failed request cannot silently produce an incomplete inventory.
    """
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    namespace = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    scene_ids = []
    with httpx.Client(timeout=30) as client:

        def folders(prefix):
            params = {"list-type": "2", "delimiter": "/", "prefix": prefix}
            while True:
                response = client.get(
                    "https://sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/",
                    params=params,
                )
                response.raise_for_status()
                page = ElementTree.fromstring(response.content)
                for folder in page.findall(f"{namespace}CommonPrefixes"):
                    yield folder.find(f"{namespace}Prefix").text
                if page.findtext(f"{namespace}IsTruncated") != "true":
                    return
                params["continuation-token"] = page.find(
                    f"{namespace}NextContinuationToken"
                ).text

        for offset in range((end_date - start_date).days + 1):
            day = start_date + timedelta(days=offset)
            for mode_prefix in folders(f"GRD/{day.year}/{day.month}/{day.day}/"):
                mode = mode_prefix.rstrip("/").rsplit("/", 1)[-1]
                if modes is not None and mode not in modes:
                    continue
                for pol_prefix in folders(mode_prefix):
                    polarization = pol_prefix.rstrip("/").rsplit("/", 1)[-1]
                    if polarizations is not None and polarization not in polarizations:
                        continue
                    scene_ids.extend(
                        prefix.rstrip("/").rsplit("/", 1)[-1]
                        for prefix in folders(pol_prefix)
                    )

    return sorted(scene_ids)
