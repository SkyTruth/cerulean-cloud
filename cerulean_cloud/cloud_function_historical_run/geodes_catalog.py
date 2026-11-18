"""Discover GEODES records with one bounded request pool across date windows."""

import os
import random
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import SimpleQueue

import requests
import shapely
from eodag import EODataAccessGateway, setup_logging
from eodag.plugins.search.qssearch import StacSearch
from shapely.geometry import shape
from urllib3.util import Retry

PAGE_SIZE = 80
MAX_WORKERS = 32
WINDOW_SIZE = timedelta(days=7)


class GeodesPageSearch(StacSearch):
    """Use EODAG to format the request and read its page and count together."""

    def do_search(self, **kwargs):
        response = self.session.post(
            self.search_urls[0], json=self.query_params, timeout=(10, 60)
        )
        response.raise_for_status()
        data = response.json()
        self.matched = data.get("context", {}).get("matched")
        return data["features"]

    def normalize_results(self, results, **kwargs):
        # Discovery only reads the record ID, scene identifier, and footprint.
        return results


def date_windows(start, end):
    """Split inclusive query bounds into UTC windows with shared endpoints."""
    start, end = (datetime.fromisoformat(value) for value in (start, end))
    start = (start if start.tzinfo else start.replace(tzinfo=timezone.utc)).astimezone(
        timezone.utc
    )
    end = (end if end.tzinfo else end.replace(tzinfo=timezone.utc)).astimezone(
        timezone.utc
    )
    while end - start > WINDOW_SIZE:
        boundary = start + WINDOW_SIZE
        yield start.isoformat(), boundary.isoformat()
        start = boundary
    yield start.isoformat(), end.isoformat()


def fetch_page(plugin, page, criteria, start, end):
    label = f"GEODES catalog page {page} ({start} to {end})"
    for attempt in range(3):
        try:
            plugin.clear()
            features, _ = plugin.query(
                **criteria,
                startTimeFromAscendingNode=start,
                completionTimeFromAscendingNode=end,
                page=page,
                items_per_page=PAGE_SIZE,
                count=False,
            )
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and status != 429 and status < 500:
                raise RuntimeError(f"{label} failed with HTTP {status}") from exc
            if attempt == 2:
                raise RuntimeError(f"{label} failed after 3 attempts") from exc
            delay = 2 ** (attempt + 1) + random.random()
            if exc.response is not None:
                delay = max(delay, Retry().get_retry_after(exc.response) or 0)
            print(f"Retrying {label} in {delay:.1f}s: {exc}")
            time.sleep(delay)
        else:
            return features, plugin.matched


def search_catalog(criteria, ocean_poly):
    """Fetch and validate all catalog pages, keeping only IDs between requests."""
    try:
        workers = int(os.getenv("GEODES_PAGE_WORKERS", str(MAX_WORKERS)))
    except ValueError:
        workers = 0
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(
            f"GEODES_PAGE_WORKERS must be an integer from 1 to {MAX_WORKERS}"
        )

    criteria = dict(criteria)
    windows = list(date_windows(criteria.pop("start"), criteria.pop("end")))
    setup_logging(2)
    gateway = EODataAccessGateway(
        user_conf_file_path=str(Path(__file__).with_name("geodes.yml"))
    )
    gateway.set_preferred_provider("geodes")
    prepared = gateway._prepare_search(**criteria)
    prototype = prepared.pop("search_plugin")
    if not isinstance(prototype, GeodesPageSearch):
        raise RuntimeError("The GEODES catalog search plugin is unavailable")
    shapely.prepare(ocean_poly)

    jobs = deque((index, 1) for index in range(len(windows)))
    matches = {}
    seen = [set() for _ in windows]
    results = {}
    clients = SimpleQueue()
    print(
        f"Searching {len(windows)} GEODES date windows with up to {workers} requests in flight"
    )

    def query(index, page):
        plugin = clients.get_nowait()
        try:
            return fetch_page(plugin, page, prepared, *windows[index])
        finally:
            clients.put(plugin)

    with ExitStack() as resources, ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in range(workers):
            plugin = deepcopy(prototype)
            plugin.session = resources.enter_context(requests.Session())
            clients.put(plugin)

        pending = {}
        try:
            while jobs or pending:
                while jobs and len(pending) < workers:
                    index, page = jobs.popleft()
                    pending[pool.submit(query, index, page)] = (index, page)
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index, page = pending.pop(future)
                    features, matched = future.result()
                    label = f"GEODES window {index + 1}, page {page}"
                    if type(matched) is not int or matched < 0:
                        raise RuntimeError(f"{label} returned an invalid match count")
                    if page == 1:
                        matches[index] = matched
                        last_page = (matched + PAGE_SIZE - 1) // PAGE_SIZE
                        jobs.extend(
                            (index, next_page) for next_page in range(2, last_page + 1)
                        )
                    elif matched != matches[index]:
                        raise RuntimeError(
                            f"{label} changed the match count during discovery"
                        )
                    expected = min(PAGE_SIZE, matched - (page - 1) * PAGE_SIZE)
                    if len(features) != expected:
                        raise RuntimeError(
                            f"{label} returned {len(features)} of {expected} expected records"
                        )

                    page_results = []
                    for feature in features:
                        record_id = feature["id"]
                        if record_id in seen[index]:
                            raise RuntimeError(
                                f"{label} repeated catalog record {record_id}"
                            )
                        seen[index].add(record_id)
                        scene_id = feature["properties"]["identifier"]
                        if not (
                            scene_id.split("_")[2] in ("GRDH", "GRDM")
                            and ocean_poly.intersects(shape(feature["geometry"]))
                        ):
                            scene_id = None
                        page_results.append((record_id, scene_id))
                    results[index, page] = page_results
                    print(f"{label}: {len(features)} products")
        finally:
            for future in pending:
                future.cancel()

    # Windows share an inclusive boundary. Count each catalog record only once.
    unique = {}
    for key in sorted(results):
        unique.update(results[key])
    return len(unique), [
        scene_id for scene_id in unique.values() if scene_id is not None
    ]
