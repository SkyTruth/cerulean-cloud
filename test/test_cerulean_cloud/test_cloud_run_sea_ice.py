import importlib
import sys
from datetime import date
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import Polygon


class FakeResponse:
    def __init__(self, status_code: int, body: bytes = b""):
        self.status_code = status_code
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        if self._body:
            yield self._body


def install_fake_requests(monkeypatch, session):
    class FakeRequestException(Exception):
        pass

    class FakeHTTPError(FakeRequestException):
        def __init__(self, response=None):
            super().__init__("http error")
            self.response = response

    monkeypatch.setitem(
        sys.modules,
        "requests",
        SimpleNamespace(
            Session=lambda: session,
            RequestException=FakeRequestException,
            HTTPError=FakeHTTPError,
        ),
    )
    return FakeRequestException


def load_handler(monkeypatch):
    class FakeApp:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def get(self, *args, **kwargs):
            del args, kwargs

            def decorator(func):
                return func

            return decorator

    monkeypatch.setitem(sys.modules, "fastapi", SimpleNamespace(FastAPI=FakeApp))
    sys.modules.pop("cerulean_cloud.cloud_run_sea_ice.handler", None)
    return importlib.import_module("cerulean_cloud.cloud_run_sea_ice.handler")


def test_default_simplify_tolerance_is_ten_kilometers(monkeypatch):
    sea_ice_handler = load_handler(monkeypatch)
    assert sea_ice_handler.DEFAULT_SIMPLIFY_TOLERANCE == 10_000.0


def test_normalize_mask_polygons_strips_holes(monkeypatch):
    sea_ice_handler = load_handler(monkeypatch)
    gdf = gpd.GeoDataFrame(
        {"DN": [3], "ice_date": [date(2026, 3, 27).isoformat()]},
        geometry=[
            Polygon(
                shell=[(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
                holes=[[(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]],
            )
        ],
        crs="EPSG:4326",
    )

    result = sea_ice_handler.normalize_mask_polygons(gdf)

    assert len(result) == 1
    assert len(result.geometry.iloc[0].interiors) == 0
    assert result.geometry.iloc[0].area == 16


def test_normalize_mask_polygons_dissolves_overlaps(monkeypatch):
    sea_ice_handler = load_handler(monkeypatch)
    gdf = gpd.GeoDataFrame(
        {
            "DN": [3, 3],
            "ice_date": [date(2026, 3, 27).isoformat()] * 2,
        },
        geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]),
            Polygon([(1, 0), (3, 0), (3, 2), (1, 2), (1, 0)]),
        ],
        crs="EPSG:4326",
    )

    result = sea_ice_handler.normalize_mask_polygons(gdf)

    assert len(result) == 1
    assert result.iloc[0]["DN"] == 3
    assert result.iloc[0]["ice_date"] == "2026-03-27"
    assert result.geometry.iloc[0].area == 6


def test_nhsi_filename_for_day_uses_yyyydoy(monkeypatch):
    sea_ice_handler = load_handler(monkeypatch)
    assert (
        sea_ice_handler.nhsi_filename_for_day(date(2026, 1, 1))
        == "ims2026001_4km_GIS_v1.3.tif.gz"
    )


def test_nhsi_source_url_for_day_builds_from_base_url(monkeypatch):
    sea_ice_handler = load_handler(monkeypatch)
    assert (
        sea_ice_handler.nhsi_source_url_for_day(
            "https://noaadata.apps.nsidc.org/NOAA/G02156/GIS/4km",
            date(2026, 3, 27),
        )
        == "https://noaadata.apps.nsidc.org/NOAA/G02156/GIS/4km/2026/ims2026086_4km_GIS_v1.3.tif.gz"
    )


def test_nhsi_source_url_for_day_supports_templates(monkeypatch):
    sea_ice_handler = load_handler(monkeypatch)
    assert (
        sea_ice_handler.nhsi_source_url_for_day(
            "https://example.com/{yyyy}/{file_name}",
            date(2026, 3, 27),
        )
        == "https://example.com/2026/ims2026086_4km_GIS_v1.3.tif.gz"
    )


def test_download_latest_source_404_skips_today_without_looking_back(
    monkeypatch, tmp_path
):
    sea_ice_handler = load_handler(monkeypatch)
    today = date(2026, 3, 27)
    responses = {
        sea_ice_handler.nhsi_source_url_for_day(
            "https://example.com/root", today
        ): FakeResponse(404)
    }

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, stream=True, timeout=None):
            self.calls.append((url, stream, timeout))
            return responses[url]

    session = FakeSession()
    install_fake_requests(monkeypatch, session)

    with pytest.raises(FileNotFoundError, match="2026-03-27"):
        sea_ice_handler.download_latest_source(
            "https://example.com/root",
            tmp_path,
            today=today,
            timeout_seconds=123,
        )

    assert [call[0] for call in session.calls] == [
        sea_ice_handler.nhsi_source_url_for_day("https://example.com/root", today)
    ]


def test_download_latest_source_raises_on_transient_failure(monkeypatch, tmp_path):
    sea_ice_handler = load_handler(monkeypatch)
    today = date(2026, 3, 27)

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, stream=True, timeout=None):
            self.calls.append((url, stream, timeout))
            raise request_exception("timeout talking to NHSI")

    session = FakeSession()
    request_exception = install_fake_requests(monkeypatch, session)

    with pytest.raises(request_exception, match="timeout talking to NHSI"):
        sea_ice_handler.download_latest_source(
            "https://example.com/root",
            tmp_path,
            today=today,
            timeout_seconds=123,
        )

    assert [call[0] for call in session.calls] == [
        sea_ice_handler.nhsi_source_url_for_day("https://example.com/root", today)
    ]
