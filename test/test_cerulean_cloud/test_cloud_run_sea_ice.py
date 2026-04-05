import importlib
import sys
from datetime import date, timedelta
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
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


def test_find_latest_available_source_walks_back_until_source_exists(monkeypatch):
    sea_ice_handler = load_handler(monkeypatch)
    today = date(2026, 3, 27)
    available_day = today - timedelta(days=2)
    responses = {
        sea_ice_handler.nhsi_source_url_for_day(
            "https://example.com/root",
            today,
        ): FakeResponse(404),
        sea_ice_handler.nhsi_source_url_for_day(
            "https://example.com/root",
            today - timedelta(days=1),
        ): FakeResponse(404),
        sea_ice_handler.nhsi_source_url_for_day(
            "https://example.com/root",
            available_day,
        ): FakeResponse(200),
    }

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, stream=True, timeout=None):
            self.calls.append((url, stream, timeout))
            return responses[url]

    session = FakeSession()
    install_fake_requests(monkeypatch, session)

    result = sea_ice_handler.find_latest_available_source(
        "https://example.com/root",
        today=today,
        timeout_seconds=123,
        max_lookback_days=5,
    )

    assert result.source_date == available_day
    assert result.source_url == sea_ice_handler.nhsi_source_url_for_day(
        "https://example.com/root",
        available_day,
    )
    assert [call[0] for call in session.calls] == [
        sea_ice_handler.nhsi_source_url_for_day(
            "https://example.com/root",
            today,
        ),
        sea_ice_handler.nhsi_source_url_for_day(
            "https://example.com/root",
            today - timedelta(days=1),
        ),
        sea_ice_handler.nhsi_source_url_for_day(
            "https://example.com/root",
            available_day,
        ),
    ]


def test_polygonize_ice_raster_only_emits_ice_class(monkeypatch, tmp_path):
    sea_ice_handler = load_handler(monkeypatch)
    raster_path = tmp_path / "ice.tif"

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype=np.uint8,
        crs="EPSG:3413",
        transform=rasterio.transform.from_origin(0, 8000, 4000, 4000),
    ) as dataset:
        dataset.write(np.array([[3, 3], [1, 0]], dtype=np.uint8), 1)

    gdf = sea_ice_handler.polygonize_ice_raster(raster_path)

    assert len(gdf) == 1
    assert set(gdf["DN"]) == {sea_ice_handler.ICE_CLASS_VALUE}
    assert gdf.crs.to_epsg() == 3413


def test_remove_land_parts_reprojects_representative_points(monkeypatch):
    sea_ice_handler = load_handler(monkeypatch)
    calls = []

    def fake_is_land(lat, lon):
        calls.append((lat, lon))
        return False

    monkeypatch.setitem(
        sys.modules,
        "global_land_mask",
        SimpleNamespace(globe=SimpleNamespace(is_land=fake_is_land)),
    )

    gdf = gpd.GeoDataFrame(
        {"DN": [3]},
        geometry=[
            Polygon(
                [
                    (1_000_000, 2_000_000),
                    (1_001_000, 2_000_000),
                    (1_001_000, 2_001_000),
                    (1_000_000, 2_001_000),
                    (1_000_000, 2_000_000),
                ]
            )
        ],
        crs="EPSG:3857",
    )

    result = sea_ice_handler.remove_land_parts(gdf)

    expected_point = gdf.geometry.representative_point().to_crs("EPSG:4326").iloc[0]
    assert len(result) == 1
    assert calls == [pytest.approx((expected_point.y, expected_point.x))]


def test_finalize_output_geometries_repairs_invalid_polygon(monkeypatch):
    sea_ice_handler = load_handler(monkeypatch)
    bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    gdf = gpd.GeoDataFrame(
        {"DN": [3], "ice_date": [date(2026, 3, 27).isoformat()]},
        geometry=[bowtie],
        crs="EPSG:4326",
    )

    result = sea_ice_handler.finalize_output_geometries(gdf)

    assert not result.empty
    assert result.geometry.is_valid.all()


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


def test_sync_returns_up_to_date_when_latest_available_source_is_vectorized(
    monkeypatch,
):
    sea_ice_handler = load_handler(monkeypatch)
    source_day = date(2026, 3, 26)
    monkeypatch.setenv("SEA_ICE_MASK_GCS_URI", "gs://example-bucket/extent_vectors/")
    monkeypatch.setattr(sea_ice_handler, "utc_today", lambda: date(2026, 3, 27))
    monkeypatch.setattr(
        sea_ice_handler,
        "find_latest_available_source",
        lambda *args, **kwargs: sea_ice_handler.AvailableSource(
            source_date=source_day,
            source_url="https://example.com/source.tif.gz",
        ),
    )

    exists_calls = []

    def fake_vector_output_exists(bucket_name, object_name):
        exists_calls.append((bucket_name, object_name))
        return True

    monkeypatch.setattr(
        sea_ice_handler,
        "vector_output_exists",
        fake_vector_output_exists,
    )
    monkeypatch.setattr(
        sea_ice_handler,
        "download_latest_source",
        lambda *args, **kwargs: pytest.fail("should not download an existing vector"),
    )

    result = sea_ice_handler._sync()

    assert result == {
        "status": "up_to_date",
        "object_name": "extent_vectors/2026-03-26_extent.geojson",
        "source_date": "2026-03-26",
    }
    assert exists_calls == [
        ("example-bucket", "extent_vectors/2026-03-26_extent.geojson")
    ]


def test_sync_returns_up_to_date_when_upload_precondition_fails(
    monkeypatch,
    tmp_path,
):
    from google.api_core.exceptions import PreconditionFailed

    sea_ice_handler = load_handler(monkeypatch)
    source_day = date(2026, 3, 26)
    monkeypatch.setenv("SEA_ICE_MASK_GCS_URI", "gs://example-bucket/extent_vectors/")
    monkeypatch.setattr(sea_ice_handler, "utc_today", lambda: date(2026, 3, 27))
    monkeypatch.setattr(
        sea_ice_handler,
        "find_latest_available_source",
        lambda *args, **kwargs: sea_ice_handler.AvailableSource(
            source_date=source_day,
            source_url="https://example.com/source.tif.gz",
        ),
    )
    monkeypatch.setattr(
        sea_ice_handler,
        "vector_output_exists",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        sea_ice_handler,
        "download_latest_source",
        lambda *args, **kwargs: sea_ice_handler.DownloadedSource(
            source_date=source_day,
            source_url="https://example.com/source.tif.gz",
            gz_path=tmp_path / "source.tif.gz",
        ),
    )
    monkeypatch.setattr(
        sea_ice_handler,
        "process_downloaded_source",
        lambda *args, **kwargs: tmp_path / "extent.geojson",
    )

    def fake_upload_outputs(*args, **kwargs):
        del args, kwargs
        raise PreconditionFailed("already exists")

    monkeypatch.setattr(sea_ice_handler, "upload_outputs", fake_upload_outputs)

    result = sea_ice_handler._sync()

    assert result == {
        "status": "up_to_date",
        "object_name": "extent_vectors/2026-03-26_extent.geojson",
        "source_date": "2026-03-26",
    }
