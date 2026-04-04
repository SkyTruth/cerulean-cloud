from datetime import date

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from cerulean_cloud.cloud_run_sea_ice.logic import (
    build_object_name,
    candidate_source_days,
    nhsi_filename_for_day,
    nhsi_source_url_for_day,
    normalize_mask_polygons,
    parse_gcs_uri,
    should_run_today,
)


def test_should_run_today_daily():
    assert should_run_today(date(2026, 3, 19), date(2026, 3, 19), 1) is True
    assert should_run_today(date(2026, 3, 20), date(2026, 3, 19), 1) is True


def test_should_run_today_every_three_days():
    assert should_run_today(date(2026, 3, 19), date(2026, 3, 19), 3) is True
    assert should_run_today(date(2026, 3, 22), date(2026, 3, 19), 3) is True
    assert should_run_today(date(2026, 3, 20), date(2026, 3, 19), 3) is False


def test_should_not_run_before_anchor_date():
    assert should_run_today(date(2026, 3, 18), date(2026, 3, 19), 7) is False


def test_should_run_today_rejects_invalid_cadence():
    with pytest.raises(ValueError, match="cadence_days"):
        should_run_today(date(2026, 3, 19), date(2026, 3, 19), 0)


def test_build_object_name():
    bucket_name, object_name = build_object_name(
        "gs://cerulean-ice/extent_vectors/",
        date(2026, 3, 27),
    )
    assert bucket_name == "cerulean-ice"
    assert object_name == "extent_vectors/2026-03-27_extent.geojson"


def test_build_object_name_rejects_file_style_uri():
    with pytest.raises(ValueError, match="prefix directory, not a file path"):
        build_object_name(
            "gs://cerulean-ice/extent_vectors/daily_ice_extent.geojson",
            date(2026, 3, 27),
        )


def test_normalize_mask_polygons_strips_holes():
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

    result = normalize_mask_polygons(gdf)

    assert len(result) == 1
    assert len(result.geometry.iloc[0].interiors) == 0
    assert result.geometry.iloc[0].area == 16


def test_normalize_mask_polygons_dissolves_overlaps():
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

    result = normalize_mask_polygons(gdf)

    assert len(result) == 1
    assert result.iloc[0]["DN"] == 3
    assert result.iloc[0]["ice_date"] == "2026-03-27"
    assert result.geometry.iloc[0].area == 6


def test_parse_gcs_uri_rejects_invalid_values():
    with pytest.raises(ValueError, match="Invalid GCS URI"):
        parse_gcs_uri("https://example.com/not-gcs.geojson")


def test_candidate_source_days_returns_today_then_lookback():
    assert candidate_source_days(date(2026, 3, 27)) == [
        date(2026, 3, 27),
        date(2026, 3, 26),
        date(2026, 3, 25),
    ]


def test_nhsi_filename_for_day_uses_yyyydoy():
    assert nhsi_filename_for_day(date(2026, 1, 1)) == "ims2026001_4km_GIS_v1.3.tif.gz"


def test_nhsi_source_url_for_day_builds_from_base_url():
    assert (
        nhsi_source_url_for_day(
            "https://noaadata.apps.nsidc.org/NOAA/G02156/GIS/4km",
            date(2026, 3, 27),
        )
        == "https://noaadata.apps.nsidc.org/NOAA/G02156/GIS/4km/2026/ims2026086_4km_GIS_v1.3.tif.gz"
    )


def test_nhsi_source_url_for_day_supports_templates():
    assert (
        nhsi_source_url_for_day(
            "https://example.com/{yyyy}/{file_name}",
            date(2026, 3, 27),
        )
        == "https://example.com/2026/ims2026086_4km_GIS_v1.3.tif.gz"
    )
