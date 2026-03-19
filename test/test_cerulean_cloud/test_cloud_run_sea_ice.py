from datetime import date

import pytest

from cerulean_cloud.cloud_run_sea_ice.logic import (
    build_object_names,
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


def test_build_object_names():
    bucket_name, archive_name, latest_name = build_object_names(
        "gs://cerulean-ice/extent_vectors/test.geojson", date(2026, 3, 19)
    )
    assert bucket_name == "cerulean-ice"
    assert archive_name == "extent_vectors/archive/2026-03-19-test.geojson"
    assert latest_name == "extent_vectors/test.geojson"


def test_parse_gcs_uri_rejects_invalid_values():
    with pytest.raises(ValueError, match="Invalid GCS URI"):
        parse_gcs_uri("https://example.com/not-gcs.geojson")
