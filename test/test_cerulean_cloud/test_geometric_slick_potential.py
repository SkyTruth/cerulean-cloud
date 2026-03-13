from types import SimpleNamespace

import pytest
from shapely.geometry import MultiPolygon, box

from cerulean_cloud.cloud_function_asa.geometric_slick_potential import (
    gsp_feature_frame_from_slick,
    predict_geometric_slick_potential,
)


def make_slick(**overrides):
    values = {
        "area": 1_000_000.0,
        "polsby_popper": 0.15,
        "fill_factor": 0.8,
        "aspect_ratio_factor": 0.7,
        "geometry_count": 1,
        "largest_area": 1_000_000.0,
        "median_area": 1_000_000.0,
        "machine_confidence": 0.9,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gsp_feature_frame_from_slick_uses_stored_fields():
    geometry = MultiPolygon([box(0, 0, 1, 1)])
    slick = make_slick()

    frame = gsp_feature_frame_from_slick(slick, geometry)

    assert frame["geometry_count"].iloc[0] == 1
    assert frame["largest_area"].iloc[0] == pytest.approx(1_000_000.0)
    assert frame["median_area"].iloc[0] == pytest.approx(1_000_000.0)
    assert frame["machine_confidence"].iloc[0] == pytest.approx(0.9)


def test_gsp_feature_frame_from_slick_requires_all_fields():
    geometry = MultiPolygon([box(0, 0, 1, 1)])
    slick = make_slick(median_area=None)

    with pytest.raises(ValueError, match="median_area"):
        gsp_feature_frame_from_slick(slick, geometry)


def test_predict_geometric_slick_potential_from_stored_fields():
    geometry = MultiPolygon([box(0, 0, 1, 1)])
    slick = make_slick()
    frame = gsp_feature_frame_from_slick(slick, geometry)

    prediction = float(predict_geometric_slick_potential(frame, preprocess=False)[0])

    assert 0.0 <= prediction <= 1.0
