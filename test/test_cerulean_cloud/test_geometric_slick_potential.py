from types import SimpleNamespace

import pytest

from cerulean_cloud.cloud_function_asa.geometric_slick_potential import (
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


def test_predict_geometric_slick_potential_from_stored_fields():
    slick = make_slick()

    prediction = predict_geometric_slick_potential(slick)

    assert isinstance(prediction, float)
    assert 0.0 <= prediction <= 1.0


def test_predict_geometric_slick_potential_requires_all_fields():
    slick = make_slick(median_area=None)

    with pytest.raises(ValueError, match="median_area"):
        predict_geometric_slick_potential(slick)


def test_predict_geometric_slick_potential_missing_model_raises(tmp_path):
    slick = make_slick()

    with pytest.raises(FileNotFoundError, match="Geometric slick potential model"):
        predict_geometric_slick_potential(slick, model_path=tmp_path / "missing.joblib")
