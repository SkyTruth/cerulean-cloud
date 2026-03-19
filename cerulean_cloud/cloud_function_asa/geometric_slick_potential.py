from pathlib import Path
import pandas as pd
from joblib import load

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _THIS_DIR / "gsp_rf_85_acc_74_F1_20260123.joblib"
_GSP_MODEL = None
_GSP_MODEL_PATH = None


def get_gsp_model(model_path: str):
    global _GSP_MODEL, _GSP_MODEL_PATH
    if _GSP_MODEL is None or _GSP_MODEL_PATH != model_path:
        _GSP_MODEL = load(model_path)
        _GSP_MODEL_PATH = model_path
    return _GSP_MODEL


def predict_geometric_slick_potential(
    slick,
    model_path: Path | str = _DEFAULT_MODEL_PATH,
):
    """
    Compute geometric slick potential from stored slick feature columns.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Geometric slick potential model not found at: {model_path}"
        )

    rf = get_gsp_model(str(model_path))
    feature_names = tuple(rf.feature_names_)
    missing_fields = [
        field for field in feature_names if getattr(slick, field, None) is None
    ]
    if missing_fields:
        raise ValueError(
            f"Cannot compute geometric slick potential; slick is missing fields: {missing_fields}"
        )

    X = pd.DataFrame([{field: getattr(slick, field) for field in feature_names}])[
        list(feature_names)
    ]

    return float(rf.predict_proba(X)[:, 1][0])
