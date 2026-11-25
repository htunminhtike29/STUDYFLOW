"""Serving and interpretation of trusted, locally trained StudyFlow models."""
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

FEATURES = ["difficulty", "days_until_deadline", "estimated_hours"]
MIN_PERSONAL = 30
MODEL_DIR = Path(__file__).resolve().parent / "ml_models"
DEMO_PATH = MODEL_DIR / "study_model.joblib"


def feature_frame(difficulty, days, estimate):
    try:
        values = np.asarray([difficulty, days, estimate], dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Enter numeric prediction inputs.") from None
    if not np.isfinite(values).all():
        raise ValueError("Prediction inputs must be finite.")
    level, horizon, duration = values
    if not level.is_integer() or not 1 <= level <= 5:
        raise ValueError("Difficulty must be an integer from 1 to 5.")
    if not horizon.is_integer() or abs(horizon) > 3652058:
        raise ValueError("Enter a valid whole number of days until the deadline.")
    if not 0 < duration <= 1_000_000:
        raise ValueError("Enter a positive estimate up to 1,000,000 hours.")
    return pd.DataFrame([values], columns=FEATURES)


def recommendations(raw, ceiling):
    """Same deployment policy for holdout evaluation and interactive predictions."""
    if not np.isfinite(raw).all():
        raise ValueError("The model produced a non-finite result.")
    return np.round(np.clip(raw, 0.1, ceiling), 1)


@dataclass
class Predictor:
    model: object
    meta: dict

    def predict(self, difficulty, days, estimate):
        frame = feature_frame(difficulty, days, estimate)
        raw = float(self.model.predict(frame)[0])
        result = float(recommendations(np.array([raw]), self.meta["ceiling"])[0])
        terms = dict(zip(FEATURES, (frame.iloc[0].to_numpy() * self.model.coef_).tolist()))
        outside = [f for f in FEATURES if not self.meta["ranges"][f][0] <= frame.at[0, f] <= self.meta["ranges"][f][1]]
        return {"hours": result, "raw": raw, "terms": terms, "outside": outside}


def deserialize(source):
    # Only application-owned artifacts are supported. Never accept uploaded pickles.
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        bundle = joblib.load(source)
    result = Predictor(bundle["model"], bundle["metadata"])
    if list(result.model.feature_names_in_) != FEATURES or result.meta["features"] != FEATURES:
        raise ValueError("Feature schema mismatch.")
    result.predict(3, 7, 4)
    return result


@lru_cache(maxsize=2)
def _load_demo(path, modified):
    return deserialize(path)


def available_model(db, sid, path=DEMO_PATH):
    personal = db.active_model(sid)
    message = ""
    if personal:
        try:
            return deserialize(BytesIO(personal["artifact"])), ""
        except Exception:
            message = "The saved personal model could not be loaded. "
    try:
        path = Path(path)
        return _load_demo(path, path.stat().st_mtime_ns), message
    except Exception:
        return None, message + "Model unavailable. Use an original or custom budget. Run python train_model.py to rebuild the demo model."
