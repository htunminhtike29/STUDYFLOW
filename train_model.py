"""Reproducible demo training and explicitly requested personal training."""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml_engine import FEATURES, MIN_PERSONAL, MODEL_DIR, Predictor, recommendations

SEED = 104
DATA = Path(__file__).resolve().parent / "data" / "study_history.csv"
TARGET = "actual_hours"


def synthetic_data():
    rng = np.random.default_rng(SEED)
    n = 400
    subjects = rng.choice(["Calculus", "Networks", "Databases", "Physics", "Algorithms",
                           "Design", "Economics", "Writing", "Psychology", "Chemistry"], n)
    difficulty = rng.integers(1, 6, n)
    days = rng.integers(0, 36, n)
    estimate = np.round(rng.uniform(0.5, 16, n), 2)
    noise = rng.normal(0, 1.1 + 0.09 * estimate, n)
    actual = np.round(np.maximum(0.1, 0.5 + 0.7*difficulty - 0.035*days + 0.96*estimate + noise), 2)
    return pd.DataFrame({"subject": subjects, "difficulty": difficulty, "days_until_deadline": days,
                         "estimated_hours": estimate, TARGET: actual, "source": "synthetic"})


def clean(frame, *, personal=False):
    needed = FEATURES + [TARGET]
    if personal:
        needed += ["task_id", "completed_at", "eligible"]
    if not set(needed).issubset(frame.columns):
        raise ValueError("Training data is missing required columns.")
    data = frame.copy()
    if personal:
        data = data[data.eligible == 1].copy()
        data["completed_at"] = pd.to_datetime(data.completed_at, errors="coerce", utc=True)
        data = data.dropna(subset=["task_id", "completed_at"])
        if data.task_id.duplicated().any():
            raise ValueError("Duplicate personal task IDs in training data.")
        data = data.sort_values(["completed_at", "task_id"])
    numeric = FEATURES + [TARGET]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(data[numeric]).all(axis=1) & data.difficulty.isin(range(1, 6))
             & (data.days_until_deadline % 1 == 0) & (data.days_until_deadline.abs() <= 3652058)
             & (data.estimated_hours > 0) & (data.estimated_hours <= 1_000_000) & (data.actual_hours > 0))
    data = data.loc[valid].reset_index(drop=True)
    if len(data) < MIN_PERSONAL:
        raise ValueError(f"At least {MIN_PERSONAL} valid completed records are required.")
    return data


def metrics(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    return {"MAE": float(mean_absolute_error(actual, predicted)), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R²": float(r2_score(actual, predicted)) if np.ptp(actual) else None}


def train(frame, *, personal=False):
    data = clean(frame, personal=personal)
    if personal:
        cut = int(0.8 * len(data))
        training, test = data.iloc[:cut], data.iloc[cut:]
    else:
        training, test = train_test_split(data, test_size=0.2, random_state=SEED)
    model = LinearRegression().fit(training[FEATURES], training[TARGET])
    raw = model.predict(test[FEATURES])
    ceiling = min(1_000_000, max(0.1, float(np.ceil(training[TARGET].max()))))
    predicted = recommendations(raw, ceiling)
    actual = test[TARGET].to_numpy()
    when = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    source = "Personal completed tasks" if personal else "Synthetic demonstration"
    version = ("personal" if personal else "demo") + "-" + hashlib.sha256(
        (when + data[FEATURES+[TARGET]].to_csv(index=False)).encode()).hexdigest()[:12]
    meta = {"algorithm": "Multiple Linear Regression", "version": version, "trained_at": when,
            "source": source, "features": FEATURES, "dataset_size": len(data), "dropped": len(frame)-len(data),
            "train_size": len(training), "test_size": len(test), "sklearn_version": sklearn.__version__,
            "split": "Chronological by completion" if personal else f"Random, seed {SEED}",
            "intercept": float(model.intercept_), "coefficients": dict(zip(FEATURES, model.coef_.tolist())),
            "ranges": {f: [float(training[f].min()), float(training[f].max())] for f in FEATURES},
            "ceiling": ceiling, "regression": metrics(actual, predicted),
            "baseline": metrics(actual, test.estimated_hours.to_numpy()),
            "holdout": {"actual": actual.tolist(), "prediction": predicted.tolist(), "raw": raw.tolist(),
                        "baseline": test.estimated_hours.tolist()}, "training_hours": training[TARGET].tolist()}
    return Predictor(model, meta)


def serialize(predictor):
    buffer = BytesIO()
    joblib.dump({"model": predictor.model, "metadata": predictor.meta}, buffer, compress=3)
    return buffer.getvalue()


def atomic_write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(contents)
        temp_path = Path(tmp.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def train_demo(data_path=DATA, output=MODEL_DIR):
    if not data_path.exists():
        atomic_write(data_path, synthetic_data().to_csv(index=False).encode())
    frame = pd.read_csv(data_path)
    if "source" not in frame or not frame.source.eq("synthetic").all():
        raise ValueError("The shared demo accepts only the labeled synthetic dataset.")
    predictor = train(frame)
    atomic_write(output / "study_model.joblib", serialize(predictor))
    atomic_write(output / "model_metadata.json", json.dumps(predictor.meta, indent=2, allow_nan=False).encode())
    return predictor


def train_personal(db, sid):
    records = db.history(sid)
    if sum(r["eligible"] for r in records) < MIN_PERSONAL:
        raise ValueError(f"Complete {MIN_PERSONAL} eligible tasks before training a personal model.")
    candidate = train(pd.DataFrame(records), personal=True)
    meta = candidate.meta
    activate = all(meta["regression"][m] < meta["baseline"][m] for m in ("MAE", "RMSE"))
    db.save_model(sid, meta, serialize(candidate), activate)
    return meta, activate


if __name__ == "__main__":
    report = train_demo().meta
    print(report["version"], report["source"], "train/test", report["train_size"], report["test_size"])
    print(json.dumps({k: report[k] for k in ("regression", "baseline")}, indent=2))
