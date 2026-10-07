"""Train the house-price pipeline and save it, with metadata, for serving.

Run from this folder:

    python train.py

This writes ``model/house_price_model.joblib`` and ``model/metadata.json``.
The API (``api.py``) loads both files at start-up. Re-run this script whenever
the data, features, or model change, and bump ``MODEL_VERSION``.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR.parent / "data" / "train.csv"
MODEL_DIR = APP_DIR / "model"
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

MODEL_VERSION = "1.0.0"
TARGET = "SalePrice"
FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
    "FullBath",
]


def build_pipeline():
    # kNN measures distances, so features must be on the same scale. Keeping
    # the scaler inside the pipeline means it is saved, and applied, with the model.
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=10)),
        ]
    )


def train(data_path=DATA_PATH):
    data = pd.read_csv(data_path)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline().fit(X_train, y_train)
    valid_mae = mean_absolute_error(y_valid, model.predict(X_valid))
    baseline_mae = mean_absolute_error(y_valid, [y_train.median()] * len(y_valid))
    linear = LinearRegression().fit(X_train, y_train)
    linear_mae = mean_absolute_error(y_valid, linear.predict(X_valid))

    # Refit on all labelled rows once the evaluation is recorded.
    model = build_pipeline().fit(X, y)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "model": "StandardScaler + KNeighborsRegressor(n_neighbors=10)",
        "training_rows": len(data),
        "target": TARGET,
        "features": FEATURES,
        "numeric_ranges": {
            col: [float(X[col].min()), float(X[col].max())] for col in FEATURES
        },
        "validation": {
            "split": "random 80/20, random_state=42",
            "mae": round(valid_mae, 2),
            "linear_regression_mae": round(linear_mae, 2),
            "median_baseline_mae": round(baseline_mae, 2),
        },
        "intended_use": (
            "Classroom practice only. Historical Ames, Iowa sales; "
            "not a valuation service."
        ),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return model, metadata


if __name__ == "__main__":
    _, meta = train()
    print(f"Saved {MODEL_PATH.name} (version {meta['model_version']})")
    print("Validation:", meta["validation"])
