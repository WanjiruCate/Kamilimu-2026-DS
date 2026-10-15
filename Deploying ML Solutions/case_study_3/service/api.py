"""Power-source API for the case study deployment.

Run from this folder, with the case-study package on the path:

    uvicorn api:app --host 0.0.0.0 --port 8010

Interactive docs: http://127.0.0.1:8010/docs
Browser client:   http://127.0.0.1:8010/
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import predictor

WEB_PAGE = Path(__file__).resolve().parent / "web" / "index.html"


class Reading(BaseModel):
    time: str = Field(description="Sierra Leone local time, ISO-8601, no timezone shift")
    voltage: float = Field(ge=0, le=500)
    frequency: float = Field(gt=0, le=150)


class PredictRequest(BaseModel):
    site_id: str
    room: str = ""
    available_power_sources: str
    history: list[Reading] = Field(min_length=1, max_length=64)
    site_median_frequency: float | None = Field(default=None, gt=0, le=150)
    n_sensors_in_bucket: int | None = Field(default=None, ge=1, le=500)
    model: str = "best"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "site_id": "masanga",
                    "room": "Ward",
                    "available_power_sources": "Grid w/Solar & Generator Backup",
                    "site_median_frequency": 50.02,
                    "n_sensors_in_bucket": 4,
                    "model": "best",
                    "history": [
                        {"time": "2024-03-01T10:00:00", "voltage": 230.1, "frequency": 50.02},
                        {"time": "2024-03-01T10:02:00", "voltage": 229.8, "frequency": 50.01},
                    ],
                }
            ]
        }
    }


class Prediction(BaseModel):
    active_power_source: str
    probabilities: dict[str, float]
    model: str
    adjusted_by_rules: bool
    warning: str | None = None
    scored_at: str
    n_history: int


class FeedbackRequest(PredictRequest):
    label: str = Field(pattern="^(generator|grid|solar)$")


@asynccontextmanager
async def lifespan(app):
    predictor.load_bundle()
    yield


app = FastAPI(
    title="KamiLimu power-source service",
    description=(
        "Classifies each sensor window as generator, grid, or solar. "
        "Classroom deployment for nLine case study 3. Not a grid-operations system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _score(request: PredictRequest) -> Prediction:
    result = predictor.score_history(
        site_id=request.site_id,
        available_power_sources=request.available_power_sources,
        times=[item.time for item in request.history],
        voltage=[item.voltage for item in request.history],
        frequency=[item.frequency for item in request.history],
        site_median_frequency=request.site_median_frequency,
        n_sensors_in_bucket=request.n_sensors_in_bucket,
        requested_model=request.model,
    )
    return Prediction(**result)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(WEB_PAGE.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    bundle = predictor.load_bundle()
    metrics = predictor.load_metrics()
    return {
        "status": "ok",
        "model_version": bundle["version"],
        "trained_at": bundle["trained_at"],
        "default_strategy": bundle["default_strategy"],
        "rows_in_feature_store": bundle["rows_in_feature_store"],
        "holdout_macro_f1": metrics.get("holdout", {})
        .get(bundle["default_strategy"], {})
        .get("overall", {})
        .get("macro_f1"),
    }


@app.get("/model-info")
def model_info():
    metrics = predictor.load_metrics()
    return {
        "version": metrics.get("version"),
        "trained_at": metrics.get("trained_at"),
        "sklearn_version": metrics.get("sklearn_version"),
        "default_strategy": metrics.get("default_strategy"),
        "rows_in_feature_store": metrics.get("rows_in_feature_store"),
        "expected_data_rows": metrics.get("expected_data_rows"),
        "duplicate_timestamps_collapsed": metrics.get("duplicate_timestamps_collapsed"),
        "holdout_start": metrics.get("holdout_start"),
        "pca_cumulative": metrics.get("pca_cumulative"),
        "comparison": metrics.get("comparison"),
        "cv_folds": metrics.get("cv_folds"),
        "forecast": metrics.get("forecast"),
        "permutation_importance": metrics.get("permutation_importance"),
        "shap": metrics.get("shap"),
    }


@app.get("/examples")
def examples():
    return predictor.load_examples()


@app.get("/usage")
def usage():
    path = predictor.artifact_dir() / "usage_durations.csv"
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


@app.post("/predict", response_model=Prediction)
def predict(request: PredictRequest):
    return _score(request)


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(requests: list[PredictRequest]):
    if len(requests) > 50:
        raise HTTPException(status_code=400, detail="Send at most 50 windows per batch.")
    return [_score(item) for item in requests]


@app.post("/online/update")
def online_update(request: FeedbackRequest):
    return predictor.update_online(
        label=request.label,
        available_power_sources=request.available_power_sources,
        times=[item.time for item in request.history],
        voltage=[item.voltage for item in request.history],
        frequency=[item.frequency for item in request.history],
        site_median_frequency=request.site_median_frequency,
        n_sensors_in_bucket=request.n_sensors_in_bucket,
    )


@app.post("/online/reset")
def online_reset():
    predictor.reset_online()
    return {"status": "reset"}
