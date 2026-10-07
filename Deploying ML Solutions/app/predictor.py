"""Load the saved model once and turn validated inputs into predictions.

Both the REST API and the agent's pricing tool use this module, so the model,
input rules, and warnings are defined in exactly one place.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


@lru_cache(maxsize=1)
def load_metadata():
    return json.loads(METADATA_PATH.read_text())


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python train.py` first."
        )
    return joblib.load(MODEL_PATH)


class HouseFeatures(BaseModel):
    """The inputs a caller must send. Pydantic rejects wrong types and
    impossible values before they ever reach the model."""

    OverallQual: int = Field(ge=1, le=10, description="Overall quality, 1-10")
    GrLivArea: float = Field(gt=0, le=10_000, description="Above-ground living area (sq ft)")
    GarageCars: int = Field(ge=0, le=5, description="Garage capacity in cars")
    TotalBsmtSF: float = Field(ge=0, le=10_000, description="Basement area (sq ft)")
    YearBuilt: int = Field(ge=1850, le=2030, description="Original construction year")
    FullBath: int = Field(ge=0, le=5, description="Full bathrooms above ground")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "OverallQual": 7,
                    "GrLivArea": 1710,
                    "GarageCars": 2,
                    "TotalBsmtSF": 856,
                    "YearBuilt": 2003,
                    "FullBath": 2,
                }
            ]
        }
    }


class Prediction(BaseModel):
    predicted_price: float
    model_version: str
    warnings: list[str]


def range_warnings(features: HouseFeatures):
    """Valid but unusual inputs: kNN can only average the prices of houses it
    has seen, so estimates outside the training range are unreliable."""
    warnings = []
    for col, (low, high) in load_metadata()["numeric_ranges"].items():
        value = getattr(features, col)
        if not low <= value <= high:
            warnings.append(
                f"{col}={value} is outside the training range [{low:g}, {high:g}]; "
                "treat this estimate with extra caution."
            )
    return warnings


def predict(features: HouseFeatures) -> Prediction:
    row = pd.DataFrame([features.model_dump()])[load_metadata()["features"]]
    price = float(load_model().predict(row)[0])
    return Prediction(
        predicted_price=round(price, -2),
        model_version=load_metadata()["model_version"],
        warnings=range_warnings(features),
    )
