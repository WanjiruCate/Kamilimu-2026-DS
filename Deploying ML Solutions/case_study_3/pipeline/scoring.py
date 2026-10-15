"""Shared scoring for the holdout pass and the API.

Linear models see standardised features, clipped to [-8, 8] so a 99 Hz glitch
cannot dominate the SGD update. Tree models see the raw feature matrix.
The clip is part of the model: inference applies it too.
"""

from __future__ import annotations

import numpy as np

from pipeline.config import FEATURE_COLUMNS
from pipeline.rules import apply_hybrid, classify_rules

CLIP = 8.0
_MAD = FEATURE_COLUMNS.index("trail_mad_freq")


def transform_linear(scaler, features: np.ndarray, pca=None) -> np.ndarray:
    scaled = scaler.transform(np.asarray(features, dtype=np.float64))
    np.clip(scaled, -CLIP, CLIP, out=scaled)
    if pca is not None:
        scaled = pca.transform(scaled)
    return scaled


def class_probabilities(bundle: dict, strategy: str, features: np.ndarray) -> np.ndarray:
    """Probability rows in CLASSES order. `strategy` is a base model name."""
    features = np.asarray(features, dtype=np.float64)
    if strategy in {"sgd", "online"}:
        model = bundle["online_sgd"] if strategy == "online" and bundle.get("online_sgd") is not None else bundle["sgd"]
        return model.predict_proba(transform_linear(bundle["scaler"], features))
    if strategy == "sgd_pca":
        return bundle["sgd_pca"].predict_proba(
            transform_linear(bundle["scaler"], features, bundle["pca"])
        )
    if strategy == "hgb":
        return bundle["hgb"].predict_proba(features)
    if strategy == "rf":
        return bundle["rf"].predict_proba(features)
    raise KeyError(strategy)


def predict_labels(
    bundle: dict,
    strategy: str,
    features: np.ndarray,
    site_median: np.ndarray,
    is_pcmh,
    is_ola,
    mad: np.ndarray | None = None,
) -> np.ndarray:
    features = np.asarray(features)
    if mad is None:
        mad = features[:, _MAD]
    if strategy in {"rules", "rules_trailing", "rules_centered"}:
        return classify_rules(features, mad, site_median, is_pcmh, is_ola)
    base = strategy.split("hybrid_", 1)[1] if strategy.startswith("hybrid_") else strategy
    if base == "online":
        probabilities = class_probabilities(bundle, "online", features)
    else:
        probabilities = class_probabilities(bundle, base, features)
    labels = probabilities.argmax(axis=1).astype(np.int8)
    if strategy.startswith("hybrid_"):
        labels = apply_hybrid(labels, features, mad, site_median, is_pcmh, is_ola)
    return labels
