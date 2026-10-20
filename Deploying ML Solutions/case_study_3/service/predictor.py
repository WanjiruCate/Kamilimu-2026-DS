"""Load the saved case-study models and score one sensor window.

Training and this module both call `engineer_sensor`, so a window scored here
uses the same trailing features as the holdout evaluation.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np

from pipeline.config import CASE_DIR, CLASSES, FEATURE_COLUMNS, OLA_SITE, PCMH_SITE
from pipeline.features import bucket_ids, engineer_sensor, isoformat_utc, source_flags, to_unix_seconds
from pipeline.scoring import class_probabilities, predict_labels, transform_linear
from pipeline.sequence_models import SequenceClassifier, load_sequence

_LOCK = threading.Lock()


def artifact_dir() -> Path:
    return Path(os.environ.get("ARTIFACT_DIR", CASE_DIR / "artifacts"))


def _strategy_name(requested: str, bundle: dict) -> str:
    default = bundle["default_strategy"]
    if requested == "best":
        return default
    if requested == "rules":
        return "rules_trailing"
    if requested == "hybrid":
        return default if str(default).startswith("hybrid_") else "hybrid_hgb"
    if requested == "online":
        return "online"
    return requested


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    path = artifact_dir() / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run python -m pipeline.train first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    path = artifact_dir() / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_examples() -> list[dict]:
    path = artifact_dir() / "examples.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _load_online() -> dict:
    path = artifact_dir() / "online_model.joblib"
    init = artifact_dir() / "online_model_init.joblib"
    if not path.exists():
        if not init.exists():
            raise FileNotFoundError("Online model files are missing.")
        path = init
    return joblib.load(path)


def load_online():
    # Not cached: classroom feedback rewrites the file and the in-memory copy.
    if not hasattr(load_online, "state"):
        with _LOCK:
            if not hasattr(load_online, "state"):
                load_online.state = _load_online()
    return load_online.state


def reset_online() -> None:
    init = joblib.load(artifact_dir() / "online_model_init.joblib")
    with _LOCK:
        load_online.state = init
        joblib.dump(init, artifact_dir() / "online_model.joblib")


def _sequence(kind: str) -> SequenceClassifier:
    return load_sequence(artifact_dir() / f"{kind}_weights.npz")


def features_from_history(
    times: list[str],
    voltage: list[float],
    frequency: list[float],
    available_power_sources: str,
    site_median_frequency: float | None,
    n_sensors_in_bucket: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrix for the whole history. Callers score the last row."""
    time_s = to_unix_seconds(times)
    order = np.argsort(time_s, kind="mergesort")
    time_s = time_s[order]
    voltage_arr = np.asarray(voltage, dtype=np.float64)[order]
    frequency_arr = np.asarray(frequency, dtype=np.float64)[order]
    flag = source_flags(available_power_sources)
    n = len(time_s)
    has_grid = np.full(n, flag[0])
    has_solar = np.full(n, flag[1])
    has_generator = np.full(n, flag[2])
    if site_median_frequency is None:
        site_median_frequency = float(np.median(frequency_arr))
        sensors = 1 if n_sensors_in_bucket is None else int(n_sensors_in_bucket)
    else:
        sensors = int(n_sensors_in_bucket or 2)
    last_bucket = int(bucket_ids(time_s[-1:])[0])
    features, _centered, site_median = engineer_sensor(
        time_s,
        voltage_arr,
        frequency_arr,
        has_grid,
        has_solar,
        has_generator,
        np.array([last_bucket], dtype=np.int64),
        np.array([site_median_frequency], dtype=np.float32),
        np.array([sensors], dtype=np.float32),
    )
    return features, site_median, time_s


def score_history(
    site_id: str,
    available_power_sources: str,
    times: list[str],
    voltage: list[float],
    frequency: list[float],
    site_median_frequency: float | None,
    n_sensors_in_bucket: int | None,
    requested_model: str,
) -> dict:
    features, site_median, time_s = features_from_history(
        times,
        voltage,
        frequency,
        available_power_sources,
        site_median_frequency,
        n_sensors_in_bucket,
    )
    bundle = load_bundle()
    strategy = _strategy_name(requested_model, bundle)
    warning = None
    is_pcmh = site_id == PCMH_SITE
    is_ola = site_id == OLA_SITE
    row = features[-1:]
    median = site_median[-1:]

    if strategy in {"lstm", "rnn"}:
        model = _sequence(strategy)
        length = 16
        if len(features) >= length:
            window = features[-length:, [0, 1]]
        else:
            pad = np.repeat(features[:1, [0, 1]], length - len(features), axis=0)
            window = np.vstack([pad, features[:, [0, 1]]])
            warning = (
                f"{strategy.upper()} was trained on 16-point windows; "
                "the history was padded by repeating the earliest reading."
            )
        probabilities = model.predict_proba(window[None, :, :])[0]
        label_id = int(np.argmax(probabilities))
        adjusted = False
    else:
        label_id = int(
            predict_labels(bundle, "online" if strategy == "online" else strategy, row, median, is_pcmh, is_ola)[0]
        )
        if strategy == "online":
            online = load_online()
            scoring_bundle = dict(bundle)
            scoring_bundle["online_sgd"] = online["sgd"]
            scoring_bundle["scaler"] = online["scaler"]
            probabilities = class_probabilities(scoring_bundle, "online", row)[0]
            # predict_labels above used the frozen SGD, not the updated one. Recompute.
            label_id = int(np.argmax(probabilities))
        elif strategy in {"rules", "rules_trailing", "rules_centered"}:
            probabilities = np.zeros(len(CLASSES))
            probabilities[label_id] = 1.0
        else:
            base = strategy.split("hybrid_", 1)[1] if strategy.startswith("hybrid_") else strategy
            probabilities = class_probabilities(bundle, base, row)[0]
        base_label = int(np.argmax(probabilities)) if strategy not in {"rules", "rules_trailing", "rules_centered"} else label_id
        adjusted = bool(strategy.startswith("hybrid_") and label_id != base_label)

    return {
        "active_power_source": CLASSES[label_id],
        "probabilities": {name: round(float(probabilities[i]), 4) for i, name in enumerate(CLASSES)},
        "model": strategy,
        "adjusted_by_rules": adjusted if strategy not in {"lstm", "rnn"} else False,
        "warning": warning,
        "scored_at": isoformat_utc(int(time_s[-1])),
        "n_history": len(times),
    }


def update_online(
    label: str,
    available_power_sources: str,
    times: list[str],
    voltage: list[float],
    frequency: list[float],
    site_median_frequency: float | None,
    n_sensors_in_bucket: int | None,
) -> dict:
    features, _median, _time_s = features_from_history(
        times,
        voltage,
        frequency,
        available_power_sources,
        site_median_frequency,
        n_sensors_in_bucket,
    )
    online = load_online()
    label_id = CLASSES.index(label)
    scaled = transform_linear(online["scaler"], features[-1:])
    with _LOCK:
        online["sgd"].partial_fit(scaled, np.array([label_id]), classes=np.arange(len(CLASSES)))
        joblib.dump(online, artifact_dir() / "online_model.joblib")
    return {"status": "updated", "label": label, "n_features": len(FEATURE_COLUMNS)}
