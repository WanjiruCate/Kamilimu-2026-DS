"""sktime forecasting workflow on one sensor's hourly mean frequency.

This follows the sktime forecasting tutorial: a temporal train/test split,
a forecasting horizon, a seasonal naive benchmark, a reduced tabular model,
and an expanding-window backtest. Frequency is the series the rule system
itself is built on, so a forecast is an early warning that the classifier
is about to see a generator-like deviation from 50 Hz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import mean_absolute_error
from sktime.split import ExpandingWindowSplitter

from pipeline.config import RAW_DIR
from pipeline.features import to_unix_seconds


def hourly_frequency(respondent_id: str) -> pd.Series:
    frame = pd.read_parquet(RAW_DIR / f"{respondent_id}.parquet", columns=["time", "frequency"])
    time_s = to_unix_seconds(frame["time"])
    index = pd.to_datetime(time_s, unit="s")
    series = pd.Series(frame["frequency"].to_numpy(np.float64), index=index).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    hourly = series.resample("h").mean()
    filled = hourly.interpolate(limit=3)
    mask = filled.notna().to_numpy()
    best_start, best_len, run = 0, 0, None
    for i, flag in enumerate(np.r_[mask, False]):
        if flag and run is None:
            run = i
        elif not flag and run is not None:
            if i - run > best_len:
                best_start, best_len = run, i - run
            run = None
    window = filled.iloc[best_start : best_start + best_len].copy()
    window.index.freq = pd.tseries.frequencies.to_offset("h")
    window.name = "frequency_hz"
    return window


def _fit_predict(model, y_train: pd.Series, horizon: ForecastingHorizon) -> pd.Series:
    model.fit(y_train)
    return model.predict(horizon)


def run_frequency_forecast(respondent_id: str, room: str, site: str) -> dict:
    y = hourly_frequency(respondent_id)
    if len(y) < 24 * 40:
        return {
            "respondent_id": respondent_id,
            "room": room,
            "site_id": site,
            "skipped": True,
            "reason": f"only {len(y)} hourly points",
        }

    test_size = 24 * 14
    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]
    horizon = ForecastingHorizon(np.arange(1, test_size + 1), is_relative=True)

    naive = NaiveForecaster(strategy="last", sp=24)
    reduced = make_reduction(Ridge(alpha=1.0), strategy="recursive", window_length=48)
    pred_naive = _fit_predict(naive, y_train, horizon)
    pred_ridge = _fit_predict(reduced, y_train, horizon)
    pred_naive.index = y_test.index
    pred_ridge.index = y_test.index

    mae_naive = float(mean_absolute_error(y_test, pred_naive))
    mae_ridge = float(mean_absolute_error(y_test, pred_ridge))

    initial = 24 * 28
    step = 24 * 21
    while len(y) - initial - 24 > 8 * step:
        step += 24 * 7
    splitter = ExpandingWindowSplitter(fh=list(range(1, 25)), initial_window=initial, step_length=step)
    cv_naive = []
    cv_ridge = []
    for train_idx, test_idx in splitter.split(y):
        fold_train = y.iloc[train_idx]
        fold_test = y.iloc[test_idx]
        fold_fh = ForecastingHorizon(np.arange(1, len(fold_test) + 1), is_relative=True)
        cv_naive.append(
            float(mean_absolute_error(fold_test, _fit_predict(NaiveForecaster(strategy="last", sp=24), fold_train, fold_fh)))
        )
        cv_ridge.append(
            float(
                mean_absolute_error(
                    fold_test,
                    _fit_predict(
                        make_reduction(Ridge(alpha=1.0), strategy="recursive", window_length=48),
                        fold_train,
                        fold_fh,
                    ),
                )
            )
        )

    return {
        "respondent_id": respondent_id,
        "room": room,
        "site_id": site,
        "skipped": False,
        "n_hours": int(len(y)),
        "series_start": str(y.index[0]),
        "series_end": str(y.index[-1]),
        "holdout_hours": test_size,
        "mae_seasonal_naive_sp24": round(mae_naive, 4),
        "mae_ridge_reduction": round(mae_ridge, 4),
        "skill_vs_naive": round(1 - mae_ridge / mae_naive, 4) if mae_naive else None,
        "expanding_window_folds": len(cv_naive),
        "cv_mae_seasonal_naive": round(float(np.mean(cv_naive)), 4),
        "cv_mae_ridge_reduction": round(float(np.mean(cv_ridge)), 4),
        "y_test": y_test,
        "pred_naive": pred_naive,
        "pred_ridge": pred_ridge,
        "series": y,
    }
