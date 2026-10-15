"""The written rule cascade, and a hybrid that keeps those constraints around an ML model.

Single-source rooms are labelled from the installation survey. Frequency rules
are applied only where more than one source is installed, which is the case
the note says the thresholds are for. Smoothing then applies the two
site-specific corrections that were written down:

- Princess Christian (PCMH): do not call generator while the site still looks
  like it is on the grid.
- Ola During: while the grid is off, a solar-band frequency is solar.

"Grid off" is not given a numeric cutoff in the note. We treat the site median
as off-grid when it sits more than 0.5 Hz from 50 Hz and at least two sensors
reported in that two-minute bucket. That constant is an explicit assumption.
"""

from __future__ import annotations

import numpy as np

from pipeline.config import (
    FEATURE_COLUMNS,
    GRID_DEV_HZ,
    GRID_OFF_DEV_HZ,
    LABEL_TO_ID,
    OLA_SITE,
    PCMH_SITE,
    SOLAR_HZ,
    SOLAR_MAD,
)

_COL = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
GENERATOR = LABEL_TO_ID["generator"]
GRID = LABEL_TO_ID["grid"]
SOLAR = LABEL_TO_ID["solar"]


def _as_bool(values, n: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        return np.full(n, bool(array))
    return array.astype(bool)


def classify_rules(
    features: np.ndarray,
    mad: np.ndarray,
    site_median: np.ndarray,
    is_pcmh,
    is_ola,
) -> np.ndarray:
    """Return class ids. `mad` is centred for the baseline and trailing for live use."""
    n = len(features)
    frequency = features[:, _COL["frequency"]]
    has_grid = features[:, _COL["has_grid"]] > 0.5
    has_solar = features[:, _COL["has_solar"]] > 0.5
    has_generator = features[:, _COL["has_generator"]] > 0.5
    n_sensors = features[:, _COL["n_sensors_in_bucket"]]
    freq_dev = features[:, _COL["freq_dev_site_median"]]
    mad = np.asarray(mad)
    site_median = np.asarray(site_median)

    single_grid = has_grid & ~has_solar & ~has_generator
    single_solar = has_solar & ~has_grid & ~has_generator
    single_generator = has_generator & ~has_grid & ~has_solar
    multi = ~(single_grid | single_solar | single_generator)

    solar_band = (frequency >= SOLAR_HZ[0]) & (frequency <= SOLAR_HZ[1])
    grid = multi & has_grid & (n_sensors >= 2) & (freq_dev <= GRID_DEV_HZ)
    solar = multi & ~grid & has_solar & solar_band & (mad <= SOLAR_MAD)

    prediction = np.full(n, GENERATOR, dtype=np.int8)
    prediction[solar] = SOLAR
    prediction[grid] = GRID
    prediction[single_grid] = GRID
    prediction[single_solar] = SOLAR
    prediction[single_generator] = GENERATOR

    grid_off = (n_sensors >= 2) & (np.abs(site_median - 50.0) > GRID_OFF_DEV_HZ)
    is_pcmh = _as_bool(is_pcmh, n)
    is_ola = _as_bool(is_ola, n)

    # PCMH: a generator call is only kept while the grid looks off.
    pcmh_block = is_pcmh & multi & (prediction == GENERATOR) & ~grid_off
    to_solar = pcmh_block & has_solar & solar_band
    to_grid = pcmh_block & ~to_solar & has_grid
    prediction[to_solar] = SOLAR
    prediction[to_grid] = GRID

    ola_solar = is_ola & multi & grid_off & has_solar & solar_band
    prediction[ola_solar] = SOLAR
    return prediction


def apply_hybrid(
    ml_prediction: np.ndarray,
    features: np.ndarray,
    mad: np.ndarray,
    site_median: np.ndarray,
    is_pcmh,
    is_ola,
) -> np.ndarray:
    """Trust the survey on single-source rooms, then apply the same smoothing."""
    n = len(features)
    has_grid = features[:, _COL["has_grid"]] > 0.5
    has_solar = features[:, _COL["has_solar"]] > 0.5
    has_generator = features[:, _COL["has_generator"]] > 0.5
    frequency = features[:, _COL["frequency"]]
    n_sensors = features[:, _COL["n_sensors_in_bucket"]]
    prediction = np.asarray(ml_prediction, dtype=np.int8).copy()

    single_grid = has_grid & ~has_solar & ~has_generator
    single_solar = has_solar & ~has_grid & ~has_generator
    single_generator = has_generator & ~has_grid & ~has_solar
    multi = ~(single_grid | single_solar | single_generator)
    prediction[single_grid] = GRID
    prediction[single_solar] = SOLAR
    prediction[single_generator] = GENERATOR

    solar_band = (frequency >= SOLAR_HZ[0]) & (frequency <= SOLAR_HZ[1])
    grid_off = (n_sensors >= 2) & (np.abs(np.asarray(site_median) - 50.0) > GRID_OFF_DEV_HZ)
    is_pcmh = _as_bool(is_pcmh, n)
    is_ola = _as_bool(is_ola, n)

    pcmh_block = is_pcmh & multi & (prediction == GENERATOR) & ~grid_off
    to_solar = pcmh_block & has_solar & solar_band
    to_grid = pcmh_block & ~to_solar & has_grid
    prediction[to_solar] = SOLAR
    prediction[to_grid] = GRID

    ola_solar = is_ola & multi & grid_off & has_solar & solar_band
    prediction[ola_solar] = SOLAR
    return prediction
