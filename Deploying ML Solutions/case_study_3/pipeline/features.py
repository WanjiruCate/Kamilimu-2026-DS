"""Causal features for one sensor, plus the site-level grid reference.

The nLine rule note uses a centred 15-minute window (three readings before
and three after). A live API does not have the future readings, so the model
features are a trailing window of the same length. The centred mean absolute
deviation is still computed and stored, and is used only for the rule
baseline that tries to reproduce the written method.

"MAD" in the case note is the mean absolute deviation from the mean, not the
median absolute deviation.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from numba import njit

from pipeline.config import (
    BUCKET_SECONDS,
    CENTER_RADIUS,
    FEATURE_COLUMNS,
    LABEL_TO_ID,
    LOW_VOLTAGE,
    MAX_GAP_SECONDS,
    NOMINAL_HZ,
    NOMINAL_VOLT,
    TRAIL_WINDOW,
)

# Unix epoch was a Thursday. pandas weekday() is Monday=0, so Thursday=3.
_EPOCH_WEEKDAY = 3


@njit(cache=True)
def _trailing_stats(freq, volt, time_s, window, max_gap):
    n = freq.shape[0]
    mean = np.empty(n, np.float64)
    std = np.empty(n, np.float64)
    mad = np.empty(n, np.float64)
    rmin = np.empty(n, np.float64)
    rmax = np.empty(n, np.float64)
    vstd = np.empty(n, np.float64)
    for i in range(n):
        start = i
        taken = 1
        while start > 0 and taken < window and (time_s[start] - time_s[start - 1]) <= max_gap:
            start -= 1
            taken += 1
        count = 0
        total = 0.0
        vtotal = 0.0
        for j in range(start, i + 1):
            total += freq[j]
            vtotal += volt[j]
            count += 1
        mean_f = total / count
        mean_v = vtotal / count
        var_f = 0.0
        abs_f = 0.0
        var_v = 0.0
        lo = freq[start]
        hi = freq[start]
        for j in range(start, i + 1):
            df = freq[j] - mean_f
            var_f += df * df
            abs_f += abs(df)
            dv = volt[j] - mean_v
            var_v += dv * dv
            if freq[j] < lo:
                lo = freq[j]
            if freq[j] > hi:
                hi = freq[j]
        mean[i] = mean_f
        std[i] = (var_f / count) ** 0.5
        mad[i] = abs_f / count
        rmin[i] = lo
        rmax[i] = hi
        vstd[i] = (var_v / count) ** 0.5
    return mean, std, mad, rmin, rmax, vstd


@njit(cache=True)
def _centered_mad(freq, time_s, radius, max_gap):
    n = freq.shape[0]
    out = np.empty(n, np.float64)
    for i in range(n):
        left = i
        got = 0
        while left > 0 and got < radius and (time_s[left] - time_s[left - 1]) <= max_gap:
            left -= 1
            got += 1
        right = i
        got = 0
        while right + 1 < n and got < radius and (time_s[right + 1] - time_s[right]) <= max_gap:
            right += 1
            got += 1
        total = 0.0
        count = 0
        for j in range(left, right + 1):
            total += freq[j]
            count += 1
        mean_f = total / count
        abs_f = 0.0
        for j in range(left, right + 1):
            abs_f += abs(freq[j] - mean_f)
        out[i] = abs_f / count
    return out


def source_flags(available: str) -> tuple[float, float, float, float]:
    """Map a survey string such as 'Grid w/Solar Backup' to indicator flags."""
    text = available.lower()
    has_grid = 1.0 if "grid" in text else 0.0
    has_solar = 1.0 if "solar" in text else 0.0
    has_generator = 1.0 if "generator" in text else 0.0
    return has_grid, has_solar, has_generator, has_grid + has_solar + has_generator


def encode_label(values: np.ndarray) -> np.ndarray:
    encoded = np.full(len(values), -1, dtype=np.int8)
    as_str = np.asarray(values).astype(str)
    for name, index in LABEL_TO_ID.items():
        encoded[as_str == name] = index
    return encoded


def to_unix_seconds(times) -> np.ndarray:
    """Naive Sierra Leone clock times -> integer seconds, interpreted as UTC."""
    dt = pd.to_datetime(times)
    nanoseconds = dt.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return (nanoseconds // 1_000_000_000).astype(np.int64)


def bucket_ids(time_s: np.ndarray) -> np.ndarray:
    return (time_s // BUCKET_SECONDS) * BUCKET_SECONDS


def collapse_duplicate_times(
    time_s: np.ndarray,
    voltage: np.ndarray,
    frequency: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Average same-timestamp readings. Disagreeing labels become unknown (-1)."""
    order = np.argsort(time_s, kind="mergesort")
    time_s = time_s[order]
    voltage = voltage[order]
    frequency = frequency[order]
    labels = labels[order]
    if len(time_s) == 0 or np.all(time_s[1:] != time_s[:-1]):
        return time_s, voltage, frequency, labels, 0
    change = np.flatnonzero(np.diff(time_s)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(time_s)]
    n = len(starts)
    out_t = time_s[starts]
    out_v = np.empty(n, np.float32)
    out_f = np.empty(n, np.float32)
    out_y = np.empty(n, np.int8)
    for i, (start, end) in enumerate(zip(starts, ends)):
        out_v[i] = voltage[start:end].mean()
        out_f[i] = frequency[start:end].mean()
        group = labels[start:end]
        out_y[i] = group[0] if np.all(group == group[0]) else np.int8(-1)
    return out_t, out_v, out_f, out_y, int(len(time_s) - n)


def _lookup_sorted(buckets: np.ndarray, keys: np.ndarray, medians: np.ndarray, counts: np.ndarray):
    if len(keys) == 0:
        return np.full(len(buckets), np.nan, np.float32), np.ones(len(buckets), np.float32)
    index = np.searchsorted(keys, buckets)
    index = np.clip(index, 0, len(keys) - 1)
    hit = keys[index] == buckets
    site_median = np.where(hit, medians[index], np.nan).astype(np.float32)
    n_sensors = np.where(hit, counts[index], 1).astype(np.float32)
    return site_median, n_sensors


def engineer_sensor(
    time_s: np.ndarray,
    voltage: np.ndarray,
    frequency: np.ndarray,
    has_grid: np.ndarray,
    has_solar: np.ndarray,
    has_generator: np.ndarray,
    bucket_keys: np.ndarray,
    bucket_medians: np.ndarray,
    bucket_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, centered_mad, site_median) aligned to the input rows.

    Rows must already be sorted by time and free of duplicate timestamps.
    """
    time_s = np.asarray(time_s, dtype=np.int64)
    voltage = np.asarray(voltage, dtype=np.float64)
    frequency = np.asarray(frequency, dtype=np.float64)
    n = len(time_s)
    if n == 0:
        empty = np.empty((0, len(FEATURE_COLUMNS)), np.float32)
        return empty, np.empty(0, np.float32), np.empty(0, np.float32)

    mean, std, mad, rmin, rmax, vstd = _trailing_stats(
        frequency, voltage, time_s, TRAIL_WINDOW, MAX_GAP_SECONDS
    )
    centered = _centered_mad(frequency, time_s, CENTER_RADIUS, MAX_GAP_SECONDS)

    freq_diff = np.zeros(n, np.float64)
    volt_diff = np.zeros(n, np.float64)
    gap_minutes = np.zeros(n, np.float64)
    if n > 1:
        delta = np.diff(time_s).astype(np.float64)
        gap_minutes[1:] = np.minimum(delta / 60.0, 180.0)
        close = delta <= MAX_GAP_SECONDS
        freq_diff[1:] = np.where(close, np.diff(frequency), 0.0)
        volt_diff[1:] = np.where(close, np.diff(voltage), 0.0)

    buckets = bucket_ids(time_s)
    site_median, n_sensors = _lookup_sorted(buckets, bucket_keys, bucket_medians, bucket_counts)
    own = site_median != site_median  # NaN
    site_median = np.where(own, frequency, site_median).astype(np.float64)
    freq_dev = np.abs(frequency - site_median)

    hour = (time_s // 3600) % 24
    # Monday=0 ... Sunday=6, matching pandas .weekday() on the UTC clock time.
    dow = (time_s // 86400 + _EPOCH_WEEKDAY) % 7
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    n_sources = has_grid + has_solar + has_generator

    columns = {
        "voltage": voltage,
        "frequency": frequency,
        "abs_freq_dev_50": np.abs(frequency - NOMINAL_HZ),
        "abs_volt_dev_230": np.abs(voltage - NOMINAL_VOLT),
        "freq_diff_1": freq_diff,
        "volt_diff_1": volt_diff,
        "trail_mean_freq": mean,
        "trail_std_freq": std,
        "trail_mad_freq": mad,
        "trail_min_freq": rmin,
        "trail_max_freq": rmax,
        "trail_std_volt": vstd,
        "freq_dev_site_median": freq_dev,
        "n_sensors_in_bucket": n_sensors,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "has_grid": has_grid,
        "has_solar": has_solar,
        "has_generator": has_generator,
        "n_sources": n_sources,
        "gap_minutes": gap_minutes,
        "low_voltage": (voltage < LOW_VOLTAGE).astype(np.float64),
    }
    matrix = np.column_stack([columns[name] for name in FEATURE_COLUMNS]).astype(np.float32)
    np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, centered.astype(np.float32), site_median.astype(np.float32)


def isoformat_utc(time_s: int) -> str:
    return datetime.fromtimestamp(int(time_s), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def contiguous_end(time_s: np.ndarray, end: int, length: int) -> int | None:
    """Return the start index of a gap-free window ending at `end`, or None."""
    if end - length + 1 < 0:
        return None
    start = end - length + 1
    if np.any(np.diff(time_s[start : end + 1]) > MAX_GAP_SECONDS):
        return None
    return start
