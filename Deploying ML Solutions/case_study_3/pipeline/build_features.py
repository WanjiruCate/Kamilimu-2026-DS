"""One-sensor feature files, EDA accumulators, and the samples models train on.

The raw extract stays partitioned. This step reads one sensor, builds the
grid-reference median for its site, writes a float32 matrix, and keeps only
stratified samples in memory.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.config import (
    FEATURE_COLUMNS,
    FEATURE_DIR,
    FEATURE_VERSION,
    HOLDOUT_START_S,
    EXPECTED_DATA_ROWS,
    RAW_DIR,
    SEQUENCE_LENGTH,
    TABULAR_PER_CLASS,
    WINDOW_TEST_PER_CLASS,
    WINDOW_TRAIN_PER_CLASS,
)
from pipeline.features import (
    bucket_ids,
    contiguous_end,
    encode_label,
    engineer_sensor,
    isoformat_utc,
    source_flags,
    to_unix_seconds,
)
from pipeline.log import log
from pipeline.metrics import StratifiedReservoir, WindowReservoir

FREQ_EDGES = np.linspace(42.0, 58.0, 65)


def _feature_marker() -> dict | None:
    path = FEATURE_DIR / "READY.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("feature_version") != FEATURE_VERSION:
        return None
    return payload


def _inventory() -> dict[str, dict]:
    """Read only the string columns needed to group sensors by site."""
    info = {}
    for path in sorted(RAW_DIR.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["site_id", "room", "available_power_sources"])
        site = str(frame["site_id"].mode().iloc[0])
        room = str(frame["room"].mode().iloc[0])
        sources = str(frame["available_power_sources"].mode().iloc[0])
        flags = source_flags(sources)
        info[path.stem] = {
            "site_id": site,
            "room": room,
            "available_power_sources": sources,
            "n_sources": int(flags[3]),
            "n_rows_raw": int(len(frame)),
            "site_nunique": int(frame["site_id"].nunique()),
            "room_nunique": int(frame["room"].nunique()),
            "source_nunique": int(frame["available_power_sources"].nunique()),
        }
    return info


def _site_reference(respondent_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    buckets = []
    values = []
    for respondent_id in respondent_ids:
        frame = pd.read_parquet(RAW_DIR / f"{respondent_id}.parquet", columns=["time", "frequency"])
        time_s = to_unix_seconds(frame["time"])
        buckets.append(bucket_ids(time_s))
        values.append(frame["frequency"].to_numpy(np.float32))
    bucket = np.concatenate(buckets)
    freq = np.concatenate(values).astype(np.float64)
    order = np.argsort(bucket, kind="mergesort")
    bucket = bucket[order]
    freq = freq[order]
    change = np.flatnonzero(np.diff(bucket)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(bucket)]
    keys = bucket[starts].astype(np.int64)
    medians = np.empty(len(starts), np.float64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        medians[i] = np.median(freq[start:end])
    counts = (ends - starts).astype(np.float64)
    return keys, medians.astype(np.float32), counts.astype(np.float32)


def _load_sensor(path, info: dict):
    frame = pd.read_parquet(path).sort_values("time")
    collapsed = int(frame["time"].duplicated().sum())
    if collapsed:
        frame = frame.groupby("time", as_index=False).agg(
            voltage=("voltage", "mean"),
            frequency=("frequency", "mean"),
            active_power_source=(
                "active_power_source",
                lambda s: s.iloc[0] if s.nunique() == 1 else "unknown",
            ),
            available_power_sources=("available_power_sources", "first"),
            site_id=("site_id", "first"),
            room=("room", "first"),
        )
    time_s = to_unix_seconds(frame["time"])
    voltage = frame["voltage"].to_numpy(np.float64)
    frequency = frame["frequency"].to_numpy(np.float64)
    labels = encode_label(frame["active_power_source"].to_numpy())
    if info["source_nunique"] == 1 and not collapsed:
        flag = source_flags(info["available_power_sources"])
        n = len(time_s)
        has_grid = np.full(n, flag[0])
        has_solar = np.full(n, flag[1])
        has_generator = np.full(n, flag[2])
    else:
        lower = np.char.lower(frame["available_power_sources"].astype(str).to_numpy())
        has_grid = (np.char.find(lower, "grid") >= 0).astype(np.float64)
        has_solar = (np.char.find(lower, "solar") >= 0).astype(np.float64)
        has_generator = (np.char.find(lower, "generator") >= 0).astype(np.float64)
    return time_s, voltage, frequency, labels, collapsed, has_grid, has_solar, has_generator


def build_feature_store(force: bool = False) -> dict:
    """Create per-sensor npz files and the in-memory samples used for fitting."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _feature_marker()
    if cached and not force:
        log("reusing feature store")
        return cached

    for old in FEATURE_DIR.glob("*.npz"):
        old.unlink()

    info = _inventory()
    by_site: dict[str, list[str]] = defaultdict(list)
    for respondent_id, row in info.items():
        by_site[row["site_id"]].append(respondent_id)

    references = {}
    for site, members in by_site.items():
        log(f"site median: {site} ({len(members)} sensors)")
        references[site] = _site_reference(members)

    freq_hist = np.zeros((3, len(FREQ_EDGES) - 1), np.int64)
    hour_label = np.zeros((24, 4), np.int64)
    label_counts = np.zeros(4, np.int64)
    low_voltage = np.zeros(4, np.int64)
    train_counts = np.zeros(3, np.int64)
    test_counts = np.zeros(3, np.int64)
    multi_train = np.zeros(3, np.int64)
    rows_modelled = 0
    collapsed_total = 0
    gap_over = 0
    gap_n = 0
    gap_sum = 0.0
    gap_draw = np.empty(200_000, np.float32)
    gap_filled = 0
    gap_seen = 0
    rng = np.random.default_rng(0)

    tabular = StratifiedReservoir(TABULAR_PER_CLASS, len(FEATURE_COLUMNS), seed=7)
    test_sample = StratifiedReservoir(2_000, len(FEATURE_COLUMNS), seed=11)
    scatter = StratifiedReservoir(1_500, 2, seed=3)
    windows_train = WindowReservoir(WINDOW_TRAIN_PER_CLASS, SEQUENCE_LENGTH, len(FEATURE_COLUMNS), seed=21)
    windows_test = WindowReservoir(WINDOW_TEST_PER_CLASS, SEQUENCE_LENGTH, len(FEATURE_COLUMNS), seed=22)
    examples: dict[str, dict] = {}
    varying = []

    for index, (respondent_id, row) in enumerate(sorted(info.items()), start=1):
        if row["site_nunique"] > 1 or row["room_nunique"] > 1 or row["source_nunique"] > 1:
            varying.append(respondent_id)
        path = RAW_DIR / f"{respondent_id}.parquet"
        time_s, voltage, frequency, labels, collapsed, has_grid, has_solar, has_generator = _load_sensor(
            path, row
        )
        collapsed_total += collapsed
        keys, medians, counts = references[row["site_id"]]
        features, centered, site_median = engineer_sensor(
            time_s,
            voltage,
            frequency,
            has_grid,
            has_solar,
            has_generator,
            keys,
            medians,
            counts,
        )
        np.savez(
            FEATURE_DIR / f"{respondent_id}.npz",
            time_s=time_s.astype(np.int64),
            y=labels.astype(np.int8),
            X=features,
            centered_mad=centered,
            site_median=site_median,
            site=np.array(row["site_id"]),
            room=np.array(row["room"]),
            sources=np.array(row["available_power_sources"]),
        )
        rows_modelled += len(time_s)
        known = labels >= 0
        unknown = ~known
        label_counts[:3] += np.bincount(labels[known], minlength=3)
        label_counts[3] += int(unknown.sum())
        hour = ((time_s // 3600) % 24).astype(np.int64)
        lab = labels.astype(np.int64).copy()
        lab[unknown] = 3
        np.add.at(hour_label, (hour, lab), 1)
        low = voltage < 50
        low_voltage[:3] += np.bincount(labels[known & low], minlength=3) if np.any(known & low) else 0
        low_voltage[3] += int(np.sum(unknown & low))
        for class_id in range(3):
            mask = labels == class_id
            if np.any(mask):
                hist, _ = np.histogram(frequency[mask], bins=FREQ_EDGES)
                freq_hist[class_id] += hist
        train_mask = known & (time_s < HOLDOUT_START_S)
        test_mask = known & (time_s >= HOLDOUT_START_S)
        train_counts += np.bincount(labels[train_mask], minlength=3)
        test_counts += np.bincount(labels[test_mask], minlength=3)
        multi = features[:, FEATURE_COLUMNS.index("n_sources")] > 1.5
        multi_train += np.bincount(labels[train_mask & multi], minlength=3)

        if len(time_s) > 1:
            delta = np.diff(time_s)
            positive = delta[delta > 0]
            gap_n += int(len(positive))
            gap_sum += float(positive.sum())
            gap_over += int(np.sum(positive > 600))
            for value in positive[::20]:
                gap_seen += 1
                if gap_filled < len(gap_draw):
                    gap_draw[gap_filled] = value
                    gap_filled += 1
                else:
                    draw = int(rng.integers(0, gap_seen))
                    if draw < len(gap_draw):
                        gap_draw[draw] = value

        tabular.add_many(features[train_mask], labels[train_mask], time_s[train_mask])
        test_sample.add_many(features[test_mask], labels[test_mask], time_s[test_mask])
        scatter_idx = np.flatnonzero(known)[::40]
        scatter_x = np.column_stack([voltage[scatter_idx], frequency[scatter_idx]]).astype(np.float32)
        scatter.add_many(scatter_x, labels[scatter_idx], time_s[scatter_idx])

        _collect_windows(windows_train, windows_test, time_s, voltage, frequency, labels, features)
        _maybe_example(examples, row, time_s, features, labels, site_median)
        if index % 5 == 0 or index == len(info):
            log(f"features {index}/{len(info)} sensors, {rows_modelled:,} rows")

    if rows_modelled + collapsed_total != EXPECTED_DATA_ROWS:
        raise RuntimeError(
            "Feature store does not account for every source row. "
            f"modelled={rows_modelled:,}, collapsed_duplicates={collapsed_total:,}, "
            f"expected={EXPECTED_DATA_ROWS:,}."
        )

    X, y, times, _ = tabular.stack()
    Xt, yt, tt, _ = test_sample.stack()
    Xs, ys, _ts, _ = scatter.stack()
    Xw, yw = windows_train.stack()
    Xw_te, yw_te = windows_test.stack()
    # Window reservoirs also keep the tabular row at the end of the window.
    Fw, _ = windows_train.stack_features()
    Fw_te, _ = windows_test.stack_features()

    np.savez(FEATURE_DIR / "tabular_train.npz", X=X, y=y, time_s=times)
    np.savez(FEATURE_DIR / "tabular_test.npz", X=Xt, y=yt, time_s=tt)
    np.savez(FEATURE_DIR / "windows_train.npz", X=Xw, y=yw, F=Fw)
    np.savez(FEATURE_DIR / "windows_test.npz", X=Xw_te, y=yw_te, F=Fw_te)
    np.savez(FEATURE_DIR / "scatter.npz", voltage=Xs[:, 0], frequency=Xs[:, 1], y=ys)

    gaps = gap_draw[:gap_filled]
    eda = {
        "feature_version": FEATURE_VERSION,
        "rows_in_feature_store": int(rows_modelled),
        "duplicate_timestamps_collapsed": int(collapsed_total),
        "label_counts": {
            "generator": int(label_counts[0]),
            "grid": int(label_counts[1]),
            "solar": int(label_counts[2]),
            "unknown": int(label_counts[3]),
        },
        "train_class_counts": {
            "generator": int(train_counts[0]),
            "grid": int(train_counts[1]),
            "solar": int(train_counts[2]),
        },
        "test_class_counts": {
            "generator": int(test_counts[0]),
            "grid": int(test_counts[1]),
            "solar": int(test_counts[2]),
        },
        "multi_source_train_counts": {
            "generator": int(multi_train[0]),
            "grid": int(multi_train[1]),
            "solar": int(multi_train[2]),
        },
        "low_voltage_counts": {
            "generator": int(low_voltage[0]),
            "grid": int(low_voltage[1]),
            "solar": int(low_voltage[2]),
            "unknown": int(low_voltage[3]),
        },
        "freq_edges": FREQ_EDGES.tolist(),
        "freq_hist": freq_hist.tolist(),
        "hour_label_counts": hour_label.tolist(),
        "gap_seconds": {
            "n": int(gap_n),
            "mean": float(gap_sum / gap_n) if gap_n else None,
            "p50": float(np.percentile(gaps, 50)) if len(gaps) else None,
            "p95": float(np.percentile(gaps, 95)) if len(gaps) else None,
            "share_over_10_min": float(gap_over / gap_n) if gap_n else None,
        },
        "sensors_with_varying_metadata": varying,
        "n_sensors": len(info),
    }
    (FEATURE_DIR / "eda.json").write_text(json.dumps(eda), encoding="utf-8")
    (FEATURE_DIR / "sensor_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (FEATURE_DIR / "examples.json").write_text(json.dumps(list(examples.values()), indent=2), encoding="utf-8")
    marker = {
        "feature_version": FEATURE_VERSION,
        "rows_in_feature_store": int(rows_modelled),
        "n_sensors": len(info),
        "duplicate_timestamps_collapsed": int(collapsed_total),
    }
    (FEATURE_DIR / "READY.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    log(f"feature store ready: {rows_modelled:,} rows, {collapsed_total:,} duplicate timestamps collapsed")
    return marker


def _collect_windows(train_res, test_res, time_s, voltage, frequency, labels, features) -> None:
    n = len(time_s)
    if n < SEQUENCE_LENGTH:
        return
    # About 1 in 11 endings. Enough to fill 1,500 per class without a 5-million-iteration copy.
    candidate = np.flatnonzero((np.arange(n) >= SEQUENCE_LENGTH - 1) & ((time_s % 11) == 0) & (labels >= 0))
    for end in candidate:
        start = contiguous_end(time_s, int(end), SEQUENCE_LENGTH)
        if start is None:
            continue
        window = np.column_stack(
            [voltage[start : end + 1], frequency[start : end + 1]]
        ).astype(np.float32)
        target = train_res if time_s[end] < HOLDOUT_START_S else test_res
        target.add(window, features[end], int(labels[end]))


def _maybe_example(examples, row, time_s, features, labels, site_median) -> None:
    from pipeline.config import CLASSES

    needed = [i for i in range(3) if CLASSES[i] not in examples]
    if not needed:
        return
    candidates = np.flatnonzero((time_s >= HOLDOUT_START_S) & np.isin(labels, needed))
    for end in candidates:
        class_id = int(labels[end])
        name = CLASSES[class_id]
        if name in examples:
            continue
        start = contiguous_end(time_s, int(end), SEQUENCE_LENGTH)
        if start is None:
            continue
        history = []
        for i in range(start, end + 1):
            history.append(
                {
                    "time": isoformat_utc(int(time_s[i])),
                    "voltage": float(features[i, 0]),
                    "frequency": float(features[i, 1]),
                }
            )
        n_col = FEATURE_COLUMNS.index("n_sensors_in_bucket")
        examples[name] = {
            "site_id": row["site_id"],
            "room": row["room"],
            "available_power_sources": row["available_power_sources"],
            "site_median_frequency": float(site_median[end]),
            "n_sensors_in_bucket": int(round(float(features[end, n_col]))),
            "true_label_for_teaching": name,
            "history": history,
        }


def load_arrays() -> dict:
    """Load the samples written by build_feature_store."""
    train = np.load(FEATURE_DIR / "tabular_train.npz")
    test = np.load(FEATURE_DIR / "tabular_test.npz")
    windows_train = np.load(FEATURE_DIR / "windows_train.npz")
    windows_test = np.load(FEATURE_DIR / "windows_test.npz")
    scatter = np.load(FEATURE_DIR / "scatter.npz")
    eda = json.loads((FEATURE_DIR / "eda.json").read_text(encoding="utf-8"))
    meta = json.loads((FEATURE_DIR / "sensor_meta.json").read_text(encoding="utf-8"))
    examples = json.loads((FEATURE_DIR / "examples.json").read_text(encoding="utf-8"))
    return {
        "X_train": train["X"],
        "y_train": train["y"],
        "t_train": train["time_s"],
        "X_test_sample": test["X"],
        "y_test_sample": test["y"],
        "t_test_sample": test["time_s"],
        "windows_train": windows_train["X"],
        "windows_train_y": windows_train["y"],
        "windows_train_F": windows_train["F"],
        "windows_test": windows_test["X"],
        "windows_test_y": windows_test["y"],
        "windows_test_F": windows_test["F"],
        "scatter": scatter,
        "eda": eda,
        "meta": meta,
        "examples": examples,
    }
