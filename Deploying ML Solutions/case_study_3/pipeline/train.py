"""Fit every model on the pre-February readings and score February-March 2024.

Nothing in the fitting path is allowed to see a row from the holdout. The
sequence models and the sktime classifiers are fit on window samples; the
linear models are updated from every pre-holdout row; the tree models are fit
on a stratified sample because a 5-million-row forest will not fit in the
memory of the laptops this class uses.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from pipeline.build_features import build_feature_store, load_arrays
from pipeline.config import (
    ARTIFACT_DIR,
    CLASSES,
    CV_CUTOFFS,
    DURATION_CAP_SECONDS,
    EXPECTED_DATA_ROWS,
    FEATURE_COLUMNS,
    FEATURE_DIR,
    FIGURE_DIR,
    HOLDOUT_START_S,
    OLA_SITE,
    PCMH_SITE,
    RANDOM_STATE,
    TWO_EXCEL_SHEETS,
)
from pipeline.forecasting import run_frequency_forecast
from pipeline.load import partition_by_sensor
from pipeline.log import log
from pipeline.metrics import empty_confusion, scores_from_confusion, update_confusion
from pipeline.plots import (
    save_class_balance,
    save_comparison,
    save_confusion,
    save_forecast,
    save_frequency_hist,
    save_hourly,
    save_importance,
    save_learning_curves,
    save_pca,
    save_scatter,
    save_usage,
)
from pipeline.scoring import predict_labels
from pipeline.sequence_models import SequenceClassifier, save_sequence

DEPLOYABLE = [
    "rules_trailing",
    "sgd",
    "sgd_pca",
    "hgb",
    "rf",
    "hybrid_hgb",
    "hybrid_sgd",
]


def _class_weights(counts: dict) -> np.ndarray:
    values = np.array([counts[name] for name in CLASSES], dtype=np.float64)
    return values.sum() / (len(CLASSES) * np.maximum(values, 1.0))


def _pass_sensors():
    meta = json.loads((FEATURE_DIR / "sensor_meta.json").read_text(encoding="utf-8"))
    for respondent_id in meta:
        path = FEATURE_DIR / f"{respondent_id}.npz"
        yield respondent_id, meta[respondent_id], np.load(path)


def _fit_linear(weights: np.ndarray) -> dict:
    scaler = StandardScaler()
    log("fitting standardiser on every pre-holdout row")
    for _respondent, _meta, data in _pass_sensors():
        y = data["y"].astype(np.int64)
        train = (y >= 0) & (data["time_s"] < HOLDOUT_START_S)
        if np.any(train):
            scaler.partial_fit(data["X"][train].astype(np.float64))

    pca = IncrementalPCA(n_components=10, batch_size=4096)
    log("fitting incremental PCA")
    for _respondent, _meta, data in _pass_sensors():
        y = data["y"].astype(np.int64)
        train = (y >= 0) & (data["time_s"] < HOLDOUT_START_S)
        if np.any(train):
            scaled = scaler.transform(data["X"][train].astype(np.float64))
            np.clip(scaled, -8, 8, out=scaled)
            pca.partial_fit(scaled)

    sgd = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        random_state=RANDOM_STATE,
        average=True,
        tol=None,
        max_iter=1,
    )
    sgd_pca = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        random_state=RANDOM_STATE,
        average=True,
        tol=None,
        max_iter=1,
    )
    classes = np.arange(len(CLASSES))
    for epoch in (1, 2):
        log(f"online logistic regression, epoch {epoch}")
        for _respondent, _meta, data in _pass_sensors():
            y = data["y"].astype(np.int64)
            train = (y >= 0) & (data["time_s"] < HOLDOUT_START_S)
            if not np.any(train):
                continue
            target = y[train]
            scaled = scaler.transform(data["X"][train].astype(np.float64))
            np.clip(scaled, -8, 8, out=scaled)
            sample_weight = weights[target]
            sgd.partial_fit(scaled, target, classes=classes, sample_weight=sample_weight)
            sgd_pca.partial_fit(pca.transform(scaled), target, classes=classes, sample_weight=sample_weight)
    return {"scaler": scaler, "pca": pca, "sgd": sgd, "sgd_pca": sgd_pca}


def _take_per_class(X, y, n, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for class_id in range(len(CLASSES)):
        options = np.flatnonzero(y == class_id)
        if len(options) == 0:
            continue
        chosen.append(rng.choice(options, size=min(n, len(options)), replace=False))
    index = np.concatenate(chosen)
    return X[index], y[index]


def _fit_trees(X, y) -> dict:
    log(f"fitting gradient boosting on {len(y):,} sampled rows")
    boosting = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=120,
        max_depth=8,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )
    boosting.fit(X, y)
    X_rf, y_rf = _take_per_class(X, y, 8_000, RANDOM_STATE)
    log(f"fitting random forest on {len(y_rf):,} sampled rows")
    forest = RandomForestClassifier(
        n_estimators=40,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    forest.fit(X_rf, y_rf)
    return {"hgb": boosting, "rf": forest}


def _calendar_cv(X, y, time_s) -> list[dict]:
    order = np.argsort(time_s)
    X, y, time_s = X[order], y[order], time_s[order]
    if len(X) > 30_000:
        stride = max(1, len(X) // 30_000)
        X, y, time_s = X[::stride], y[::stride], time_s[::stride]
    boundaries = [int(item.timestamp()) for item in CV_CUTOFFS] + [HOLDOUT_START_S]
    folds = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        train = time_s < start
        test = (time_s >= start) & (time_s < end)
        if train.sum() < 200 or test.sum() < 50:
            continue
        scaler = StandardScaler().fit(X[train])
        linear = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            random_state=RANDOM_STATE,
            max_iter=30,
            tol=1e-3,
        )
        linear.fit(scaler.transform(X[train]), y[train])
        linear_pred = linear.predict(scaler.transform(X[test]))
        tree = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=60,
            max_depth=6,
            random_state=RANDOM_STATE,
        )
        tree.fit(X[train], y[train])
        tree_pred = tree.predict(X[test])
        folds.append(
            {
                "train_before": datetime.fromtimestamp(start, timezone.utc).date().isoformat(),
                "test_before": datetime.fromtimestamp(end, timezone.utc).date().isoformat(),
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "logistic_macro_f1": scores_from_confusion(_confusion(y[test], linear_pred))["macro_f1"],
                "boosting_macro_f1": scores_from_confusion(_confusion(y[test], tree_pred))["macro_f1"],
            }
        )
    return folds


def _confusion(y_true, y_pred) -> np.ndarray:
    matrix = empty_confusion()
    update_confusion(matrix, y_true, y_pred)
    return matrix


def _fit_sequences(windows, labels) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    index = rng.permutation(len(labels))
    cut = max(1, int(len(index) * 0.15))
    val, train = index[:cut], index[cut:]
    fitted = {}
    losses = {}
    for kind, lr in (("rnn", 0.01), ("lstm", 0.005)):
        log(f"training {kind}")
        model = SequenceClassifier(kind, n_features=2, n_hidden=32, seed=RANDOM_STATE)
        losses[kind] = model.train(
            windows[train],
            labels[train],
            windows[val],
            labels[val],
            epochs=8,
            batch_size=128,
            lr=lr,
        )
        fitted[kind] = model
    return {"models": fitted, "loss": losses}


def _fit_sktime(windows, labels) -> dict:
    # sktime panels are (instances, channels, time). Our windows are (instances, time, channels).
    panel = np.transpose(windows, (0, 2, 1)).astype(np.float64)
    frequency = panel[:, 1, :]
    fitted = {}
    log("fitting sktime SummaryClassifier and TimeSeriesForest")
    from sktime.classification.feature_based import SummaryClassifier
    from sktime.classification.interval_based import TimeSeriesForestClassifier

    summary = SummaryClassifier(random_state=RANDOM_STATE)
    summary.fit(panel, labels)
    fitted["sktime_summary"] = summary
    forest = TimeSeriesForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=1)
    forest.fit(frequency, labels)
    fitted["sktime_forest"] = forest
    try:
        from sktime.classification.kernel_based import RocketClassifier

        log("fitting sktime RocketClassifier")
        rocket = RocketClassifier(num_kernels=1_000, random_state=RANDOM_STATE, n_jobs=1)
        rocket.fit(panel, labels)
        fitted["sktime_rocket"] = rocket
    except Exception as exc:
        log(f"RocketClassifier skipped: {exc}")
    return fitted


def _score_windows(bundle, sktime_models, sequences, windows, labels, features) -> dict:
    panel = np.transpose(windows, (0, 2, 1)).astype(np.float64)
    results = {}
    boosting_pred = bundle["hgb"].predict(features)
    results["hgb_on_windows"] = scores_from_confusion(_confusion(labels, boosting_pred))
    for kind, model in sequences.items():
        pred = model.predict(windows)
        results[kind] = scores_from_confusion(_confusion(labels, pred))
    for name, model in sktime_models.items():
        if name == "sktime_forest":
            pred = model.predict(panel[:, 1, :])
        else:
            pred = model.predict(panel)
        results[name] = scores_from_confusion(_confusion(labels, pred))
    return results


def _evaluate(bundle) -> tuple[dict, dict, dict]:
    strategies = ["rules_centered", *DEPLOYABLE]
    mad_column = FEATURE_COLUMNS.index("trail_mad_freq")
    overall = {name: empty_confusion() for name in strategies}
    multi = {name: empty_confusion() for name in strategies}
    true_hours: dict[tuple, np.ndarray] = {}
    pred_hours = {name: {} for name in DEPLOYABLE}
    seen_test = 0
    for respondent_id, meta, data in _pass_sensors():
        y = data["y"].astype(np.int64)
        test = y >= 0
        test &= data["time_s"] >= HOLDOUT_START_S
        seen_test += int(test.sum())
        if not np.any(test):
            continue
        features = data["X"]
        site = str(data["site"])
        room = str(data["room"])
        is_pcmh = site == PCMH_SITE
        is_ola = site == OLA_SITE
        predictions = {}
        multi_mask = test & (features[:, FEATURE_COLUMNS.index("n_sources")] > 1.5)
        for name in strategies:
            mad_all = data["centered_mad"] if name == "rules_centered" else features[:, mad_column]
            predicted = np.full(len(y), -1, dtype=np.int8)
            predicted[test] = predict_labels(
                bundle,
                name,
                features[test],
                data["site_median"][test],
                is_pcmh,
                is_ola,
                mad=mad_all[test],
            )
            predictions[name] = predicted
            update_confusion(overall[name], y[test], predicted[test])
            update_confusion(multi[name], y[multi_mask], predicted[multi_mask])

        if len(y) < 2:
            continue
        interval = np.clip(np.diff(data["time_s"]).astype(np.float64), 0, DURATION_CAP_SECONDS)
        start_y = y[:-1]
        valid = start_y >= 0
        # Usage is reported for the holdout window only.
        valid &= data["time_s"][:-1] >= HOLDOUT_START_S
        if not np.any(valid):
            continue
        key = (site, room, respondent_id)
        true_hours.setdefault(key, np.zeros(3, np.float64))
        np.add.at(true_hours[key], start_y[valid], interval[valid])
        for name in DEPLOYABLE:
            pred_hours[name].setdefault(key, np.zeros(3, np.float64))
            np.add.at(pred_hours[name][key], predictions[name][:-1][valid], interval[valid])
        log(f"scored {respondent_id} ({seen_test:,} holdout rows so far)")
    return overall, multi, {"true": true_hours, "pred": pred_hours, "n_test": seen_test}


def _usage_frame(usage: dict, strategy: str) -> pd.DataFrame:
    rows = []
    for (site, room, respondent_id), true_seconds in usage["true"].items():
        predicted = usage["pred"][strategy].get((site, room, respondent_id), np.zeros(3))
        row = {"site_id": site, "room": room, "respondent_id": respondent_id}
        for index, name in enumerate(CLASSES):
            row[f"hours_{name}_true"] = round(float(true_seconds[index]) / 3600, 2)
            row[f"hours_{name}_pred"] = round(float(predicted[index]) / 3600, 2)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["site_id", "room"])
    return frame


def _importance(model, X, y) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    take = rng.choice(len(y), size=min(2_000, len(y)), replace=False)
    result = permutation_importance(
        model,
        X[take],
        y[take],
        n_repeats=4,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
        n_jobs=1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    return {
        FEATURE_COLUMNS[i]: round(float(result.importances_mean[i]), 4) for i in order
    }


def run(force_rebuild: bool = False) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    partition_by_sensor(force=force_rebuild)
    build_feature_store(force=force_rebuild)
    arrays = load_arrays()
    eda = arrays["eda"]
    weights = _class_weights(eda["train_class_counts"])
    linear = _fit_linear(weights)
    trees = _fit_trees(arrays["X_train"], arrays["y_train"].astype(np.int64))
    bundle = {**linear, **trees, "online_sgd": None}
    for name, model in {"sgd": bundle["sgd"], "sgd_pca": bundle["sgd_pca"], "hgb": bundle["hgb"], "rf": bundle["rf"]}.items():
        if not np.array_equal(model.classes_, np.arange(len(CLASSES))):
            raise RuntimeError(f"{name} class order is {model.classes_}, expected 0,1,2")

    log("temporal cross-validation on the teaching sample")
    cv_folds = _calendar_cv(arrays["X_train"], arrays["y_train"].astype(np.int64), arrays["t_train"])
    sequences = _fit_sequences(arrays["windows_train"], arrays["windows_train_y"].astype(np.int64))
    sktime_models = _fit_sktime(arrays["windows_train"], arrays["windows_train_y"].astype(np.int64))
    window_scores = _score_windows(
        bundle,
        sktime_models,
        sequences["models"],
        arrays["windows_test"],
        arrays["windows_test_y"].astype(np.int64),
        arrays["windows_test_F"],
    )

    log("scoring the full holdout")
    overall, multi, usage = _evaluate(bundle)
    full_scores = {}
    table = []
    for name in ["rules_centered", *DEPLOYABLE]:
        overall_scores = scores_from_confusion(overall[name])
        multi_scores = scores_from_confusion(multi[name])
        full_scores[name] = {"overall": overall_scores, "multi_source": multi_scores}
        table.append(
            {
                "model": name,
                "macro_f1": overall_scores["macro_f1"],
                "multi_macro_f1": multi_scores["macro_f1"],
                "generator_recall": overall_scores["generator_recall"],
                "accuracy": overall_scores["accuracy"],
            }
        )
    winner = max(DEPLOYABLE, key=lambda name: (full_scores[name]["multi_source"]["macro_f1"], full_scores[name]["overall"]["macro_f1"]))
    log(f"selected {winner}")

    importance = _importance(bundle["hgb"], arrays["X_test_sample"], arrays["y_test_sample"].astype(np.int64))
    multi_meta = [(rid, row) for rid, row in arrays["meta"].items() if row["n_sources"] >= 2]
    forecast_id, forecast_meta = max(multi_meta, key=lambda item: item[1]["n_rows_raw"])
    log(f"forecasting hourly frequency for {forecast_id}")
    forecast = run_frequency_forecast(forecast_id, forecast_meta["room"], forecast_meta["site_id"])

    save_class_balance(eda["label_counts"], FIGURE_DIR / "class_balance.png")
    save_frequency_hist(eda["freq_edges"], eda["freq_hist"], FIGURE_DIR / "frequency_by_source.png")
    save_scatter(arrays["scatter"]["voltage"], arrays["scatter"]["frequency"], arrays["scatter"]["y"], FIGURE_DIR / "voltage_frequency.png")
    save_hourly(eda["hour_label_counts"], FIGURE_DIR / "hour_of_day.png")
    save_pca(linear["pca"].explained_variance_ratio_, FIGURE_DIR / "pca_scree.png")
    save_comparison(table, FIGURE_DIR / "model_comparison.png")
    save_confusion(overall[winner], FIGURE_DIR / "confusion_matrix.png", f"Holdout confusion, {winner}")
    save_importance(list(importance), list(importance.values()), FIGURE_DIR / "permutation_importance.png")
    save_learning_curves(sequences["loss"]["rnn"], sequences["loss"]["lstm"], FIGURE_DIR / "sequence_loss.png")
    if not forecast.get("skipped"):
        save_forecast(
            forecast["y_test"],
            forecast["pred_naive"],
            forecast["pred_ridge"],
            FIGURE_DIR / "frequency_forecast.png",
            f"Hourly frequency, {forecast['room']}",
        )
        forecast["series"].to_csv(ARTIFACT_DIR / "hourly_frequency.csv", header=True)
        forecast["y_test"].to_csv(ARTIFACT_DIR / "forecast_holdout.csv", header=["actual"])

    usage_frame = _usage_frame(usage, winner)
    usage_frame.to_csv(ARTIFACT_DIR / "usage_durations.csv", index=False)
    save_usage(usage_frame, FIGURE_DIR / "generator_hours.png")

    save_sequence(sequences["models"]["rnn"], ARTIFACT_DIR / "rnn_weights.npz")
    save_sequence(sequences["models"]["lstm"], ARTIFACT_DIR / "lstm_weights.npz")
    rng = np.random.default_rng(RANDOM_STATE)
    teach = rng.choice(len(arrays["y_train"]), size=min(8_000, len(arrays["y_train"])), replace=False)
    np.savez(
        ARTIFACT_DIR / "teaching_sample.npz",
        X=arrays["X_train"][teach],
        y=arrays["y_train"][teach],
        time_s=arrays["t_train"][teach],
        feature_names=np.array(FEATURE_COLUMNS),
    )
    np.savez(ARTIFACT_DIR / "windows_train.npz", X=arrays["windows_train"], y=arrays["windows_train_y"])
    np.savez(ARTIFACT_DIR / "windows_test.npz", X=arrays["windows_test"], y=arrays["windows_test_y"])
    (ARTIFACT_DIR / "examples.json").write_text(json.dumps(arrays["examples"], indent=2), encoding="utf-8")

    forecast_public = {key: value for key, value in forecast.items() if key not in {"y_test", "pred_naive", "pred_ridge", "series"}}
    payload = {
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sklearn_version": sklearn_version,
        "expected_data_rows": EXPECTED_DATA_ROWS,
        "excel_two_sheet_trap": TWO_EXCEL_SHEETS,
        "rows_in_feature_store": eda["rows_in_feature_store"],
        "duplicate_timestamps_collapsed": eda["duplicate_timestamps_collapsed"],
        "holdout_start": "2024-02-01T00:00:00Z",
        "holdout_rows": usage["n_test"],
        "default_strategy": winner,
        "label_counts": eda["label_counts"],
        "train_class_counts": eda["train_class_counts"],
        "gap_seconds": eda["gap_seconds"],
        "pca_explained_variance_ratio": [round(float(v), 4) for v in linear["pca"].explained_variance_ratio_],
        "pca_cumulative": round(float(np.sum(linear["pca"].explained_variance_ratio_)), 4),
        "cv_folds": cv_folds,
        "holdout": full_scores,
        "comparison": table,
        "window_holdout": window_scores,
        "sequence_train_loss": sequences["loss"],
        "permutation_importance": importance,
        "forecast": forecast_public,
        "usage_generator_hours_true": round(float(usage_frame["hours_generator_true"].sum()), 1),
        "usage_generator_hours_pred": round(float(usage_frame["hours_generator_pred"].sum()), 1),
        "sensors_with_varying_metadata": eda["sensors_with_varying_metadata"],
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "eda_summary.json").write_text(json.dumps(eda), encoding="utf-8")

    model_bundle = {
        "version": payload["version"],
        "trained_at": payload["trained_at"],
        "sklearn_version": sklearn_version,
        "default_strategy": winner,
        "classes": CLASSES,
        "feature_names": FEATURE_COLUMNS,
        "holdout_start": payload["holdout_start"],
        "rows_in_feature_store": eda["rows_in_feature_store"],
        "scaler": bundle["scaler"],
        "pca": bundle["pca"],
        "sgd": bundle["sgd"],
        "sgd_pca": bundle["sgd_pca"],
        "hgb": bundle["hgb"],
        "rf": bundle["rf"],
    }
    joblib.dump(model_bundle, ARTIFACT_DIR / "model.joblib")
    online = {"sgd": bundle["sgd"], "scaler": bundle["scaler"], "classes": CLASSES, "feature_names": FEATURE_COLUMNS}
    joblib.dump(online, ARTIFACT_DIR / "online_model_init.joblib")
    shutil.copyfile(ARTIFACT_DIR / "online_model_init.joblib", ARTIFACT_DIR / "online_model.joblib")
    log(f"wrote artifacts to {ARTIFACT_DIR}")
    try:
        from pipeline.shap_analysis import run_shap

        summary = run_shap()
        payload["shap"] = {
            "method": summary["method"],
            "rows_explained": summary["rows_explained"],
            "mean_abs_shap": {
                name: dict(list(features.items())[:8])
                for name, features in summary["mean_abs_shap"].items()
            },
        }
    except Exception as exc:
        log(f"SHAP skipped ({type(exc).__name__}: {exc}). Run python -m pipeline.shap_analysis later.")
    return payload


def main() -> None:
    run(force_rebuild=False)


if __name__ == "__main__":
    main()
