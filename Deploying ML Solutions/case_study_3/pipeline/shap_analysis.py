"""SHAP explanations for the deployed boosting model.

Tree SHAP does not yet support multiclass
``HistGradientBoostingClassifier`` when the output is a probability
(shap issue 1028). The explanations below are permutation SHAP values
for ``predict_proba``, so a positive value is "this feature raised the
probability of this class".

The background rows come from the training sample and the explained
rows come from the holdout sample. Both files are written by
``build_features``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.config import ARTIFACT_DIR, CLASSES, FEATURE_COLUMNS, FEATURE_DIR, FIGURE_DIR
from pipeline.log import log

PER_CLASS = 120
# Thirty-four rows from each class is about 100 background rows, which is
# also shap's default masker cap. Staying under that cap means every
# background row is actually used.
BACKGROUND_PER_CLASS = 34
SEED = 11


def mean_abs_by_class(values: np.ndarray) -> dict[str, dict[str, float]]:
    """Average |SHAP| per feature. ``values`` has shape (rows, features, classes)."""
    if values.ndim != 3 or values.shape[1] != len(FEATURE_COLUMNS) or values.shape[2] != len(CLASSES):
        raise ValueError(f"Expected (rows, {len(FEATURE_COLUMNS)}, {len(CLASSES)}), got {values.shape}")
    magnitude = np.mean(np.abs(values), axis=0)
    report: dict[str, dict[str, float]] = {}
    for class_index, name in enumerate(CLASSES):
        order = np.argsort(-magnitude[:, class_index])
        report[name] = {
            FEATURE_COLUMNS[int(feature)]: round(float(magnitude[feature, class_index]), 6)
            for feature in order
        }
    return report


def _stratified(matrix: np.ndarray, labels: np.ndarray, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    chosen = []
    for class_id in range(len(CLASSES)):
        options = np.flatnonzero(labels == class_id)
        if len(options) == 0:
            continue
        take = min(per_class, len(options))
        chosen.append(rng.choice(options, size=take, replace=False))
    index = np.concatenate(chosen)
    rng.shuffle(index)
    return matrix[index].astype(np.float64), labels[index]


def _waterfall(ax, feature_names: list[str], values: np.ndarray, data: np.ndarray, base: float, title: str) -> None:
    order = np.argsort(-np.abs(values))[:8]
    names = [f"{feature_names[i]} = {data[i]:.3g}" for i in order][::-1]
    shap_values = values[order][::-1]
    colors = ["#B45309" if value > 0 else "#1D4ED8" for value in shap_values]
    ax.barh(names, shap_values, color=colors)
    ax.axvline(0, color="#44403C", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(f"change in probability (base {base:.2f})")


def run_shap(per_class: int = PER_CLASS) -> dict:
    import joblib
    import shap

    model_path = ARTIFACT_DIR / "model.joblib"
    train_path = FEATURE_DIR / "tabular_train.npz"
    test_path = FEATURE_DIR / "tabular_test.npz"
    for path in (model_path, train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Train first (`python -m pipeline.train`) "
                "so the holdout feature sample exists."
            )

    bundle = joblib.load(model_path)
    model = bundle["hgb"]
    train = np.load(train_path)
    test = np.load(test_path)
    background, _ = _stratified(train["X"], train["y"], BACKGROUND_PER_CLASS, SEED)
    explained_x, explained_y = _stratified(test["X"], test["y"], per_class, SEED + 1)
    log(
        f"SHAP: explaining {len(explained_x)} holdout rows "
        f"with a background of {len(background)} training rows"
    )

    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.PermutationExplainer(model.predict_proba, masker, seed=SEED)
    explanation = explainer(explained_x, max_evals=2 * len(FEATURE_COLUMNS) + 1)
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    ranking = mean_abs_by_class(values)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _plot_mean_bars(ranking)
    _plot_beeswarm(shap, explanation, class_index=0, filename="shap_beeswarm_generator.png")
    _plot_dependence(explained_x, values)
    _plot_local(explained_x, explained_y, values, base)

    np.savez_compressed(
        ARTIFACT_DIR / "shap_values.npz",
        values=values.astype(np.float32),
        base=base.astype(np.float32),
        X=explained_x.astype(np.float32),
        y=explained_y.astype(np.int8),
        feature_names=np.array(FEATURE_COLUMNS),
    )

    local = []
    for class_id, name in enumerate(CLASSES):
        matches = np.flatnonzero(explained_y == class_id)
        if len(matches) == 0:
            continue
        row = int(matches[0])
        order = np.argsort(-np.abs(values[row, :, class_id]))[:5]
        local.append({
            "true_label": name,
            "base_probability": round(float(base[row, class_id]), 4),
            "top_features": [
                {
                    "feature": FEATURE_COLUMNS[int(feature)],
                    "value": round(float(explained_x[row, feature]), 4),
                    "shap": round(float(values[row, feature, class_id]), 4),
                }
                for feature in order
            ],
        })

    summary = {
        "method": "permutation SHAP on HistGradientBoosting predict_proba",
        "why_not_tree_shap": (
            "shap.TreeExplainer raises NotImplementedError for multiclass "
            "HistGradientBoostingClassifier unless model_output='raw'."
        ),
        "explained_model": "hgb",
        "rows_explained": int(len(explained_x)),
        "background_rows": int(len(background)),
        "rows_per_class": {name: int(np.sum(explained_y == i)) for i, name in enumerate(CLASSES)},
        "mean_abs_shap": ranking,
        "local_examples": local,
    }
    (ARTIFACT_DIR / "shap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _merge_into_metrics(summary)
    log("SHAP figures written under artifacts/figures")
    return summary


def _merge_into_metrics(summary: dict) -> None:
    metrics_path = ARTIFACT_DIR / "metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["shap"] = {
        "method": summary["method"],
        "rows_explained": summary["rows_explained"],
        "mean_abs_shap": {
            name: dict(list(features.items())[:8])
            for name, features in summary["mean_abs_shap"].items()
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _plot_mean_bars(ranking: dict[str, dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), sharex=False)
    colors = {"generator": "#C2410C", "grid": "#1D4ED8", "solar": "#CA8A04"}
    for ax, name in zip(axes, CLASSES):
        items = list(ranking[name].items())[:8][::-1]
        ax.barh([item[0] for item in items], [item[1] for item in items], color=colors[name])
        ax.set_title(f"Mean |SHAP| · {name}")
        ax.set_xlabel("average |change in probability|")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_mean_abs.png", dpi=140)
    plt.close(fig)


def _plot_beeswarm(shap, explanation, class_index: int, filename: str) -> None:
    class_explanation = shap.Explanation(
        values=np.asarray(explanation.values)[:, :, class_index],
        base_values=np.asarray(explanation.base_values)[:, class_index],
        data=np.asarray(explanation.data),
        feature_names=list(FEATURE_COLUMNS),
    )
    plt.figure(figsize=(8, 5.2))
    shap.plots.beeswarm(class_explanation, max_display=12, show=False)
    plt.title(f"SHAP beeswarm · probability of {CLASSES[class_index]}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=140, bbox_inches="tight")
    plt.close()


def _plot_dependence(features: np.ndarray, values: np.ndarray) -> None:
    pairs = [
        (FEATURE_COLUMNS.index("has_grid"), 0, "has_grid vs generator probability"),
        (FEATURE_COLUMNS.index("abs_freq_dev_50"), 0, "frequency deviation vs generator probability"),
        (FEATURE_COLUMNS.index("freq_dev_site_median"), 2, "site deviation vs solar probability"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (feature, class_index, title) in zip(axes, pairs):
        ax.scatter(
            features[:, feature],
            values[:, feature, class_index],
            s=10,
            alpha=0.45,
            c="#44403C",
        )
        ax.axhline(0, color="#A8A29E", linewidth=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(FEATURE_COLUMNS[feature])
        ax.set_ylabel("SHAP")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_dependence.png", dpi=140)
    plt.close(fig)


def _plot_local(features, labels, values, base) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, class_id, name in zip(axes, range(3), CLASSES):
        matches = np.flatnonzero(labels == class_id)
        row = int(matches[0])
        _waterfall(
            ax,
            FEATURE_COLUMNS,
            values[row, :, class_id],
            features[row],
            float(base[row, class_id]),
            f"One {name} window",
        )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_waterfall.png", dpi=140)
    plt.close(fig)


def main() -> None:
    run_shap()


if __name__ == "__main__":
    main()
