"""Figures for the training session. Matplotlib only, so the image stays small."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.config import CLASSES

COLORS = {"generator": "#C2410C", "grid": "#1D4ED8", "solar": "#B45309", "unknown": "#6B7280"}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def save_class_balance(counts: dict, path: Path) -> None:
    _style()
    names = ["generator", "grid", "solar", "unknown"]
    values = [counts[name] for name in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values, color=[COLORS[name] for name in names])
    ax.set_ylabel("Readings")
    ax.set_title("Active power source labels across every sensor reading")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value/1e6:.2f}M", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_frequency_hist(edges, hist, path: Path) -> None:
    _style()
    edges = np.asarray(edges, dtype=float)
    hist = np.asarray(hist, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    width = float(edges[1] - edges[0])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, name in enumerate(CLASSES):
        total = hist[index].sum()
        share = hist[index] / total if total else hist[index]
        ax.plot(centers, share, color=COLORS[name], label=name, linewidth=2)
    ax.axvspan(49.8, 50.2, color="#FDE68A", alpha=0.5, label="solar band 49.8-50.2")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Share of that class")
    ax.set_title("Frequency by the rule-based label")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_scatter(voltage, frequency, labels, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for index, name in enumerate(CLASSES):
        mask = np.asarray(labels) == index
        ax.scatter(
            frequency[mask],
            voltage[mask],
            s=8,
            alpha=0.35,
            c=COLORS[name],
            label=name,
            linewidths=0,
        )
    ax.set_xlim(45, 55)
    ax.set_ylim(180, 270)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Stratified sample of readings (generator is over-represented here)")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_hourly(hour_counts, path: Path) -> None:
    _style()
    counts = np.asarray(hour_counts, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hours = np.arange(24)
    for index, name in enumerate(CLASSES):
        total = counts[:, index] 
        share = total / counts[:, :3].sum(axis=1).clip(min=1)
        ax.plot(hours, share, color=COLORS[name], label=name, linewidth=2)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (Sierra Leone, UTC)")
    ax.set_ylabel("Share of known labels")
    ax.set_title("When each source carries the load")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_pca(ratio, path: Path) -> None:
    _style()
    ratio = np.asarray(ratio, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    index = np.arange(1, len(ratio) + 1)
    ax.bar(index, ratio, color="#1D4ED8", label="component")
    ax.plot(index, np.cumsum(ratio), color="#C2410C", marker="o", label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Share of variance")
    ax.set_ylim(0, 1.05)
    ax.set_title("Incremental PCA on every pre-holdout reading")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_comparison(table: list[dict], path: Path) -> None:
    _style()
    names = [row["model"] for row in table]
    overall = [row["macro_f1"] for row in table]
    multi = [row["multi_macro_f1"] for row in table]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(y - 0.18, overall, height=0.36, color="#1D4ED8", label="all rooms")
    ax.barh(y + 0.18, multi, height=0.36, color="#B45309", label="multi-source rooms")
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 on the Feb-Mar 2024 holdout")
    ax.set_title("Model comparison")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_confusion(matrix, path: Path, title: str) -> None:
    _style()
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Rule-based label")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(matrix[i, j]):,}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_importance(names, values, path: Path) -> None:
    _style()
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.asarray(names)[order], np.asarray(values)[order], color="#1D4ED8")
    ax.set_xlabel("Drop in macro F1 when shuffled")
    ax.set_title("Permutation importance, gradient boosting, holdout sample")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_usage(frame: pd.DataFrame, path: Path) -> None:
    _style()
    grouped = frame.groupby("site_id")[["hours_generator_true", "hours_generator_pred"]].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(grouped))
    ax.bar(x - 0.18, grouped["hours_generator_true"], width=0.36, color="#9A3412", label="rule label")
    ax.bar(x + 0.18, grouped["hours_generator_pred"], width=0.36, color="#C2410C", label="model")
    ax.set_xticks(x, grouped.index, rotation=20, ha="right")
    ax.set_ylabel("Generator hours in the holdout")
    ax.set_title("Generator runtime by hospital")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_forecast(y_test, pred_naive, pred_ridge, path: Path, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    show = slice(-24 * 7, None)
    ax.plot(y_test.index[show], y_test.to_numpy()[show], color="#111827", label="actual", linewidth=1.5)
    ax.plot(pred_naive.index[show], pred_naive.to_numpy()[show], color="#1D4ED8", label="seasonal naive", linewidth=1)
    ax.plot(pred_ridge.index[show], pred_ridge.to_numpy()[show], color="#C2410C", label="Ridge reduction", linewidth=1)
    ax.set_ylabel("Hz")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_learning_curves(rnn_loss, lstm_loss, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(rnn_loss) + 1), rnn_loss, label="RNN", color="#1D4ED8")
    ax.plot(np.arange(1, len(lstm_loss) + 1), lstm_loss, label="LSTM", color="#C2410C")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training cross-entropy")
    ax.set_title("Sequence models on raw voltage and frequency windows")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
