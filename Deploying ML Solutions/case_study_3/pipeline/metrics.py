"""Confusion-matrix metrics that can be updated one batch at a time."""

from __future__ import annotations

import numpy as np

from pipeline.config import CLASSES


def empty_confusion() -> np.ndarray:
    return np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)


def update_confusion(matrix: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if len(y_true) == 0:
        return
    np.add.at(matrix, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)


def scores_from_confusion(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=np.float64)
    total = matrix.sum()
    correct = np.trace(matrix)
    accuracy = float(correct / total) if total else 0.0
    per_class = {}
    f1s = []
    recalls = []
    for index, name in enumerate(CLASSES):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = int(matrix[index, :].sum())
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        f1s.append(f1)
        recalls.append(recall)
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "macro_recall": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "generator_recall": per_class["generator"]["recall"],
        "per_class": per_class,
        "n": int(total),
    }


class StratifiedReservoir:
    """Algorithm-R sample that keeps up to `per_class` rows of each label."""

    def __init__(self, per_class: int, width: int, seed: int, extra: int = 0):
        self.per_class = per_class
        self.width = width
        self.extra = extra
        self.rng = np.random.default_rng(seed)
        self.seen = {c: 0 for c in range(3)}
        self.filled = {c: 0 for c in range(3)}
        self.X = {c: np.empty((per_class, width), np.float32) for c in range(3)}
        self.time = {c: np.empty(per_class, np.int64) for c in range(3)}
        self.extra_store = {c: np.empty((per_class, extra), np.float32) for c in range(3)} if extra else None

    def add(self, x: np.ndarray, y: int, time_s: int, extra: np.ndarray | None = None) -> None:
        if y < 0 or y > 2:
            return
        self.seen[y] += 1
        slot = self.filled[y]
        take = None
        if slot < self.per_class:
            take = slot
            self.filled[y] = slot + 1
        else:
            draw = int(self.rng.integers(0, self.seen[y]))
            if draw < self.per_class:
                take = draw
        if take is None:
            return
        self.X[y][take] = x
        self.time[y][take] = time_s
        if self.extra_store is not None and extra is not None:
            self.extra_store[y][take] = extra

    def add_many(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_s: np.ndarray,
        extra: np.ndarray | None = None,
    ) -> None:
        for i in range(len(y)):
            self.add(X[i], int(y[i]), int(time_s[i]), None if extra is None else extra[i])

    def stack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        parts_x = []
        parts_y = []
        parts_t = []
        parts_e = []
        for label in range(3):
            n = self.filled[label]
            if n == 0:
                continue
            parts_x.append(self.X[label][:n])
            parts_y.append(np.full(n, label, np.int8))
            parts_t.append(self.time[label][:n])
            if self.extra_store is not None:
                parts_e.append(self.extra_store[label][:n])
        if not parts_x:
            empty_x = np.empty((0, self.width), np.float32)
            empty_e = np.empty((0, self.extra), np.float32) if self.extra else None
            return empty_x, np.empty(0, np.int8), np.empty(0, np.int64), empty_e
        extra_out = np.vstack(parts_e) if parts_e else None
        return np.vstack(parts_x), np.concatenate(parts_y), np.concatenate(parts_t), extra_out


class WindowReservoir:
    """Stratified sample of raw windows plus the tabular feature row at the window end."""

    def __init__(self, per_class: int, length: int, n_features: int, seed: int):
        self.per_class = per_class
        self.length = length
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)
        self.seen = {c: 0 for c in range(3)}
        self.filled = {c: 0 for c in range(3)}
        self.X = {c: np.empty((per_class, length, 2), np.float32) for c in range(3)}
        self.F = {c: np.empty((per_class, n_features), np.float32) for c in range(3)}

    def add(self, window: np.ndarray, features: np.ndarray, y: int) -> None:
        if y < 0 or y > 2:
            return
        self.seen[y] += 1
        slot = self.filled[y]
        take = None
        if slot < self.per_class:
            take = slot
            self.filled[y] = slot + 1
        else:
            draw = int(self.rng.integers(0, self.seen[y]))
            if draw < self.per_class:
                take = draw
        if take is not None:
            self.X[y][take] = window
            self.F[y][take] = features

    def stack(self) -> tuple[np.ndarray, np.ndarray]:
        windows, _features, labels = self._parts()
        return windows, labels

    def stack_features(self) -> tuple[np.ndarray, np.ndarray]:
        _windows, features, labels = self._parts()
        return features, labels

    def _parts(self):
        xs, fs, ys = [], [], []
        for label in range(3):
            n = self.filled[label]
            if n == 0:
                continue
            xs.append(self.X[label][:n])
            fs.append(self.F[label][:n])
            ys.append(np.full(n, label, np.int8))
        if not xs:
            return (
                np.empty((0, self.length, 2), np.float32),
                np.empty((0, self.n_features), np.float32),
                np.empty(0, np.int8),
            )
        return np.concatenate(xs), np.concatenate(fs), np.concatenate(ys)
