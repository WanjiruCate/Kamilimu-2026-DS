"""Many-to-one RNN and LSTM classifiers in NumPy.

sktime ships `SimpleRNNClassifier` / `LSTMFCNClassifier` (TensorFlow) and
`SimpleRNNClassifierTorch` / `LSTMFCNClassifierTorch` (PyTorch). Those stacks
are the right choice on a GPU workstation. They are a poor fit for an 8 GB
classroom laptop and for a small production image, so this module trains the
same two architectures directly. The deployed API can score either set of
weights with no deep-learning runtime installed.

Both nets read a window and emit one class from the final hidden state.
"""

from __future__ import annotations

import numpy as np

from pipeline.config import RANDOM_STATE


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _clip(grads: list[np.ndarray], max_norm: float = 5.0) -> list[np.ndarray]:
    total = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-8)
        return [g * scale for g in grads]
    return grads


class _Adam:
    def __init__(self, shapes: list[tuple], lr: float):
        self.lr = lr
        self.m = [np.zeros(shape) for shape in shapes]
        self.v = [np.zeros(shape) for shape in shapes]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for param, grad, moment, second in zip(params, grads, self.m, self.v):
            moment *= b1
            moment += (1 - b1) * grad
            second *= b2
            second += (1 - b2) * (grad * grad)
            moment_hat = moment / (1 - b1**self.t)
            second_hat = second / (1 - b2**self.t)
            param -= self.lr * moment_hat / (np.sqrt(second_hat) + eps)


class SequenceClassifier:
    def __init__(self, kind: str, n_features: int, n_hidden: int, n_classes: int = 3, seed: int = RANDOM_STATE):
        if kind not in {"rnn", "lstm"}:
            raise ValueError(kind)
        self.kind = kind
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        rng = np.random.default_rng(seed)
        scale = 0.08
        if kind == "rnn":
            self.Wxh = rng.normal(0, scale, size=(n_features, n_hidden))
            self.Whh = rng.normal(0, scale, size=(n_hidden, n_hidden))
            self.bh = np.zeros(n_hidden)
        else:
            self.W = rng.normal(0, scale, size=(n_hidden + n_features, 4 * n_hidden))
            self.b = np.zeros(4 * n_hidden)
            # Forget-gate bias starts open so early training can carry state.
            self.b[n_hidden : 2 * n_hidden] = 1.0
        self.Wy = rng.normal(0, scale, size=(n_hidden, n_classes))
        self.by = np.zeros(n_classes)
        self.mean_ = np.zeros(n_features)
        self.std_ = np.ones(n_features)

    def fit_scaler(self, windows: np.ndarray) -> None:
        flat = windows.reshape(-1, windows.shape[-1]).astype(np.float64)
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0

    def _scale(self, windows: np.ndarray) -> np.ndarray:
        return (windows.astype(np.float64) - self.mean_) / self.std_

    def _params(self) -> list[np.ndarray]:
        if self.kind == "rnn":
            return [self.Wxh, self.Whh, self.bh, self.Wy, self.by]
        return [self.W, self.b, self.Wy, self.by]

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        scaled = self._scale(windows)
        logits = self._forward(scaled)
        return _softmax(logits)

    def predict(self, windows: np.ndarray) -> np.ndarray:
        return self.predict_proba(windows).argmax(axis=1).astype(np.int8)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "rnn":
            return _rnn_forward(X, self.Wxh, self.Whh, self.bh, self.Wy, self.by)[0]
        return _lstm_forward(X, self.W, self.b, self.Wy, self.by, self.n_hidden)[0]

    def train(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        windows_val: np.ndarray,
        labels_val: np.ndarray,
        epochs: int = 12,
        batch_size: int = 128,
        lr: float = 0.01,
        seed: int = RANDOM_STATE,
    ) -> list[float]:
        self.fit_scaler(windows)
        X = self._scale(windows)
        Xv = self._scale(windows_val) if len(windows_val) else np.empty((0, 1, self.n_features))
        y = labels.astype(np.int64)
        yv = labels_val.astype(np.int64)
        adam = _Adam([p.shape for p in self._params()], lr=lr)
        rng = np.random.default_rng(seed)
        history = []
        best_loss = np.inf
        best = [p.copy() for p in self._params()]
        n = len(X)
        for _epoch in range(epochs):
            order = rng.permutation(n)
            total = 0.0
            seen = 0
            for start in range(0, n, batch_size):
                batch = order[start : start + batch_size]
                loss, grads = self._loss_grads(X[batch], y[batch])
                grads = _clip(grads)
                adam.step(self._params(), grads)
                total += loss * len(batch)
                seen += len(batch)
            history.append(float(total / max(seen, 1)))
            val_loss = history[-1]
            if len(Xv):
                val_loss = self._mean_loss(Xv, yv)
            if val_loss < best_loss:
                best_loss = val_loss
                best = [p.copy() for p in self._params()]
        for param, saved in zip(self._params(), best):
            param[...] = saved
        return history

    def _mean_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        logits = self._forward(X)
        probs = _softmax(logits)
        return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())

    def _loss_grads(self, X: np.ndarray, y: np.ndarray):
        if self.kind == "rnn":
            return _rnn_grads(X, y, self.Wxh, self.Whh, self.bh, self.Wy, self.by)
        return _lstm_grads(X, y, self.W, self.b, self.Wy, self.by, self.n_hidden)


def _rnn_forward(X, Wxh, Whh, bh, Wy, by):
    batch, steps, _features = X.shape
    hidden = Whh.shape[0]
    h = np.zeros((batch, hidden))
    caches = []
    for t in range(steps):
        h_prev = h
        a = X[:, t, :] @ Wxh + h_prev @ Whh + bh
        h = np.tanh(a)
        caches.append((h_prev, X[:, t, :], a, h))
    logits = h @ Wy + by
    return logits, caches


def _rnn_grads(X, y, Wxh, Whh, bh, Wy, by):
    logits, caches = _rnn_forward(X, Wxh, Whh, bh, Wy, by)
    probs = _softmax(logits)
    loss = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())
    dlogits = probs
    dlogits[np.arange(len(y)), y] -= 1.0
    dlogits /= len(y)
    h_last = caches[-1][3]
    dWy = h_last.T @ dlogits
    dby = dlogits.sum(axis=0)
    dh = dlogits @ Wy.T
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros_like(bh)
    for h_prev, x_t, a, h in reversed(caches):
        da = dh * (1.0 - h * h)
        dWxh += x_t.T @ da
        dWhh += h_prev.T @ da
        dbh += da.sum(axis=0)
        dh = da @ Whh.T
    return loss, [dWxh, dWhh, dbh, dWy, dby]


def _lstm_forward(X, W, b, Wy, by, hidden):
    batch, steps, _features = X.shape
    h = np.zeros((batch, hidden))
    c = np.zeros((batch, hidden))
    caches = []
    for t in range(steps):
        z = np.concatenate([h, X[:, t, :]], axis=1)
        gate = z @ W + b
        i = _sigmoid(gate[:, 0:hidden])
        f = _sigmoid(gate[:, hidden : 2 * hidden])
        g = np.tanh(gate[:, 2 * hidden : 3 * hidden])
        o = _sigmoid(gate[:, 3 * hidden : 4 * hidden])
        c_prev = c
        c = f * c_prev + i * g
        tanh_c = np.tanh(c)
        h = o * tanh_c
        caches.append((z, i, f, g, o, c_prev, tanh_c, h))
    logits = h @ Wy + by
    return logits, caches


def _lstm_grads(X, y, W, b, Wy, by, hidden):
    logits, caches = _lstm_forward(X, W, b, Wy, by, hidden)
    probs = _softmax(logits)
    loss = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())
    dlogits = probs
    dlogits[np.arange(len(y)), y] -= 1.0
    dlogits /= len(y)
    h_last = caches[-1][7]
    dWy = h_last.T @ dlogits
    dby = dlogits.sum(axis=0)
    dh_next = dlogits @ Wy.T
    dc_next = np.zeros_like(h_last)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    for z, i, f, g, o, c_prev, tanh_c, h in reversed(caches):
        do = dh_next * tanh_c
        dc = dh_next * o * (1.0 - tanh_c * tanh_c) + dc_next
        di = dc * g
        df = dc * c_prev
        dg = dc * i
        dc_prev = dc * f
        di_pre = di * i * (1.0 - i)
        df_pre = df * f * (1.0 - f)
        dg_pre = dg * (1.0 - g * g)
        do_pre = do * o * (1.0 - o)
        da = np.concatenate([di_pre, df_pre, dg_pre, do_pre], axis=1)
        dW += z.T @ da
        db += da.sum(axis=0)
        dz = da @ W.T
        dh_next = dz[:, :hidden]
        dc_next = dc_prev
    return loss, [dW, db, dWy, dby]


def save_sequence(model: SequenceClassifier, path) -> None:
    payload = {
        "kind": np.array(model.kind),
        "n_features": np.int64(model.n_features),
        "n_hidden": np.int64(model.n_hidden),
        "n_classes": np.int64(model.n_classes),
        "mean": model.mean_,
        "std": model.std_,
        "Wy": model.Wy,
        "by": model.by,
    }
    if model.kind == "rnn":
        payload.update(Wxh=model.Wxh, Whh=model.Whh, bh=model.bh)
    else:
        payload.update(W=model.W, b=model.b)
    np.savez(path, **payload)


def load_sequence(path) -> SequenceClassifier:
    data = np.load(path)
    model = SequenceClassifier(
        kind=str(data["kind"]),
        n_features=int(data["n_features"]),
        n_hidden=int(data["n_hidden"]),
        n_classes=int(data["n_classes"]),
    )
    model.mean_ = data["mean"]
    model.std_ = data["std"]
    model.Wy = data["Wy"]
    model.by = data["by"]
    if model.kind == "rnn":
        model.Wxh = data["Wxh"]
        model.Whh = data["Whh"]
        model.bh = data["bh"]
    else:
        model.W = data["W"]
        model.b = data["b"]
    return model


def numerical_gradient_ok(kind: str = "lstm", seed: int = 0) -> bool:
    """Finite-difference check on one weight. Used by the unit test."""
    rng = np.random.default_rng(seed)
    model = SequenceClassifier(kind, n_features=2, n_hidden=3, n_classes=2, seed=seed)
    model.mean_ = np.zeros(2)
    model.std_ = np.ones(2)
    X = rng.normal(size=(4, 3, 2))
    y = rng.integers(0, 2, size=4)
    loss, grads = model._loss_grads(X, y)
    eps = 1e-5
    if kind == "lstm":
        param = model.W
        grad = grads[0]
        index = (1, 2)
    else:
        param = model.Wxh
        grad = grads[0]
        index = (0, 1)
    original = param[index]
    param[index] = original + eps
    plus, _ = model._loss_grads(X, y)
    param[index] = original - eps
    minus, _ = model._loss_grads(X, y)
    param[index] = original
    numeric = (plus - minus) / (2 * eps)
    return abs(numeric - grad[index]) < 1e-4 and np.isfinite(loss)
