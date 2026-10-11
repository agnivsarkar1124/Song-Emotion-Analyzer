"""
Optional neural taste model: a small Keras "Wide & Deep" network.

    wide  = linear layer over every taste feature, including the hand-made
            emotion x energy interaction terms   -> memorises combos you rated
    deep  = 2-layer MLP over the main features only (tone, energy, emotions,
            themes)                              -> can find interactions nobody
                                                    hand-coded
    p(you like it) = sigmoid(wide + deep)

It is trained on your like / dislike / save / skip feedback with sample weights
(a save counts more than a like, a skip much less than a dislike) and blended
with the always-on ridge model in ``preferences.py``. Its share of the blend grows
with the amount of feedback, because a neural net trained on five ratings is
mostly noise and one trained on fifty is not.

Keras 3 is backend-agnostic, so this runs on TensorFlow *or* JAX - whichever is
installed (set ``KERAS_BACKEND`` to force one). If neither is installed, or
``SONG_MOOD_DISABLE_NEURAL=1`` is set, ``available()`` is False and the app
uses the ridge model alone. TensorFlow wheels: Apple Silicon needs Python
3.10-3.13; Intel Macs top out at TensorFlow 2.16.2 on Python 3.9-3.12.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import threading
from collections import OrderedDict

import numpy as np

MIN_RATINGS = 4          # rated songs needed (incl. at least one positive and one negative)
TRAIN_STEPS = 300        # full-batch gradient steps
LEARNING_RATE = 0.02
HIDDEN = (16, 8)
L2_WIDE = 0.01
L2_DEEP = 0.003
_CACHE_SIZE = 8
_TRAINABLE_BACKENDS = ("tensorflow", "jax", "torch")

_lock = threading.RLock()
_keras = None            # module once loaded, False if unavailable
_cache: "OrderedDict[str, WideDeepModel]" = OrderedDict()


def _load_keras():
    """Import Keras with a trainable backend, once. Returns the module or None."""
    global _keras
    if _keras is not None:
        return _keras or None
    with _lock:
        if _keras is not None:
            return _keras or None
        if os.environ.get("SONG_MOOD_DISABLE_NEURAL", "").strip() not in ("", "0"):
            _keras = False
            return None
        try:
            if "KERAS_BACKEND" not in os.environ:
                for backend in _TRAINABLE_BACKENDS:
                    if importlib.util.find_spec(backend) is not None:
                        os.environ["KERAS_BACKEND"] = backend
                        break
                else:
                    raise ImportError("no Keras backend installed")
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import keras
            if keras.backend.backend() not in _TRAINABLE_BACKENDS:
                raise ImportError(f"Keras backend {keras.backend.backend()!r} can't train")
            _keras = keras
        except Exception:      # ImportError, broken installs, unsupported CPUs ...
            _keras = False
    return _keras or None


def available() -> bool:
    return _load_keras() is not None


def backend_info() -> dict:
    """{'available': bool, 'label': str} for the UI."""
    keras = _load_keras()
    if keras is None:
        return {"available": False, "label": "not installed (linear taste model only)"}
    backend = keras.backend.backend()
    try:
        backend_version = __import__(backend).__version__
    except Exception:
        backend_version = ""
    return {"available": True, "label": f"Keras {keras.__version__} on {backend} {backend_version}".strip()}


class _Trainer:
    """One compiled Keras Wide & Deep network per input shape, reused for every fit.

    Building and tracing a fresh model costs ~1 s; resetting the weights of an
    already-traced one and refitting costs ~0.2 s, which keeps clicks snappy."""

    def __init__(self, n_wide: int, n_deep: int, seed: int = 0):
        keras = _load_keras()
        if keras is None:
            raise RuntimeError("Keras with a trainable backend is not installed")
        keras.utils.set_random_seed(seed)
        layers, reg = keras.layers, keras.regularizers

        wide_in = keras.Input(shape=(n_wide,), name="wide")
        deep_in = keras.Input(shape=(n_deep,), name="deep")
        wide = layers.Dense(1, kernel_regularizer=reg.L2(L2_WIDE), name="wide_linear")(wide_in)
        h = deep_in
        for i, units in enumerate(HIDDEN):
            h = layers.Dense(units, activation="relu", kernel_regularizer=reg.L2(L2_DEEP), name=f"deep_{i}")(h)
        deep = layers.Dense(1, kernel_regularizer=reg.L2(L2_DEEP), name="deep_out")(h)
        prob = layers.Activation("sigmoid", name="p_like")(layers.Add(name="wide_plus_deep")([wide, deep]))

        self.model = keras.Model(inputs=[wide_in, deep_in], outputs=prob, name="wide_and_deep_taste")
        self.model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="binary_crossentropy",
                           steps_per_execution=50)
        self.initial = self.model.get_weights()
        self.optimizer_built = False

    def fit(self, X_wide, X_deep, y, sample_weight, steps: int = TRAIN_STEPS) -> dict:
        """Reset to the initial weights, train full-batch, and return the learned weights.
        The tiny dataset is tiled so all ``steps`` gradient steps run as one Keras epoch."""
        self.model.set_weights(self.initial)
        if self.optimizer_built:
            # Fresh Adam state: zero the step counter and moment estimates, keep the learning rate.
            for var in self.model.optimizer.variables:
                if "learning_rate" not in (getattr(var, "path", "") or var.name):
                    var.assign(np.zeros(var.shape, dtype=var.dtype))
        X_wide, X_deep = np.asarray(X_wide, "float32"), np.asarray(X_deep, "float32")
        y01 = ((np.asarray(y, "float32") + 1.0) / 2.0).reshape(-1, 1)
        w = np.asarray(sample_weight, "float32")
        tile = lambda a: np.tile(a, (steps,) + (1,) * (a.ndim - 1))
        self.model.fit({"wide": tile(X_wide), "deep": tile(X_deep)}, tile(y01), sample_weight=tile(w),
                       batch_size=len(y01), epochs=1, shuffle=False, verbose=0)
        self.optimizer_built = True
        names = ["wide_linear"] + [f"deep_{i}" for i in range(len(HIDDEN))] + ["deep_out"]
        return {name: [np.array(a) for a in self.model.get_layer(name).get_weights()] for name in names}

    def predict(self, X_wide, X_deep) -> np.ndarray:
        """Keras forward pass (used by the tests to check the numpy one)."""
        out = self.model.predict({"wide": np.asarray(X_wide, "float32"), "deep": np.asarray(X_deep, "float32")},
                                 verbose=0)
        return 2.0 * np.asarray(out).reshape(-1) - 1.0


_trainers: dict = {}


def _trainer(n_wide: int, n_deep: int) -> _Trainer:
    key = (n_wide, n_deep)
    if key not in _trainers:
        _trainers[key] = _Trainer(n_wide, n_deep)
    return _trainers[key]


class WideDeepModel:
    """A trained Wide & Deep network, stored as plain numpy weights.

    Prediction is a numpy forward pass, so a cached model costs nothing to keep and is
    safe to share between Streamlit sessions. ``predict`` returns preferences in (-1, 1)."""

    def __init__(self, weights: dict):
        self.weights = weights

    @classmethod
    def train(cls, X_wide, X_deep, y, sample_weight) -> "WideDeepModel":
        with _lock:
            trainer = _trainer(np.shape(X_wide)[1], np.shape(X_deep)[1])
            return cls(trainer.fit(X_wide, X_deep, y, sample_weight))

    def predict(self, X_wide, X_deep) -> np.ndarray:
        X_wide, X_deep = np.asarray(X_wide, float), np.asarray(X_deep, float)
        kernel, bias = self.weights["wide_linear"]
        logit = X_wide @ kernel + bias
        h = X_deep
        for i in range(len(HIDDEN)):
            kernel, bias = self.weights[f"deep_{i}"]
            h = np.maximum(h @ kernel + bias, 0.0)
        kernel, bias = self.weights["deep_out"]
        logit = (logit + h @ kernel + bias).reshape(-1)
        return 2.0 / (1.0 + np.exp(-logit)) - 1.0       # = 2 * sigmoid(logit) - 1


def _fingerprint(*arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, "float64").round(6))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def trained_model(X_wide, X_deep, y, sample_weight):
    """Train (or fetch from cache) a Wide & Deep model for this exact feedback set.
    Returns None when Keras isn't available or there isn't enough signal yet."""
    if not available():
        return None
    y = np.asarray(y, float)
    if len(y) < MIN_RATINGS or not (y > 0).any() or not (y < 0).any():
        return None
    key = _fingerprint(X_wide, X_deep, y, sample_weight)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        model = WideDeepModel.train(X_wide, X_deep, y, sample_weight)
        _cache[key] = model
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
        return model


def cross_validate(X_wide, X_deep, y, sample_weight, ridge_fit, folds: int = 5, seed: int = 0) -> dict | None:
    """K-fold check of whether the neural model actually beats the linear one on *your* ratings.

    ``ridge_fit(X, y, c)`` must return linear weights. Scores are weighted accuracy of the
    predicted sign (like vs not) on held-out ratings. Returns None if there's too little data."""
    y = np.asarray(y, float)
    c = np.asarray(sample_weight, float)
    X_wide, X_deep = np.asarray(X_wide, float), np.asarray(X_deep, float)
    n = len(y)
    if n < 8 or not (y > 0).any() or not (y < 0).any():
        return None
    folds = max(2, min(folds, n))
    order = np.random.default_rng(seed).permutation(n)
    buckets = np.array_split(order, folds)
    hits = {"linear": 0.0, "neural": 0.0}
    total = 0.0
    neural_ok = available()
    for test in buckets:
        train = np.setdiff1d(order, test)
        if not (y[train] > 0).any() or not (y[train] < 0).any():
            continue
        w = ridge_fit(X_wide[train], y[train], c[train])
        lin = np.sign(X_wide[test] @ w)
        hits["linear"] += float(np.sum(c[test] * (lin == np.sign(y[test]))))
        if neural_ok:
            model = WideDeepModel.train(X_wide[train], X_deep[train], y[train], c[train])
            nn = np.sign(model.predict(X_wide[test], X_deep[test]))
            hits["neural"] += float(np.sum(c[test] * (nn == np.sign(y[test]))))
        total += float(np.sum(c[test]))
    if total == 0:
        return None
    return {
        "n": n,
        "folds": folds,
        "linear_accuracy": round(hits["linear"] / total, 3),
        "neural_accuracy": round(hits["neural"] / total, 3) if neural_ok else None,
    }
