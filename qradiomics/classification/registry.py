"""Unified model registry for binary classification benchmarks.

Each entry in MODEL_REGISTRY maps a short model key to a builder function
``(random_state: int) -> (Pipeline, param_grid: dict)``.  A param_grid of
``{}`` means no HPO is performed (the model manages its own tuning, e.g.
FLAML).

Usage::

    from qradiomics.classification import build_model
    pipe, grid = build_model("RF", random_state=42)
    gs = GridSearchCV(pipe, grid, ...)
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def _lr(rs: int) -> tuple[Pipeline, dict[str, list[Any]]]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear", class_weight="balanced",
            max_iter=2000, random_state=rs,
        )),
    ])
    return pipe, {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l1", "l2"]}


def _svm(rs: int) -> tuple[Pipeline, dict[str, list[Any]]]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=rs)),
    ])
    return pipe, {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__gamma": [1e-4, 1e-3, 1e-2, 1e-1, "scale", "auto"],
    }


def _extra_trees(rs: int) -> tuple[Pipeline, dict[str, list[Any]]]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(
            n_estimators=400, class_weight="balanced", random_state=rs, n_jobs=1,
        )),
    ])
    return pipe, {
        "clf__max_depth": [None, 6, 12],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features": ["sqrt", 0.3],
    }


def _rf(rs: int) -> tuple[Pipeline, dict[str, list[Any]]]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=rs, n_jobs=1,
        )),
    ])
    return pipe, {
        "clf__max_depth": [None, 6, 12],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features": ["sqrt", 0.3],
    }


def _gbm(rs: int) -> tuple[Pipeline, dict[str, list[Any]]]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=rs)),
    ])
    return pipe, {
        "clf__max_depth": [2, 3, 5],
        "clf__learning_rate": [0.03, 0.1, 0.2],
        "clf__subsample": [0.7, 1.0],
    }


def _mlp(rs: int) -> tuple[Pipeline, dict[str, list[Any]]]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(max_iter=500, early_stopping=True, random_state=rs)),
    ])
    return pipe, {
        "clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [1e-3, 5e-4],
    }


def _xgb(rs: int) -> tuple[Pipeline, dict[str, list[Any]]]:
    try:
        from xgboost import XGBClassifier  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("pip install xgboost") from exc
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=rs, n_estimators=400, n_jobs=1,
        )),
    ])
    return pipe, {
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.03, 0.1],
        "clf__reg_lambda": [1.0, 5.0],
        "clf__scale_pos_weight": [1.0, 2.0],
    }


def _lgbm(rs: int) -> tuple[Pipeline, dict[str, list[Any]]]:
    try:
        from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("pip install lightgbm") from exc
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(
            objective="binary", class_weight="balanced",
            random_state=rs, n_estimators=400, n_jobs=1, verbose=-1,
        )),
    ])
    return pipe, {
        "clf__max_depth": [-1, 5, 8],
        "clf__num_leaves": [15, 31, 63],
        "clf__learning_rate": [0.03, 0.1],
        "clf__reg_lambda": [0.0, 1.0],
    }


def _flaml(rs: int) -> tuple[Pipeline, dict[str, list[Any]]]:
    try:
        from flaml import AutoML  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("pip install flaml") from exc
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _FLAMLWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, time_budget: int = 60, random_state: int = 42):
            self.time_budget = time_budget
            self.random_state = random_state

        def fit(self, X, y):
            self._automl = AutoML()
            self._automl.fit(
                X, y, task="classification", metric="roc_auc",
                time_budget=self.time_budget, seed=self.random_state, verbose=0,
            )
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return self._automl.predict(X)

        def predict_proba(self, X):
            p = self._automl.predict_proba(X)
            return np.column_stack([1 - p, p]) if p.ndim == 1 else p

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", _FLAMLWrapper(time_budget=60, random_state=rs)),
    ])
    return pipe, {}


def _tpot(rs: int) -> tuple[Pipeline, dict[str, list[Any]]]:
    """TPOT AutoML (genetic-programming pipeline search), new Dask-based API.

    tpot>=1.1 is a ground-up rewrite of the legacy (<=0.12) DEAP-based TPOT —
    the ``template=``/``generations=``-at-top-level API of the old package no
    longer exists. This builder uses the current ``search_space=`` /
    ``scorers=`` API (``tpot.search_spaces.pipelines.SequentialPipeline``
    over selectors -> transformers -> classifiers).

    Known reproducibility limitation (verified empirically, not theoretical):
    TPOT's genetic search is **not** reproducible run-to-run even with a
    fixed ``random_state`` and ``n_jobs=1``. Internally it gathers fitness
    results via ``dask.distributed.as_completed()``, so which population
    member is available when the evolver next calls ``rng.choice()`` depends
    on wall-clock completion order, not just the seed. Two identical-input
    runs can converge on different champion pipelines (observed AUC delta of
    ~0.01, occasionally larger). The only honest mitigation is to repeat the
    fit N times with different seeds and report mean +/- std rather than a
    single number — see ``--tpot-repeats`` in ``qr bench`` / ``qr ml
    classify``, which special-cases this model for that reason.
    """
    try:
        import tpot as _tpot_pkg  # type: ignore[import-not-found]
        from tpot import TPOTClassifier  # type: ignore[import-not-found]
        from tpot.search_spaces.pipelines import SequentialPipeline  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("pip install qradiomics[automl]") from exc
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _TPOTWrapper(BaseEstimator, ClassifierMixin):
        """sklearn-compatible wrapper around tpot.TPOTClassifier.

        Bounded by both ``generations`` and ``max_time_mins`` so a single
        fit (e.g. inside nested outer CV) cannot run away — whichever limit
        is hit first stops the search.
        """

        def __init__(self, generations: int = 10, population_size: int = 30,
                     cv: int = 10, max_time_mins: float = 10.0,
                     n_jobs: int = 1, random_state: int = 42):
            self.generations = generations
            self.population_size = population_size
            self.cv = cv
            self.max_time_mins = max_time_mins
            self.n_jobs = n_jobs
            self.random_state = random_state

        def fit(self, X, y):
            search_space = SequentialPipeline([
                _tpot_pkg.config.get_search_space("selectors_classification"),
                _tpot_pkg.config.get_search_space("all_transformers"),
                _tpot_pkg.config.get_search_space("classifiers"),
            ])
            self._automl = TPOTClassifier(
                search_space=search_space,
                generations=self.generations,
                population_size=self.population_size,
                cv=self.cv,
                max_time_mins=int(self.max_time_mins),
                scorers=["roc_auc"],
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=0,
            )
            self._automl.fit(X, y)
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return self._automl.predict(X)

        def predict_proba(self, X):
            p = self._automl.predict_proba(X)
            return np.column_stack([1 - p, p]) if p.ndim == 1 else p

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", _TPOTWrapper(random_state=rs)),
    ])
    return pipe, {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ModelBuilder = Callable[[int], tuple[Pipeline, dict[str, list[Any]]]]

MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "LR":         _lr,
    "SVM":        _svm,
    "ExtraTrees": _extra_trees,
    "RF":         _rf,
    "GBM":        _gbm,
    "MLP":        _mlp,
    "XGB":        _xgb,
    "LGBM":       _lgbm,
    "FLAML":      _flaml,
    "TPOT":       _tpot,
}

# Models that are expensive/non-deterministic enough that they must be
# requested explicitly (--models TPOT) and are never pulled in silently by
# --all-models or the library-level "all models" default.
OPT_IN_ONLY_MODELS = ("TPOT",)

# Every registered model, literally — matches the plain-English meaning of
# the name. Public symbol; do not silently exclude models from it (that
# broke backward compatibility for consumers importing ALL_MODELS to mean
# "the full model set" — see DEFAULT_MODELS below for the opt-in-excluded
# variant used internally by --all-models / cross_val_benchmark(models=None)).
ALL_MODELS = tuple(MODEL_REGISTRY)

# The models pulled in by default: --all-models, and cross_val_benchmark's
# models=None default. Excludes opt-in-only models (TPOT) because they are
# materially more expensive and not reproducible from a single run.
DEFAULT_MODELS = tuple(k for k in MODEL_REGISTRY if k not in OPT_IN_ONLY_MODELS)

BASELINE_MODELS = ("LR", "SVM", "ExtraTrees")
EXTENDED_MODELS = ("LR", "SVM", "ExtraTrees", "RF", "GBM", "MLP", "XGB", "LGBM", "FLAML")


def build_model(name: str, random_state: int = 42) -> tuple[Pipeline, dict[str, list[Any]]]:
    """Return (pipeline, param_grid) for the named model."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](random_state)
