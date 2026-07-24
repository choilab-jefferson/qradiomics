"""Multi-model cross-validation benchmark for binary classification.

Usage::

    from qradiomics.classification import cross_val_benchmark
    results = cross_val_benchmark(X, y, models=["LR","RF","XGB"], cv=10, random_state=42)
    print(results.summary())
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from qradiomics.classification.registry import DEFAULT_MODELS, MODEL_REGISTRY, build_model


@dataclass
class ModelResult:
    model: str
    cv_auc_mean: float
    cv_auc_std: float
    cv_ap_mean: float
    best_params: dict
    oof_y_true: np.ndarray = field(repr=False)
    oof_y_score: np.ndarray = field(repr=False)

    @property
    def oof_auc(self) -> float:
        return float(roc_auc_score(self.oof_y_true, self.oof_y_score))

    @property
    def oof_ap(self) -> float:
        return float(average_precision_score(self.oof_y_true, self.oof_y_score))

    @property
    def oof_brier(self) -> float:
        return float(brier_score_loss(self.oof_y_true, self.oof_y_score))


@dataclass
class BenchmarkResult:
    model_results: list[ModelResult]
    cv_folds: int
    random_state: int
    n_train: int
    n_features: int

    def summary(self) -> pd.DataFrame:
        rows = []
        for r in sorted(self.model_results, key=lambda x: x.oof_auc, reverse=True):
            rows.append({
                "model":        r.model,
                "oof_auc":      round(r.oof_auc, 4),
                "oof_ap":       round(r.oof_ap, 4),
                "oof_brier":    round(r.oof_brier, 4),
                "cv_auc_mean":  round(r.cv_auc_mean, 4),
                "cv_auc_std":   round(r.cv_auc_std, 4),
            })
        return pd.DataFrame(rows)

    def best(self) -> ModelResult:
        return max(self.model_results, key=lambda r: r.oof_auc)


def cross_val_benchmark(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    models: Sequence[str] | None = None,
    cv: int = 10,
    inner_cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    hpo: str = "grid",
    optuna_trials: int = 20,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run nested CV benchmark over multiple models.

    Parameters
    ----------
    X : feature matrix (n_samples, n_features)
    y : binary outcome (0/1)
    models : list of model keys from MODEL_REGISTRY; default = DEFAULT_MODELS
        (opt-in-only models such as "TPOT" are excluded from the default —
        request them explicitly)
    cv : outer folds
    inner_cv : inner folds for HPO
    random_state : reproducibility seed
    n_jobs : parallelism for GridSearchCV
    hpo : 'grid' or 'optuna'
    optuna_trials : trials when hpo='optuna'
    verbose : print progress

    Returns
    -------
    BenchmarkResult with per-model OOF scores
    """
    if models is None:
        models = list(DEFAULT_MODELS)
    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown models: {unknown}. Available: {list(MODEL_REGISTRY)}")

    X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)

    outer_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    inner_cv_obj = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state + 1)

    results: list[ModelResult] = []

    for model_name in models:
        if verbose:
            print(f"  [{model_name}] fitting {cv}-fold nested CV ...", flush=True)

        pipe, grid = build_model(model_name, random_state=random_state)
        fold_aucs: list[float] = []
        fold_aps: list[float] = []
        oof_scores = np.zeros(len(y_arr))
        best_params_list: list[dict] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(outer_cv.split(X_arr, y_arr)):
            X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
            y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

            if grid:
                if hpo == "optuna":
                    est, bp = _optuna_search(
                        pipe, grid, X_tr, y_tr,
                        inner=inner_cv_obj, n_jobs=n_jobs,
                        n_trials=optuna_trials, rs=random_state + fold_idx,
                    )
                else:
                    gs = GridSearchCV(pipe, grid, cv=inner_cv_obj,
                                      scoring="roc_auc", n_jobs=n_jobs, refit=True)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gs.fit(X_tr, y_tr)
                    est = gs.best_estimator_
                    bp = gs.best_params_
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X_tr, y_tr)
                est = pipe
                bp = {}

            best_params_list.append(bp)
            p = est.predict_proba(X_va)[:, 1]
            oof_scores[va_idx] = p
            fold_aucs.append(float(roc_auc_score(y_va, p)))
            fold_aps.append(float(average_precision_score(y_va, p)))

        # modal best params (most common across folds)
        best_params = _modal_params(best_params_list)

        results.append(ModelResult(
            model=model_name,
            cv_auc_mean=float(np.mean(fold_aucs)),
            cv_auc_std=float(np.std(fold_aucs)),
            cv_ap_mean=float(np.mean(fold_aps)),
            best_params=best_params,
            oof_y_true=y_arr,
            oof_y_score=oof_scores,
        ))

        if verbose:
            print(f"    AUC fold-mean={np.mean(fold_aucs):.3f}±{np.std(fold_aucs):.3f}  OOF={results[-1].oof_auc:.3f}")

    return BenchmarkResult(
        model_results=results,
        cv_folds=cv,
        random_state=random_state,
        n_train=len(y_arr),
        n_features=X_arr.shape[1],
    )


def _modal_params(param_list: list[dict]) -> dict:
    if not param_list:
        return {}
    from collections import Counter
    merged: dict[str, list] = {}
    for d in param_list:
        for k, v in d.items():
            merged.setdefault(k, []).append(v)
    return {k: Counter(vs).most_common(1)[0][0] for k, vs in merged.items()}


def _optuna_search(pipe, grid, X_tr, y_tr, *, inner, n_jobs, n_trials, rs):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _sample(trial, g):
        return {k: trial.suggest_categorical(k, v) for k, v in g.items()}

    def objective(trial):
        params = _sample(trial, grid)
        p = pipe.set_params(**params)
        from sklearn.model_selection import cross_val_score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(p, X_tr, y_tr, cv=inner,
                                     scoring="roc_auc", n_jobs=n_jobs)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=rs),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best_params = study.best_params
    pipe.set_params(**best_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)
    return pipe, best_params
