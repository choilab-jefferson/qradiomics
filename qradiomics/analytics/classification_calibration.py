"""Rigour metrics for binary classifiers: discrimination uncertainty +
calibration quality. Closes QA checks C06 (discrimination with CI) and
C05 (calibration: Brier + ECE) for any reproduction that produces pooled
out-of-fold predicted probabilities.

Also provides a generic out-of-fold Youden's-J threshold-calibration
utility (``oof_proba`` + ``youden_threshold``) for picking a
decision-boundary that is not the default 0.5 — useful whenever the
outcome is imbalanced (0.5 systematically favours the majority class even
when the ranking, i.e. AUC, is good).
"""
from __future__ import annotations

import numpy as np

__all__ = ["brier_ece", "auc_ci", "youden_threshold", "oof_proba"]


def brier_ece(y_true, prob, n_bins: int = 10) -> tuple[float, float]:
    """Brier score + equal-width-bin Expected Calibration Error."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    if len(y) == 0:
        return float("nan"), float("nan")
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        hi_inclusive = i == n_bins - 1
        m = (p >= bins[i]) & (p <= bins[i + 1] if hi_inclusive else p < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return brier, float(ece)


def auc_ci(y_true, prob, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """ROC-AUC + percentile bootstrap 95% CI on pooled predictions."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    p = np.asarray(prob)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    if not boots:
        return auc, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)


def youden_threshold(y_true, y_proba) -> float:
    """Decision threshold maximizing Youden's J = sensitivity + specificity - 1.

    The default 0.5 threshold is a poor decision boundary under class
    imbalance: probabilities can be well-ranked (good AUC) without being
    calibrated to the class prevalence, so 0.5 systematically over-predicts
    the majority class. Youden's J picks the point on the ROC curve
    furthest from the diagonal (chance) line — a discrimination-optimal,
    prevalence-agnostic cutoff.

    Parameters
    ----------
    y_true : array-like of {0,1}
    y_proba : array-like of predicted probability of the positive class

    Returns
    -------
    float — the probability threshold at which J = tpr - fpr is maximised.
    """
    from sklearn.metrics import roc_curve

    y = np.asarray(y_true)
    p = np.asarray(y_proba, dtype=float)
    fpr, tpr, thresholds = roc_curve(y, p)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def oof_proba(estimator, X, y, n_splits: int = 10, random_state: int = 42) -> np.ndarray:
    """Out-of-fold predicted probabilities via a fresh clone refit per fold.

    Intended ONLY for picking a decision threshold (e.g. via
    ``youden_threshold``) — never for reporting a discrimination/calibration
    metric. Reusing these folds for threshold selection does not leak into
    metrics computed elsewhere (e.g. each fold's own held-out AUC, or a
    separate external/held-out test evaluation) because the threshold is a
    single scalar decision boundary, not a fitted parameter of the scoring
    model itself.

    Parameters
    ----------
    estimator : an unfitted sklearn-compatible estimator/Pipeline exposing
        ``predict_proba``. Cloned (via ``sklearn.base.clone``) once per fold
        so no state leaks across folds.
    X : feature matrix (n_samples, n_features) — DataFrame or array.
    y : binary outcome (0/1) — Series or array.
    n_splits : number of stratified folds (default 10).
    random_state : fold-split reproducibility seed.

    Returns
    -------
    np.ndarray of shape (n_samples,) with the out-of-fold P(y=1) for every
    sample (each sample's prediction comes from a model that never saw it
    during fitting).
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_arr = np.asarray(y)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(y_arr), np.nan, dtype=float)
    for tr_idx, va_idx in cv.split(X_arr, y_arr):
        fold_est = clone(estimator)
        fold_est.fit(X_arr[tr_idx], y_arr[tr_idx])
        oof[va_idx] = fold_est.predict_proba(X_arr[va_idx])[:, 1]
    return oof
