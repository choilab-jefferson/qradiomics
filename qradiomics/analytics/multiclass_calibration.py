"""Rigour metrics for MULTI-class classifiers: discrimination uncertainty +
calibration quality. The multi-class companion to ``classification_calibration``
(which is binary-only). Closes QA checks C06 (discrimination with CI) and C05
(calibration: Brier + ECE) for any reproduction whose endpoint has >2 classes.

All functions accept a hard-label prediction or a (n_samples, n_classes)
probability matrix as appropriate, and never raise on degenerate input — they
return NaN so a caller can report "not estimable" rather than crash.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "accuracy_ci",
    "multiclass_brier",
    "multiclass_ece",
    "macro_auc_ovr_ci",
]


def accuracy_ci(y_true, y_pred, n_boot: int = 2000, seed: int = 42):
    """Accuracy + percentile bootstrap 95% CI (resample test rows with replacement)."""
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    acc = float((y == p).mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = (y[idx] == p[idx]).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return acc, float(lo), float(hi)


def multiclass_brier(y_true, proba, classes=None) -> float:
    """Multi-class Brier score = mean_i sum_k (p_ik - onehot_ik)^2.

    Range [0, 2]; 0 is perfect. ``proba`` is (n_samples, n_classes) aligned to
    ``classes`` (defaults to sorted unique of y_true).
    """
    y = np.asarray(y_true)
    P = np.asarray(proba, dtype=float)
    if P.ndim != 2 or len(y) == 0:
        return float("nan")
    if classes is None:
        classes = np.sort(np.unique(y))
    classes = np.asarray(classes)
    onehot = (y[:, None] == classes[None, :]).astype(float)
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def multiclass_ece(y_true, proba, classes=None, n_bins: int = 10) -> float:
    """Top-label Expected Calibration Error (equal-width confidence bins).

    Confidence = max predicted probability; correctness = top-label == truth.
    """
    y = np.asarray(y_true)
    P = np.asarray(proba, dtype=float)
    if P.ndim != 2 or len(y) == 0:
        return float("nan")
    if classes is None:
        classes = np.sort(np.unique(y))
    classes = np.asarray(classes)
    conf = P.max(axis=1)
    pred = classes[P.argmax(axis=1)]
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    ece = 0.0
    for i in range(n_bins):
        hi_inclusive = i == n_bins - 1
        m = (conf >= bins[i]) & (conf <= bins[i + 1] if hi_inclusive else conf < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(float(conf[m].mean()) - float(correct[m].mean()))
    return float(ece)


def macro_auc_ovr_ci(y_true, proba, classes=None, n_boot: int = 2000, seed: int = 42):
    """Macro one-vs-rest ROC-AUC + percentile bootstrap 95% CI.

    Averages per-class OvR AUC over classes present in the resample; returns NaN
    when no class has both positives and negatives.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    P = np.asarray(proba, dtype=float)
    if P.ndim != 2 or len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    if classes is None:
        classes = np.sort(np.unique(y))
    classes = np.asarray(classes)

    def _macro(yy, PP):
        scores = []
        for k, c in enumerate(classes):
            yk = (yy == c).astype(int)
            if yk.sum() == 0 or yk.sum() == len(yk):
                continue
            scores.append(roc_auc_score(yk, PP[:, k]))
        return float(np.mean(scores)) if scores else float("nan")

    auc = _macro(y, P)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = _macro(y[idx], P[idx])
        if not np.isnan(v):
            boots.append(v)
    if not boots:
        return auc, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)
