"""Rigour metrics for survival models: discrimination uncertainty +
calibration. Closes QA checks C06 (discrimination with CI) and C05
(calibration quality) for any Cox/risk-score reproduction.

* :func:`concordance_ci` — Harrell's c-index with a percentile bootstrap
  95% CI on pooled out-of-fold risk scores.
* :func:`fold_integrated_brier` — Integrated Brier Score for one CV fold
  (best-effort: returns NaN if the time grid / follow-up ranges make the
  estimate ill-defined, so it never breaks a reproduction run).

The helper is deliberately library-guarded: lifelines provides the
concordance index, scikit-survival the IBS. Both are optional at import
time so unrelated CLI paths do not pay the import cost.
"""
from __future__ import annotations

import numpy as np

__all__ = ["concordance_ci", "fold_integrated_brier"]


def concordance_ci(times, risk, events, n_boot: int = 2000, seed: int = 42):
    """Harrell's c-index + percentile bootstrap 95% CI.

    ``risk`` is a risk score where higher == higher hazard (e.g. Cox
    ``predict_partial_hazard``); internally negated so the concordance is
    computed against survival time correctly.
    """
    from lifelines.utils import concordance_index

    T = np.asarray(times, dtype=float)
    R = np.asarray(risk, dtype=float)
    E = np.asarray(events, dtype=float)
    n = len(T)
    if n < 3 or E.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(concordance_index(T, -R, E))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.asarray(E)[idx].sum() < 2:
            continue
        try:
            boots.append(concordance_index(T[idx], -R[idx], E[idx]))
        except Exception:
            continue
    if not boots:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def fold_integrated_brier(cph, X_te, t_train, e_train, t_test, e_test) -> float:
    """Integrated Brier Score for one fold (best-effort; NaN on failure).

    Uses the fold's own Cox model to predict survival functions, then
    scikit-survival's IBS over an event-time grid restricted to the
    overlap of the train and test follow-up (sksurv requires test times to
    fall within the train range).
    """
    try:
        # Compatibility shim: scikit-survival <= 0.25 calls np.trapz, removed
        # in NumPy 2.x (renamed np.trapezoid). Restore it so IBS works under
        # either NumPy without forcing a downgrade.
        if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
            np.trapz = np.trapezoid  # type: ignore[attr-defined]
        from sksurv.metrics import integrated_brier_score
        from sksurv.util import Surv

        T_tr = np.asarray(t_train, dtype=float)
        E_tr = np.asarray(e_train, dtype=float)
        T_te = np.asarray(t_test, dtype=float)
        E_te = np.asarray(e_test, dtype=float)

        y_tr = Surv.from_arrays(E_tr.astype(bool), T_tr)
        y_te = Surv.from_arrays(E_te.astype(bool), T_te)

        # Time grid built from quantiles of the *overlap* of the train and
        # test follow-up (sksurv requires evaluation times strictly inside the
        # test range and where the train censoring survival is estimable).
        # Using time quantiles over the overlap (not just test event times)
        # is far more robust to sparse per-fold events than the event-time
        # grid, which returned NaN whenever a fold's events clustered at the
        # boundary.
        lo = max(float(T_tr.min()), float(T_te.min()))
        hi = min(float(T_tr.max()), float(T_te.max()))
        if not (hi > lo):
            return float("nan")
        pooled = np.concatenate([T_tr, T_te])
        grid = np.quantile(pooled, np.linspace(0.10, 0.90, 10))
        grid = np.unique(grid[(grid > lo) & (grid < hi)])
        if len(grid) < 2:
            return float("nan")

        sf = cph.predict_survival_function(X_te, times=grid)  # index=times, cols=samples
        est = np.asarray(sf.T.values, dtype=float)            # samples x times
        return float(integrated_brier_score(y_tr, y_te, est, grid))
    except Exception:
        return float("nan")
