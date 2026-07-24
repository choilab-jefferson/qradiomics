"""Decision-curve analysis (net benefit) for binary risk predictions.

Closes QA check C22 (clinical utility). Net benefit at a risk threshold
``p_t`` is::

    NB = TP/n - (FP/n) * p_t/(1 - p_t)

compared against the two default strategies treat-all and treat-none.
A model is clinically useful at ``p_t`` when its net benefit exceeds both.
Operates on already-produced predicted probabilities, so it is a cheap
add-on wherever a calibrated/pooled prediction exists.
"""
from __future__ import annotations

import numpy as np

__all__ = ["net_benefit", "decision_curve_summary"]


def net_benefit(y_true, prob, threshold: float) -> tuple[float, float]:
    """Return (model_net_benefit, treat_all_net_benefit) at ``threshold``."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    n = len(y)
    if n == 0 or threshold <= 0.0 or threshold >= 1.0:
        return float("nan"), float("nan")
    pred_pos = p >= threshold
    tp = float(np.sum(pred_pos & (y == 1)))
    fp = float(np.sum(pred_pos & (y == 0)))
    w = threshold / (1.0 - threshold)
    nb_model = tp / n - (fp / n) * w
    prev = float(np.mean(y))
    nb_all = prev - (1.0 - prev) * w
    return float(nb_model), float(nb_all)


def decision_curve_summary(y_true, prob,
                           thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)) -> dict:
    """Net benefit at several thresholds plus whether the model beats the
    best default strategy (treat-all / treat-none, the latter NB = 0)."""
    out: dict[str, dict] = {}
    for t in thresholds:
        nb_m, nb_all = net_benefit(y_true, prob, t)
        best_default = max(nb_all, 0.0)
        out[f"{t:.2f}"] = {
            "net_benefit": round(nb_m, 4),
            "treat_all": round(nb_all, 4),
            "beats_default": bool(nb_m > best_default),
        }
    return out
