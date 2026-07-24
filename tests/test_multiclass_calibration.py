"""Regression tests for the shared multi-class rigour metrics
(qradiomics.analytics.multiclass_calibration), the C05/C06 companion to the
binary classification_calibration module.

Locks the contract that every multi-class reproduction depends on:
  - perfect, calibrated predictions => Brier 0, ECE 0, AUC 1, acc 1
  - degenerate / empty input => NaN (never raises)
  - accuracy CI brackets the point estimate
"""
from __future__ import annotations

import math

import numpy as np

from qradiomics.analytics.multiclass_calibration import (
    accuracy_ci,
    macro_auc_ovr_ci,
    multiclass_brier,
    multiclass_ece,
)

CLASSES = np.array([0, 1, 2])


def _onehot(y):
    return (np.asarray(y)[:, None] == CLASSES[None, :]).astype(float)


def test_perfect_confident_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    P = _onehot(y)  # perfectly confident & correct
    assert multiclass_brier(y, P, CLASSES) == 0.0
    assert multiclass_ece(y, P, CLASSES) == 0.0
    auc, lo, hi = macro_auc_ovr_ci(y, P, CLASSES, n_boot=200, seed=1)
    assert auc == 1.0
    acc, alo, ahi = accuracy_ci(y, np.argmax(P, axis=1), n_boot=200, seed=1)
    assert acc == 1.0 and alo <= acc <= ahi


def test_brier_worst_case_bounds():
    # fully confident but always wrong => per-sample (1)^2+(1)^2 = 2
    y = np.array([0, 0, 0])
    P = np.tile([0.0, 1.0, 0.0], (3, 1))
    assert math.isclose(multiclass_brier(y, P, CLASSES), 2.0, rel_tol=1e-9)


def test_ece_detects_miscalibration():
    # 90% confident but only 33% correct => ECE around the gap
    y = np.array([0, 1, 2])
    P = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
    ece = multiclass_ece(y, P, CLASSES)
    assert 0.4 < ece < 0.7  # |0.9 confidence - ~0.33 accuracy|


def test_accuracy_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 80)
    pred = y.copy()
    flip = rng.choice(80, 20, replace=False)
    pred[flip] = (pred[flip] + 1) % 3  # 75% correct
    acc, lo, hi = accuracy_ci(y, pred, n_boot=500, seed=0)
    assert math.isclose(acc, 0.75, abs_tol=1e-9)
    assert lo <= acc <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_degenerate_inputs_return_nan():
    assert math.isnan(multiclass_brier([], np.empty((0, 3)), CLASSES))
    assert math.isnan(multiclass_ece([], np.empty((0, 3)), CLASSES))
    auc, lo, hi = macro_auc_ovr_ci([], np.empty((0, 3)), CLASSES)
    assert math.isnan(auc) and math.isnan(lo) and math.isnan(hi)
    # single-class truth => no positive/negative split => NaN AUC, no raise
    y = np.zeros(5, dtype=int)
    P = np.tile([0.8, 0.1, 0.1], (5, 1))
    auc, _, _ = macro_auc_ovr_ci(y, P, CLASSES, n_boot=50)
    assert math.isnan(auc)


def test_classes_default_inferred():
    y = np.array([0, 1, 2, 1])
    P = _onehot(y)
    # no classes arg => inferred from y, same perfect result
    assert multiclass_brier(y, P) == 0.0
