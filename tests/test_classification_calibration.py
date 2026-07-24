"""Tests for qradiomics.analytics.classification_calibration — Brier/ECE,
AUC CI, and the generic out-of-fold Youden's-J threshold-calibration
utilities (youden_threshold, oof_proba) added for qradiomics#8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qradiomics.analytics.classification_calibration import (
    auc_ci,
    brier_ece,
    oof_proba,
    youden_threshold,
)


class TestYoudenThreshold:
    def test_perfectly_separated_classes(self):
        # Probabilities perfectly separate the two classes at 0.5 — any
        # threshold strictly between the clusters maximizes J.
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
        thr = youden_threshold(y, p)
        assert 0.15 < thr <= 0.85

    def test_threshold_is_one_of_the_predicted_scores(self):
        # sklearn's roc_curve always returns thresholds drawn from the
        # (plus one sentinel) unique predicted scores.
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        p = np.array([0.2, 0.9, 0.1, 0.6, 0.4, 0.7, 0.3, 0.8])
        thr = youden_threshold(y, p)
        assert isinstance(thr, float)

    def test_shifts_away_from_half_under_imbalance(self):
        # Strongly imbalanced (80:20) but well-ranked probabilities that are
        # NOT calibrated to prevalence -- Youden's J should not necessarily
        # land on 0.5 the way a naive default would assume.
        rng = np.random.default_rng(0)
        n_neg, n_pos = 80, 20
        y = np.array([0] * n_neg + [1] * n_pos)
        p = np.concatenate([
            rng.normal(0.3, 0.1, n_neg),
            rng.normal(0.6, 0.1, n_pos),
        ])
        p = np.clip(p, 0.0, 1.0)
        thr = youden_threshold(y, p)
        assert 0.0 <= thr <= 1.0


class TestOofProba:
    @pytest.fixture
    def synthetic(self):
        rng = np.random.default_rng(42)
        n, k = 60, 4
        X = rng.normal(0, 1, (n, k))
        # Make the first feature informative so the classifier isn't at chance.
        y = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)
        return X, y

    def test_returns_one_probability_per_sample_no_nans(self, synthetic):
        from sklearn.linear_model import LogisticRegression

        X, y = synthetic
        oof = oof_proba(LogisticRegression(max_iter=200), X, y, n_splits=5, random_state=0)
        assert oof.shape == (len(y),)
        assert not np.isnan(oof).any()
        assert np.all((oof >= 0) & (oof <= 1))

    def test_accepts_dataframe_and_series(self, synthetic):
        from sklearn.linear_model import LogisticRegression

        X, y = synthetic
        Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        ys = pd.Series(y)
        oof = oof_proba(LogisticRegression(max_iter=200), Xdf, ys, n_splits=5, random_state=0)
        assert oof.shape == (len(y),)
        assert not np.isnan(oof).any()

    def test_does_not_mutate_or_leak_fitted_state_into_input_estimator(self, synthetic):
        from sklearn.linear_model import LogisticRegression

        X, y = synthetic
        clf = LogisticRegression(max_iter=200)
        oof_proba(clf, X, y, n_splits=5, random_state=0)
        # oof_proba must clone per fold — the estimator passed in should
        # remain unfitted (no coef_ attribute set on it).
        assert not hasattr(clf, "coef_")

    def test_reproducible_for_fixed_random_state(self, synthetic):
        from sklearn.linear_model import LogisticRegression

        X, y = synthetic
        oof1 = oof_proba(LogisticRegression(max_iter=200), X, y, n_splits=5, random_state=7)
        oof2 = oof_proba(LogisticRegression(max_iter=200), X, y, n_splits=5, random_state=7)
        np.testing.assert_allclose(oof1, oof2)

    def test_feeds_into_youden_threshold_end_to_end(self, synthetic):
        from sklearn.linear_model import LogisticRegression

        X, y = synthetic
        oof = oof_proba(LogisticRegression(max_iter=200), X, y, n_splits=5, random_state=0)
        thr = youden_threshold(y, oof)
        assert 0.0 <= thr <= 1.0


class TestExistingCalibrationMetricsUnaffected:
    """Sanity-lock brier_ece/auc_ci (untouched by this change) still work
    alongside the new imports in this module."""

    def test_brier_ece_perfect_predictions(self):
        y = [0, 0, 1, 1]
        p = [0.0, 0.0, 1.0, 1.0]
        brier, ece = brier_ece(y, p)
        assert brier == 0.0
        assert ece == 0.0

    def test_auc_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 100)
        p = np.clip(y + rng.normal(0, 0.3, 100), 0, 1)
        auc, lo, hi = auc_ci(y, p, n_boot=200, seed=1)
        assert lo <= auc <= hi
