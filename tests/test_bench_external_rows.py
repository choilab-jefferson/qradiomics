"""Tests for the shared external-test-refit aggregation logic
(qradiomics.cli.commands.bench.build_external_test_rows), added for
qradiomics#8 to (a) dedupe the near-identical loop that used to be inline in
both `qr bench` and `qr ml classify`, and (b) special-case TPOT's
--tpot-repeats mean/std reporting, kept separate from the k-fold
cv_auc_mean/cv_auc_std columns (a different statistic).

These tests monkeypatch the underlying single-run fit/score function so no
model is ever actually fit -- they verify the aggregation arithmetic and
column shape only, per the qradiomics#8 test-plan constraint that no test
may run a real TPOT genetic search.
"""
from __future__ import annotations

import numpy as np
import pytest

from qradiomics.cli.commands import bench as bench_mod


@pytest.fixture
def dummy_data():
    X_tr = np.zeros((10, 3))
    y_tr = np.array([0, 1] * 5)
    X_ext = np.zeros((6, 3))
    y_ext = np.array([0, 1, 0, 1, 0, 1])
    return X_tr, y_tr, X_ext, y_ext


def _fake_fit_and_score(pipe, grid, X_tr, y_tr, X_ext, y_ext, *,
                        inner_cv, n_jobs, seed, calibrate_threshold):
    """Deterministic stand-in for _fit_and_score_external: metrics depend
    only on `seed` so the test can compute the exact expected mean/std."""
    metrics = {
        "auc": 0.70 + 0.01 * seed,
        "ap": 0.60 + 0.01 * seed,
        "brier": 0.30 - 0.01 * seed,
    }
    if calibrate_threshold:
        metrics.update({
            "threshold": 0.50 + 0.001 * seed,
            "accuracy": 0.80 + 0.001 * seed,
            "sensitivity": 0.75 + 0.001 * seed,
            "specificity": 0.70 + 0.001 * seed,
        })
    return metrics, {"seed_used": seed}


class TestBuildExternalTestRowsAggregation:
    def test_non_tpot_model_is_fit_once(self, monkeypatch, dummy_data):
        monkeypatch.setattr(bench_mod, "_fit_and_score_external", _fake_fit_and_score)
        X_tr, y_tr, X_ext, y_ext = dummy_data

        rows = bench_mod.build_external_test_rows(
            ["LR"], {"LR": 0.9}, X_tr, y_tr, X_ext, y_ext,
            seed=10, inner_cv=5, n_jobs=1, tpot_repeats=5,
            calibrate_threshold=False,
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["model"] == "LR"
        assert row["ext_repeats"] == 1
        assert row["ext_auc"] == round(0.70 + 0.01 * 10, 4)
        assert row["ext_auc_std"] == 0.0
        assert row["cv_oof_auc"] == 0.9
        assert row["best_params"] == "{'seed_used': 10}"

    def test_tpot_is_fit_tpot_repeats_times_and_aggregated(self, monkeypatch, dummy_data):
        monkeypatch.setattr(bench_mod, "_fit_and_score_external", _fake_fit_and_score)
        X_tr, y_tr, X_ext, y_ext = dummy_data
        seed = 10
        tpot_repeats = 3

        rows = bench_mod.build_external_test_rows(
            ["TPOT"], {"TPOT": 0.5}, X_tr, y_tr, X_ext, y_ext,
            seed=seed, inner_cv=5, n_jobs=1, tpot_repeats=tpot_repeats,
            calibrate_threshold=False,
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["ext_repeats"] == tpot_repeats

        expected_aucs = [0.70 + 0.01 * (seed + i) for i in range(tpot_repeats)]
        assert row["ext_auc"] == round(float(np.mean(expected_aucs)), 4)
        assert row["ext_auc_std"] == round(float(np.std(expected_aucs)), 4)
        assert row["ext_auc_std"] > 0.0
        # best_params must come from the FIRST repeat only (seed, not seed+i).
        assert row["best_params"] == f"{{'seed_used': {seed}}}"

    def test_tpot_repeats_are_never_reported_under_cv_columns(self, monkeypatch, dummy_data):
        # Regression guard: the repeat-and-aggregate stats must live under
        # ext_*_std / ext_repeats, never overload cv_auc_mean/cv_auc_std
        # (a different, k-fold-variance statistic computed elsewhere).
        monkeypatch.setattr(bench_mod, "_fit_and_score_external", _fake_fit_and_score)
        X_tr, y_tr, X_ext, y_ext = dummy_data

        rows = bench_mod.build_external_test_rows(
            ["TPOT"], {"TPOT": 0.5}, X_tr, y_tr, X_ext, y_ext,
            seed=1, inner_cv=5, n_jobs=1, tpot_repeats=4,
            calibrate_threshold=False,
        )
        row = rows[0]
        assert "cv_auc_mean" not in row
        assert "cv_auc_std" not in row
        assert "ext_auc_std" in row
        assert "ext_repeats" in row

    def test_calibrate_threshold_columns_are_aggregated_too(self, monkeypatch, dummy_data):
        monkeypatch.setattr(bench_mod, "_fit_and_score_external", _fake_fit_and_score)
        X_tr, y_tr, X_ext, y_ext = dummy_data
        seed, tpot_repeats = 5, 2

        rows = bench_mod.build_external_test_rows(
            ["TPOT"], {"TPOT": 0.5}, X_tr, y_tr, X_ext, y_ext,
            seed=seed, inner_cv=5, n_jobs=1, tpot_repeats=tpot_repeats,
            calibrate_threshold=True,
        )
        row = rows[0]
        expected_thr = [0.50 + 0.001 * (seed + i) for i in range(tpot_repeats)]
        assert row["threshold"] == round(float(np.mean(expected_thr)), 4)
        assert "threshold_std" in row
        assert "accuracy" in row and "accuracy_std" in row
        assert "sensitivity" in row and "sensitivity_std" in row
        assert "specificity" in row and "specificity_std" in row

    def test_calibrate_threshold_off_by_default_no_threshold_columns(self, monkeypatch, dummy_data):
        monkeypatch.setattr(bench_mod, "_fit_and_score_external", _fake_fit_and_score)
        X_tr, y_tr, X_ext, y_ext = dummy_data

        rows = bench_mod.build_external_test_rows(
            ["LR"], {"LR": 0.5}, X_tr, y_tr, X_ext, y_ext,
            seed=1, inner_cv=5, n_jobs=1,
        )
        assert "threshold" not in rows[0]

    def test_multiple_models_uniform_schema(self, monkeypatch, dummy_data):
        # LR (1 repeat) and TPOT (N repeats) must produce rows with the
        # exact same set of keys so downstream DataFrame construction never
        # produces ragged/NaN-filled columns.
        monkeypatch.setattr(bench_mod, "_fit_and_score_external", _fake_fit_and_score)
        X_tr, y_tr, X_ext, y_ext = dummy_data

        rows = bench_mod.build_external_test_rows(
            ["LR", "TPOT"], {"LR": 0.7, "TPOT": 0.75}, X_tr, y_tr, X_ext, y_ext,
            seed=2, inner_cv=5, n_jobs=1, tpot_repeats=3,
            calibrate_threshold=True,
        )
        assert len(rows) == 2
        assert set(rows[0].keys()) == set(rows[1].keys())
