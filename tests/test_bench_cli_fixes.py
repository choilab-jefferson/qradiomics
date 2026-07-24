"""Regression tests for the qradiomics#8 code-review fix pass on `qr bench` /
`qr ml classify` / `qr ml benchmark`:

  * [0] --tpot-repeats must reject 0/negative values (click.IntRange(min=1))
        instead of silently producing NaN metrics via an empty repeat loop.
  * [1] GridSearchCV's fit (the real-model, has-a-grid branch of
        `_fit_and_score_external`) must NOT suppress warnings -- only the
        no-grid/AutoML branch should.
  * [4] --calibrate-threshold must not crash when the training minority
        class has fewer than 2 members; it should skip calibration for that
        run (metrics dict lacks threshold/accuracy/...) and emit a warning,
        not force a degenerate StratifiedKFold split.
  * [6] the --tpot-repeats repeat loop in build_external_test_rows is
        parallelized (joblib.Parallel) for N>1 repeats, with each repeat's
        own internal parallelism pinned to 1 to avoid oversubscription; the
        aggregation result must be identical in shape/content to the
        pre-refactor sequential loop.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from qradiomics.cli.commands import bench as bench_mod
from qradiomics.cli.commands.bench import bench
from qradiomics.cli.commands.ml import ml


@pytest.fixture
def tiny_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 20
    outcome = rng.integers(0, 2, size=n)
    df = pd.DataFrame({
        "feature_A": rng.normal(0, 1, size=n) + outcome * 1.5,
        "feature_B": rng.normal(5, 2, size=n),
        "outcome": outcome,
    })
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# [0] --tpot-repeats lower-bound validation
# ---------------------------------------------------------------------------

class TestTpotRepeatsLowerBound:
    @pytest.mark.parametrize("bad_value", ["0", "-1", "-5"])
    def test_bench_rejects_non_positive_tpot_repeats(self, tiny_csv, bad_value):
        result = CliRunner().invoke(bench, [
            "--input", str(tiny_csv), "--outcome", "outcome",
            "--tpot-repeats", bad_value,
        ])
        assert result.exit_code != 0
        assert "tpot-repeats" in result.output.lower() or "invalid value" in result.output.lower()

    def test_ml_classify_rejects_non_positive_tpot_repeats(self, tiny_csv):
        result = CliRunner().invoke(ml, [
            "classify", "--input", str(tiny_csv), "--outcome", "outcome",
            "--tpot-repeats", "0",
        ])
        assert result.exit_code != 0

    def test_ml_benchmark_rejects_non_positive_tpot_repeats(self, tiny_csv):
        result = CliRunner().invoke(ml, [
            "benchmark", "--input", str(tiny_csv), "--outcome", "outcome",
            "--tpot-repeats", "0",
        ])
        assert result.exit_code != 0

    def test_positive_tpot_repeats_still_accepted(self, tiny_csv):
        # Sanity: the IntRange lower bound must not reject valid values.
        result = CliRunner().invoke(bench, [
            "--input", str(tiny_csv), "--outcome", "outcome",
            "--models", "LR", "--cv", "2", "--inner-cv", "2",
            "--tpot-repeats", "1",
            "--output-dir", str(tiny_csv.parent / "out"),
        ])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# [1] Grid-branch warning suppression regression
# ---------------------------------------------------------------------------

class _WarningGridSearchCV:
    """Stand-in for sklearn.model_selection.GridSearchCV that always warns
    during fit, so the test can assert the warning is NOT swallowed."""

    def __init__(self, estimator, param_grid, **kwargs):
        self._estimator = estimator

    def fit(self, X, y):
        warnings.warn("dummy grid-search convergence warning", UserWarning)
        self._estimator.fit(X, y)
        self.best_estimator_ = self._estimator
        self.best_params_ = {"dummy": True}
        return self


class TestGridBranchWarningsNotSuppressed:
    def test_grid_branch_surfaces_warnings(self, monkeypatch):
        import sklearn.model_selection as skms

        monkeypatch.setattr(skms, "GridSearchCV", _WarningGridSearchCV)

        from qradiomics.classification.registry import build_model
        pipe, grid = build_model("LR", random_state=0)
        assert grid  # LR has a real param grid -> takes the `if grid:` branch

        rng = np.random.default_rng(0)
        X_tr = rng.normal(size=(20, 3))
        y_tr = np.array([0, 1] * 10)
        X_ext = rng.normal(size=(6, 3))
        y_ext = np.array([0, 1, 0, 1, 0, 1])

        with pytest.warns(UserWarning, match="dummy grid-search convergence warning"):
            bench_mod._fit_and_score_external(
                pipe, grid, X_tr, y_tr, X_ext, y_ext,
                inner_cv=2, n_jobs=1, seed=0, calibrate_threshold=False,
            )

    def test_no_grid_branch_still_suppresses_warnings(self, monkeypatch):
        """Regression guard: the AutoML/no-grid branch must keep suppressing
        warnings (unchanged pre-refactor behavior) -- only the grid branch
        was over-broadened and needed fixing."""
        class _WarningPipe:
            def fit(self, X, y):
                warnings.warn("dummy automl warning", UserWarning)
                return self

            def predict_proba(self, X):
                n = len(X)
                return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        rng = np.random.default_rng(0)
        X_tr = rng.normal(size=(10, 3))
        y_tr = np.array([0, 1] * 5)
        X_ext = rng.normal(size=(6, 3))
        y_ext = np.array([0, 1, 0, 1, 0, 1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bench_mod._fit_and_score_external(
                _WarningPipe(), {}, X_tr, y_tr, X_ext, y_ext,
                inner_cv=2, n_jobs=1, seed=0, calibrate_threshold=False,
            )
        assert not any("dummy automl warning" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# [4] --calibrate-threshold minority-class clamp
# ---------------------------------------------------------------------------

class TestCalibrateThresholdMinorityClassGuard:
    def test_single_member_minority_class_skips_calibration_with_warning(self):
        from qradiomics.classification.registry import build_model

        pipe, grid = build_model("LR", random_state=0)
        rng = np.random.default_rng(1)
        n = 15
        X_tr = rng.normal(size=(n, 3))
        y_tr = np.zeros(n, dtype=int)
        y_tr[0] = 1  # single-member minority class
        X_ext = rng.normal(size=(6, 3))
        y_ext = np.array([0, 1, 0, 1, 0, 1])

        with pytest.warns(UserWarning, match="Skipping --calibrate-threshold"):
            metrics, best_params = bench_mod._fit_and_score_external(
                pipe, grid, X_tr, y_tr, X_ext, y_ext,
                inner_cv=5, n_jobs=1, seed=0, calibrate_threshold=True,
            )

        # Calibration was skipped -> no threshold/accuracy/... keys, but the
        # base auc/ap/brier metrics must still be present (no crash).
        assert "threshold" not in metrics
        assert "accuracy" not in metrics
        assert "sensitivity" not in metrics
        assert "specificity" not in metrics
        assert "auc" in metrics and "ap" in metrics and "brier" in metrics

    def test_two_member_minority_class_still_calibrates(self):
        from qradiomics.classification.registry import build_model

        pipe, grid = build_model("LR", random_state=0)
        rng = np.random.default_rng(2)
        n = 20
        y_tr = np.zeros(n, dtype=int)
        y_tr[:2] = 1  # exactly 2 -> minority_count == 2, must NOT be skipped
        X_tr = rng.normal(size=(n, 3)) + y_tr[:, None] * 1.0
        X_ext = rng.normal(size=(6, 3))
        y_ext = np.array([0, 1, 0, 1, 0, 1])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics, _ = bench_mod._fit_and_score_external(
                pipe, grid, X_tr, y_tr, X_ext, y_ext,
                inner_cv=5, n_jobs=1, seed=0, calibrate_threshold=True,
            )
        assert "threshold" in metrics
        assert not any("Skipping --calibrate-threshold" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# [6] Parallel --tpot-repeats loop
# ---------------------------------------------------------------------------

def _fake_fit_and_score(pipe, grid, X_tr, y_tr, X_ext, y_ext, *,
                        inner_cv, n_jobs, seed, calibrate_threshold):
    return {"auc": 0.70 + 0.01 * seed, "ap": 0.6, "brier": 0.3}, {"seed_used": seed}


class TestParallelRepeatLoop:
    def test_n_repeats_produce_n_independent_rows_under_parallel_path(self, monkeypatch):
        monkeypatch.setattr(bench_mod, "_fit_and_score_external", _fake_fit_and_score)
        X_tr = np.zeros((10, 3))
        y_tr = np.array([0, 1] * 5)
        X_ext = np.zeros((6, 3))
        y_ext = np.array([0, 1, 0, 1, 0, 1])

        # n_jobs=1 keeps joblib on its in-process SequentialBackend (no
        # process spawn) -- this test targets the aggregation logic of the
        # now-parallel-capable code path, not the OS-level parallelism
        # itself (which subprocess-based test sandboxes may not support).
        seed, tpot_repeats = 100, 4
        rows = bench_mod.build_external_test_rows(
            ["TPOT"], {"TPOT": 0.5}, X_tr, y_tr, X_ext, y_ext,
            seed=seed, inner_cv=5, n_jobs=1, tpot_repeats=tpot_repeats,
            calibrate_threshold=False,
        )
        row = rows[0]
        expected_aucs = [0.70 + 0.01 * (seed + i) for i in range(tpot_repeats)]
        assert row["ext_repeats"] == tpot_repeats
        assert row["ext_auc"] == round(float(np.mean(expected_aucs)), 4)
        assert row["ext_auc_std"] == round(float(np.std(expected_aucs)), 4)
        # best_params from the first repeat (seed, not seed+i)
        assert row["best_params"] == f"{{'seed_used': {seed}}}"

    def test_repeat_pins_inner_n_jobs_to_avoid_oversubscription(self, monkeypatch):
        """Each repeat's own model n_jobs must be pinned to 1 when the outer
        repeat loop is parallelized (n_repeats > 1), so TPOT's internal
        multiprocessing doesn't stack with the outer joblib.Parallel."""
        seen_n_jobs = []

        class _FakePipe:
            def set_params(self, **kwargs):
                seen_n_jobs.append(kwargs.get("clf__n_jobs"))
                return self

        def _fake_build_model(name, random_state=42):
            return _FakePipe(), {}

        monkeypatch.setattr(bench_mod, "_fit_and_score_external", _fake_fit_and_score)
        monkeypatch.setattr(
            "qradiomics.classification.registry.build_model", _fake_build_model
        )

        X_tr = np.zeros((10, 3))
        y_tr = np.array([0, 1] * 5)
        X_ext = np.zeros((6, 3))
        y_ext = np.array([0, 1, 0, 1, 0, 1])

        bench_mod.build_external_test_rows(
            ["TPOT"], {"TPOT": 0.5}, X_tr, y_tr, X_ext, y_ext,
            seed=1, inner_cv=5, n_jobs=1, tpot_repeats=3,
            calibrate_threshold=False,
        )
        assert seen_n_jobs == [1, 1, 1]

    def test_single_repeat_model_not_pinned(self, monkeypatch):
        """Non-TPOT (n_repeats == 1) models must skip the pin/parallel
        machinery entirely -- set_params should not even be called."""
        seen_calls = []

        class _FakePipe:
            def set_params(self, **kwargs):
                seen_calls.append(kwargs)
                return self

        def _fake_build_model(name, random_state=42):
            return _FakePipe(), {}

        monkeypatch.setattr(bench_mod, "_fit_and_score_external", _fake_fit_and_score)
        monkeypatch.setattr(
            "qradiomics.classification.registry.build_model", _fake_build_model
        )

        X_tr = np.zeros((10, 3))
        y_tr = np.array([0, 1] * 5)
        X_ext = np.zeros((6, 3))
        y_ext = np.array([0, 1, 0, 1, 0, 1])

        bench_mod.build_external_test_rows(
            ["LR"], {"LR": 0.5}, X_tr, y_tr, X_ext, y_ext,
            seed=1, inner_cv=5, n_jobs=1, tpot_repeats=5,
            calibrate_threshold=False,
        )
        assert seen_calls == []
