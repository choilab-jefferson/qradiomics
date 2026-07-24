"""Tests for multi-class support in qr ml train/evaluate/predict (qradiomics#11).

Covers _train_classify's multi-class branch directly (metrics shape,
skip of binary-only calibration helpers, backward-compatible binary
path), plus CLI-level predict/evaluate on a trained multi-class model.
Also covers composition with patient-grouped CV (qradiomics#10) — the
two features touch the same function from independent branches.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from click.testing import CliRunner

from qradiomics.cli.commands.ml import _train_classify
from qradiomics.cli.main import cli


def _multiclass_df(n_per_class=15, n_classes=3, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for c in range(n_classes):
        for _ in range(n_per_class):
            row = {"patient_id": f"P{len(rows):03d}", "y": c}
            for f in range(n_features):
                row[f"feat_{f}"] = float(rng.normal(loc=c, scale=1.5))
            rows.append(row)
    return pd.DataFrame(rows)


def _multiclass_multi_row_df(n_patients=12, rows_per_patient=3, n_classes=3,
                             n_features=4, seed=0):
    """Multi-row-per-patient AND multi-class — the qradiomics#10 x #11
    intersection (e.g. a future lesion-grain cohort with a >2-class label).
    Signal is on the features, not perfectly patient-determined, so AUC
    is a meaningful (non-degenerate) number. Class assignment is randomized
    (not a fixed p % n_classes cycle) so StratifiedGroupKFold has a better
    chance of representing every class in every fold at small N."""
    rng = np.random.RandomState(seed)
    patient_classes = rng.randint(0, n_classes, n_patients)
    rows = []
    for p in range(n_patients):
        c = int(patient_classes[p])
        for _ in range(rows_per_patient):
            row = {"patient_id": f"P{p:03d}", "y": c}
            for f in range(n_features):
                row[f"feat_{f}"] = float(rng.normal(loc=c, scale=2.5))
            rows.append(row)
    return pd.DataFrame(rows)


def _binary_df(n=30, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(n)],
        "y": rng.randint(0, 2, n),
        "feat_0": rng.normal(size=n),
        "feat_1": rng.normal(size=n),
    })


def _invoke(*args):
    return CliRunner().invoke(cli, list(args))


class TestTrainClassifyMulticlass:
    def test_detects_three_classes(self):
        df = _multiclass_df()
        _, metrics = _train_classify(df, outcome="y", folds=4, top_features=3)
        assert metrics["n_classes"] == 3
        assert metrics["classes"] == [0, 1, 2]

    def test_ovr_auc_computed_not_binary_slice(self):
        df = _multiclass_df()
        _, metrics = _train_classify(df, outcome="y", folds=4, top_features=3)
        assert metrics["cv_auc_mean"] is not None
        assert 0.0 <= metrics["cv_auc_mean"] <= 1.0

    def test_classification_report_present_for_multiclass(self):
        df = _multiclass_df()
        _, metrics = _train_classify(df, outcome="y", folds=4, top_features=3)
        report = metrics["cv_classification_report"]
        assert report is not None
        # classification_report keys include per-class labels + averages
        assert "macro avg" in report
        assert "weighted avg" in report

    def test_binary_only_calibration_skipped_for_multiclass(self):
        df = _multiclass_df()
        _, metrics = _train_classify(df, outcome="y", folds=4, top_features=3)
        assert metrics["cv_brier"] is None
        assert metrics["cv_ece"] is None
        assert metrics["cv_decision_curve"] is None

    def test_final_model_predicts_all_classes(self):
        df = _multiclass_df()
        model, _ = _train_classify(df, outcome="y", folds=4, top_features=3)
        assert sorted(model["classifier"].classes_.tolist()) == [0, 1, 2]

    def test_binary_path_unaffected(self):
        df = _binary_df()
        _, metrics = _train_classify(df, outcome="y", folds=5, top_features=2)
        assert metrics["n_classes"] == 2
        assert metrics["cv_classification_report"] is None
        # binary calibration fields still populated as before
        assert metrics["cv_brier"] is not None
        assert metrics["cv_auc_ci_lo"] is not None


class TestPredictEvaluateMulticlass:
    def _train_and_save(self, tmp_path, df):
        input_csv = tmp_path / "train.csv"
        df.to_csv(input_csv, index=False)
        model_path = tmp_path / "model.pkl"
        metrics_path = tmp_path / "metrics.json"
        result = _invoke("ml", "train", "--input", str(input_csv), "--task", "classify",
                         "--outcome", "y", "--folds", "3", "--top-features", "3",
                         "--model", str(model_path), "--metrics", str(metrics_path))
        assert result.exit_code == 0, result.output
        return model_path

    def test_predict_multiclass_outputs_per_class_columns(self, tmp_path):
        df = _multiclass_df()
        model_path = self._train_and_save(tmp_path, df)
        out_csv = tmp_path / "preds.csv"
        result = _invoke("ml", "predict", "--input", str(tmp_path / "train.csv"),
                         "--model", str(model_path), "--task", "classify",
                         "--output", str(out_csv))
        assert result.exit_code == 0, result.output
        preds = pd.read_csv(out_csv)
        assert {"proba_0", "proba_1", "proba_2", "pred_class"}.issubset(preds.columns)
        # per-row probabilities across the 3 classes sum to ~1
        row_sums = preds[["proba_0", "proba_1", "proba_2"]].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_evaluate_multiclass_reports_classification_report(self, tmp_path):
        df = _multiclass_df()
        model_path = self._train_and_save(tmp_path, df)
        report_path = tmp_path / "eval.json"
        result = _invoke("ml", "evaluate", "--input", str(tmp_path / "train.csv"),
                         "--model", str(model_path), "--task", "classify",
                         "--outcome", "y", "--report", str(report_path))
        assert result.exit_code == 0, result.output
        report = json.loads(report_path.read_text())
        assert report["n_classes"] == 3
        assert "classification_report" in report

    def test_predict_binary_unaffected(self, tmp_path):
        df = _binary_df()
        model_path = self._train_and_save(tmp_path, df)
        out_csv = tmp_path / "preds.csv"
        result = _invoke("ml", "predict", "--input", str(tmp_path / "train.csv"),
                         "--model", str(model_path), "--task", "classify",
                         "--output", str(out_csv))
        assert result.exit_code == 0, result.output
        preds = pd.read_csv(out_csv)
        assert list(preds.columns) == ["patient_id", "proba"]


class TestMulticlassWithGroupedCV:
    """qradiomics#10 x #11 intersection: multi-row-per-patient + >2 classes."""

    def test_grouping_and_multiclass_both_active(self):
        # 30 patients / 3 folds = 10 patients per test fold, giving
        # StratifiedGroupKFold enough groups per fold to represent all 3
        # classes reliably (avoids a degenerate single-class test fold).
        df = _multiclass_multi_row_df(n_patients=30, rows_per_patient=3, n_classes=3)
        _, metrics = _train_classify(df, outcome="y", folds=3, top_features=3,
                                     group_col="patient_id")
        assert metrics["n_classes"] == 3
        assert metrics["group_col"] == "patient_id"
        assert metrics["n_groups"] == 30
        assert metrics["cv_classification_report"] is not None
        assert metrics["cv_auc_mean"] is not None
        assert not np.isnan(metrics["cv_auc_mean"])
