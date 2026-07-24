"""Tests for patient-grouped CV in qr ml train (qradiomics#10).

Covers the group-aware splitter helper directly, plus _train_classify /
_train_survival end-to-end on synthetic multi-row-per-patient data, to
confirm no same-patient row ever splits across train/test folds, and
that single-row-per-patient cohorts keep using the original (ungrouped)
splitter unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qradiomics.cli.commands.ml import (
    _assert_no_group_leakage,
    _make_group_aware_splitter,
    _train_classify,
    _train_survival,
)


def _multi_row_classify_df(n_patients=6, rows_per_patient=4, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_patients):
        y = int(p % 2 == 0)
        for _ in range(rows_per_patient):
            row = {"patient_id": f"P{p:03d}", "y": y}
            for f in range(n_features):
                row[f"feat_{f}"] = float(rng.normal(loc=y, scale=1.0))
            rows.append(row)
    return pd.DataFrame(rows)


def _multi_row_survival_df(n_patients=6, rows_per_patient=3, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_patients):
        event = int(p % 2 == 0)
        months = float(rng.uniform(6, 60))
        for _ in range(rows_per_patient):
            row = {"patient_id": f"P{p:03d}", "OS_months": months, "OS_event": event}
            for f in range(n_features):
                row[f"feat_{f}"] = float(rng.normal(loc=event, scale=1.0))
            rows.append(row)
    return pd.DataFrame(rows)


class TestMakeGroupAwareSplitter:
    def test_no_group_col_uses_plain_splitter(self):
        df = _multi_row_classify_df()
        splitter, split_args, n_effective, groups = _make_group_aware_splitter(
            df, df["y"], folds=3, group_col=None, stratify=True)
        assert groups is None
        assert n_effective == len(df)
        assert splitter.__class__.__name__ == "StratifiedKFold"

    def test_unique_patient_per_row_uses_plain_splitter(self):
        # one row per patient -> nunique(patient_id) == len(df) -> no grouping
        df = pd.DataFrame({
            "patient_id": [f"P{i}" for i in range(10)],
            "y": [i % 2 for i in range(10)],
        })
        splitter, split_args, n_effective, groups = _make_group_aware_splitter(
            df, df["y"], folds=3, group_col="patient_id", stratify=True)
        assert groups is None
        assert splitter.__class__.__name__ == "StratifiedKFold"

    def test_duplicate_patient_id_switches_to_grouped_splitter(self):
        df = _multi_row_classify_df(n_patients=6, rows_per_patient=4)
        splitter, split_args, n_effective, groups = _make_group_aware_splitter(
            df, df["y"], folds=3, group_col="patient_id", stratify=True)
        assert groups is not None
        assert n_effective == 6
        assert splitter.__class__.__name__ == "StratifiedGroupKFold"

    def test_folds_clamped_to_n_groups(self):
        df = _multi_row_classify_df(n_patients=3, rows_per_patient=5)
        splitter, split_args, n_effective, groups = _make_group_aware_splitter(
            df, df["y"], folds=10, group_col="patient_id", stratify=True)
        assert splitter.get_n_splits() <= 3


class TestAssertNoGroupLeakage:
    def test_none_groups_is_noop(self):
        _assert_no_group_leakage(None, [0, 1], [2, 3], 1, 3)  # must not raise

    def test_raises_on_overlap(self):
        groups = pd.Series(["A", "A", "B", "B"])
        with pytest.raises(RuntimeError, match="leakage"):
            _assert_no_group_leakage(groups, [0, 1], [1, 2], 1, 3)

    def test_no_raise_when_disjoint(self):
        groups = pd.Series(["A", "A", "B", "B"])
        _assert_no_group_leakage(groups, [0, 1], [2, 3], 1, 3)  # must not raise


class TestTrainClassifyGrouped:
    def test_runs_without_leakage_error_on_multi_row_data(self):
        df = _multi_row_classify_df(n_patients=8, rows_per_patient=4)
        model, metrics = _train_classify(df, outcome="y", folds=4,
                                         top_features=3, group_col="patient_id")
        assert metrics["task"] == "classify"
        assert metrics["group_col"] == "patient_id"
        assert metrics["n_groups"] == 8

    def test_single_row_per_patient_metrics_report_no_grouping(self):
        rng = np.random.RandomState(1)
        n = 20
        df = pd.DataFrame({
            "patient_id": [f"P{i}" for i in range(n)],
            "y": [i % 2 for i in range(n)],
            "feat_0": rng.normal(size=n),
            "feat_1": rng.normal(size=n),
        })
        model, metrics = _train_classify(df, outcome="y", folds=4, top_features=2)
        assert metrics["group_col"] is None
        assert metrics["n_groups"] is None

    def test_group_col_none_disables_grouping_even_with_duplicates(self):
        df = _multi_row_classify_df(n_patients=6, rows_per_patient=4)
        model, metrics = _train_classify(df, outcome="y", folds=3,
                                         top_features=3, group_col=None)
        assert metrics["group_col"] is None


class TestTrainSurvivalGrouped:
    def test_runs_without_leakage_error_on_multi_row_data(self):
        df = _multi_row_survival_df(n_patients=8, rows_per_patient=3)
        model, metrics = _train_survival(df, time_col="OS_months", event_col="OS_event",
                                         folds=4, top_features=3, group_col="patient_id")
        assert metrics["task"] == "survival"
        assert metrics["group_col"] == "patient_id"
        assert metrics["n_groups"] == 8
