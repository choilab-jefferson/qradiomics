"""Tests for qr analyze importance command."""

import csv

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from qradiomics.cli.commands.analyze import analyze


@pytest.fixture
def classification_csv(tmp_path):
    """CSV with binary outcome and numeric features."""
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(
        {
            "patient_id": [f"PT{i:03d}" for i in range(n)],
            "label": rng.integers(0, 2, size=n),
            "feat_A": rng.normal(0, 1, size=n),
            "feat_B": rng.normal(5, 2, size=n),
            "feat_C": rng.normal(-1, 3, size=n),
            "feat_D": rng.normal(10, 1, size=n),
        }
    )
    path = tmp_path / "classify.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def survival_csv(tmp_path):
    """CSV with survival outcome and numeric features."""
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(
        {
            "patient_id": [f"PT{i:03d}" for i in range(n)],
            "OS_months": rng.exponential(scale=24, size=n),
            "OS_event": rng.integers(0, 2, size=n),
            "feat_A": rng.normal(0, 1, size=n),
            "feat_B": rng.normal(5, 2, size=n),
            "feat_C": rng.normal(-1, 3, size=n),
        }
    )
    path = tmp_path / "survival.csv"
    df.to_csv(path, index=False)
    return path


def test_importance_classification_model_method(classification_csv, tmp_path):
    runner = CliRunner()
    output = tmp_path / "importance.csv"
    result = runner.invoke(
        analyze,
        [
            "importance",
            "--input",
            str(classification_csv),
            "--outcome",
            "label",
            "--method",
            "model",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    rows = list(csv.DictReader(output.open()))
    assert len(rows) >= 1
    assert "feature" in rows[0]
    assert "importance" in rows[0]
    assert "method" in rows[0]


def test_importance_classification_permutation_method(classification_csv, tmp_path):
    runner = CliRunner()
    output = tmp_path / "importance.csv"
    result = runner.invoke(
        analyze,
        [
            "importance",
            "--input",
            str(classification_csv),
            "--outcome",
            "label",
            "--method",
            "permutation",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(output.open()))
    assert len(rows) >= 1
    assert "importance" in rows[0]


def test_importance_classification_shap_method(classification_csv, tmp_path):
    pytest.importorskip("shap")
    runner = CliRunner()
    output = tmp_path / "importance.csv"
    result = runner.invoke(
        analyze,
        [
            "importance",
            "--input",
            str(classification_csv),
            "--outcome",
            "label",
            "--method",
            "shap",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(output.open()))
    assert len(rows) >= 1
    assert "importance" in rows[0]
    assert rows[0]["method"] == "shap"


def test_importance_all_methods_outputs_ranked_table(classification_csv, tmp_path):
    pytest.importorskip("shap")
    runner = CliRunner()
    output = tmp_path / "importance.csv"
    result = runner.invoke(
        analyze,
        [
            "importance",
            "--input",
            str(classification_csv),
            "--outcome",
            "label",
            "--method",
            "all",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(output.open()))
    methods_in_output = {r["method"] for r in rows}
    assert "model" in methods_in_output
    assert "permutation" in methods_in_output
    assert "shap" in methods_in_output


def test_importance_survival_binarizes_and_runs(survival_csv, tmp_path):
    runner = CliRunner()
    output = tmp_path / "importance.csv"
    result = runner.invoke(
        analyze,
        [
            "importance",
            "--input",
            str(survival_csv),
            "--outcome",
            "OS_months",
            "--event",
            "OS_event",
            "--method",
            "model",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    rows = list(csv.DictReader(output.open()))
    assert len(rows) >= 1


def test_importance_top_n_limits_console_output(classification_csv, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "importance",
            "--input",
            str(classification_csv),
            "--outcome",
            "label",
            "--method",
            "model",
            "--top-n",
            "2",
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "feat_" in result.output


def test_importance_fails_on_missing_outcome(classification_csv, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "importance",
            "--input",
            str(classification_csv),
            "--outcome",
            "MISSING_COL",
            "--method",
            "model",
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code != 0
    assert "MISSING_COL" in result.output or "not found" in result.output.lower()
