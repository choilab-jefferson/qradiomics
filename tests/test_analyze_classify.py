"""Tests for qr analyze classify command (logistic regression, binary/ordinal outcomes)."""

import csv

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from qradiomics.cli.commands.analyze import analyze


@pytest.fixture
def classify_csv(tmp_path):
    """Minimal CSV with binary outcome and radiomics features."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "patient_id": [f"PT{i:03d}" for i in range(n)],
            "fdg_uptake": rng.integers(0, 2, size=n),  # binary 0/1
            "feature_A": rng.normal(0, 1, size=n),
            "feature_B": rng.normal(5, 2, size=n),
            "feature_C": rng.normal(2, 0.5, size=n),
        }
    )
    path = tmp_path / "classify_ready.csv"
    df.to_csv(path, index=False)
    return path


def test_classify_produces_output_csv(classify_csv, tmp_path):
    """classify should write a CSV with feature, auc, p columns."""
    runner = CliRunner()
    output = tmp_path / "classify_results.csv"
    result = runner.invoke(
        analyze,
        [
            "classify",
            "--input",
            str(classify_csv),
            "--outcome",
            "fdg_uptake",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    rows = list(csv.DictReader(output.open()))
    assert len(rows) >= 1
    assert "feature" in rows[0]
    assert "auc" in rows[0]
    assert "p" in rows[0]


def test_classify_results_sorted_by_p(classify_csv, tmp_path):
    """Results should be sorted by p-value ascending."""
    runner = CliRunner()
    output = tmp_path / "out.csv"
    runner.invoke(
        analyze,
        [
            "classify",
            "--input",
            str(classify_csv),
            "--outcome",
            "fdg_uptake",
            "--output",
            str(output),
        ],
    )
    rows = list(csv.DictReader(output.open()))
    p_values = [float(r["p"]) for r in rows]
    assert p_values == sorted(p_values)


def test_classify_shows_summary_in_console(classify_csv, tmp_path):
    """Console output should show AUC and feature names."""
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "classify",
            "--input",
            str(classify_csv),
            "--outcome",
            "fdg_uptake",
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 0
    assert "AUC" in result.output or "auc" in result.output.lower()


def test_classify_fails_on_missing_outcome_column(classify_csv, tmp_path):
    """Should exit non-zero when outcome column is missing."""
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "classify",
            "--input",
            str(classify_csv),
            "--outcome",
            "NONEXISTENT",
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code != 0
    assert "NONEXISTENT" in result.output or "not found" in result.output.lower()


def test_classify_top_n_limits_console_output(classify_csv, tmp_path):
    """--top-n should limit how many features are shown in console."""
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "classify",
            "--input",
            str(classify_csv),
            "--outcome",
            "fdg_uptake",
            "--output",
            str(tmp_path / "out.csv"),
            "--top-n",
            "1",
        ],
    )
    assert result.exit_code == 0
    # Only 1 feature line shown
    feature_lines = [line for line in result.output.splitlines() if "feature_" in line.lower()]
    assert len(feature_lines) <= 1
