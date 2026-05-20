"""Tests for qr analyze survival command."""

import csv

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from qradiomics.cli.commands.analyze import analyze


@pytest.fixture
def analysis_ready_csv(tmp_path):
    """Minimal analysis_ready.csv with known survival data."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "patient_id": [f"PT{i:03d}" for i in range(n)],
            "OS_months": rng.exponential(scale=24, size=n),
            "OS_event": rng.integers(0, 2, size=n),
            "feature_A": rng.normal(0, 1, size=n),
            "feature_B": rng.normal(5, 2, size=n),
        }
    )
    path = tmp_path / "analysis_ready.csv"
    df.to_csv(path, index=False)
    return path


def test_survival_produces_output_csv(analysis_ready_csv, tmp_path):
    runner = CliRunner()
    output = tmp_path / "cox_results.csv"
    result = runner.invoke(
        analyze,
        [
            "survival",
            "--input",
            str(analysis_ready_csv),
            "--outcome",
            "OS_months",
            "--event",
            "OS_event",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    rows = list(csv.DictReader(output.open()))
    assert len(rows) >= 1
    assert "feature" in rows[0]
    assert "coef" in rows[0]
    assert "p" in rows[0]
    assert "hr" in rows[0]


def test_survival_shows_summary_in_console(analysis_ready_csv, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "survival",
            "--input",
            str(analysis_ready_csv),
            "--outcome",
            "OS_months",
            "--event",
            "OS_event",
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 0
    # Should print a table with feature and HR columns
    assert "HR" in result.output or "hr" in result.output.lower()


def test_survival_fails_on_missing_outcome_column(analysis_ready_csv, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        analyze,
        [
            "survival",
            "--input",
            str(analysis_ready_csv),
            "--outcome",
            "NONEXISTENT",
            "--event",
            "OS_event",
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code != 0
    assert "NONEXISTENT" in result.output or "not found" in result.output.lower()
