"""Tests for qr results merge command."""

import csv

import pytest
from click.testing import CliRunner

from qradiomics.cli.commands.results import results


@pytest.fixture
def tmp_features(tmp_path):
    f = tmp_path / "features.csv"
    f.write_text("patient_id,original_shape_Elongation\n" "PT001,0.73\n" "PT002,0.81\n")
    return f


@pytest.fixture
def tmp_clinical_days(tmp_path):
    """Clinical CSV with time in days (NSCLC-style)."""
    c = tmp_path / "clinical.csv"
    c.write_text("PatientID,Survival.time,deadstatus.event\n" "PT001,365,1\n" "PT002,730,0\n")
    return c


@pytest.fixture
def tmp_clinical_months(tmp_path):
    """Clinical CSV with time already in months (Cetuximab-style)."""
    c = tmp_path / "clinical_months.csv"
    c.write_text("patid,survival_months,survival_status\n" "PT001,12.0,1\n" "PT002,24.0,0\n")
    return c


def test_merge_produces_analysis_ready_csv(tmp_features, tmp_clinical_days, tmp_path):
    runner = CliRunner()
    output = tmp_path / "analysis_ready.csv"
    result = runner.invoke(
        results,
        [
            "merge",
            "--features",
            str(tmp_features),
            "--clinical",
            str(tmp_clinical_days),
            "--clinical-id-col",
            "PatientID",
            "--time-col",
            "Survival.time",
            "--event-col",
            "deadstatus.event",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 2
    assert "OS_months" in rows[0]
    assert "OS_event" in rows[0]
    assert "original_shape_Elongation" in rows[0]
    # 365 days / 30.44 ≈ 11.99
    assert abs(float(rows[0]["OS_months"]) - 365 / 30.44) < 0.01


def test_merge_months_column_not_converted(tmp_features, tmp_clinical_months, tmp_path):
    """survival_months (median ~18) stays as-is — not divided by 30.44."""
    runner = CliRunner()
    output = tmp_path / "out.csv"
    result = runner.invoke(
        results,
        [
            "merge",
            "--features",
            str(tmp_features),
            "--clinical",
            str(tmp_clinical_months),
            "--clinical-id-col",
            "patid",
            "--time-col",
            "survival_months",
            "--event-col",
            "survival_status",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(output.open()))
    assert float(rows[0]["OS_months"]) == 12.0  # not converted


def test_merge_warns_unmatched_patients(tmp_features, tmp_path):
    """When a patient is in features but not clinical, merge warns and continues."""
    clinical = tmp_path / "clinical_small.csv"
    clinical.write_text("PatientID,Survival.time,deadstatus.event\n" "PT001,365,1\n")
    runner = CliRunner()
    output = tmp_path / "out.csv"
    result = runner.invoke(
        results,
        [
            "merge",
            "--features",
            str(tmp_features),
            "--clinical",
            str(clinical),
            "--clinical-id-col",
            "PatientID",
            "--time-col",
            "Survival.time",
            "--event-col",
            "deadstatus.event",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 1  # only PT001 matched
    assert "unmatched" in result.output.lower() or "PT002" in result.output


def test_merge_fails_gracefully_on_missing_features_file(tmp_clinical_days, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        results,
        [
            "merge",
            "--features",
            str(tmp_path / "nonexistent.csv"),
            "--clinical",
            str(tmp_clinical_days),
            "--output",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code != 0
