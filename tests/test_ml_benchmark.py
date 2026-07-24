"""Tests for the qradiomics#9 unification of `qr bench` / `qr ml classify`
under the new canonical `qr ml benchmark` command.

All three commands share one engine (qradiomics.cli.commands.bench.
_run_benchmark). These tests confirm:
  * all three --help invocations still work
  * `qr bench` and `qr ml benchmark` produce equivalent CV results on a tiny
    synthetic dataset when given equivalent flags (corr-threshold disabled,
    since `bench` has no correlation pre-filter)
  * `qr ml classify` and `qr ml benchmark` produce equivalent CV results when
    given the same --corr-threshold
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from qradiomics.cli.commands.bench import bench
from qradiomics.cli.commands.ml import ml


@pytest.fixture
def tiny_classification_csv(tmp_path):
    """Small, clearly-separable binary classification feature set."""
    rng = np.random.default_rng(0)
    n = 40
    outcome = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "feature_A": rng.normal(0, 1, size=n) + outcome * 1.5,
            "feature_B": rng.normal(5, 2, size=n) + outcome * 0.5,
            "feature_C": rng.normal(2, 0.5, size=n),
            "event": outcome,
        }
    )
    path = tmp_path / "tiny_features.csv"
    df.to_csv(path, index=False)
    return path


class TestHelpTextStillWorks:
    def test_bench_help(self):
        result = CliRunner().invoke(bench, ["--help"])
        assert result.exit_code == 0
        assert "qr ml benchmark" in " ".join(result.output.split())

    def test_ml_classify_help(self):
        result = CliRunner().invoke(ml, ["classify", "--help"])
        assert result.exit_code == 0
        # click reflows/wraps the docstring, so check for the substring
        # with whitespace (including newlines) normalized.
        assert "qr ml benchmark" in " ".join(result.output.split())

    def test_ml_benchmark_help(self):
        result = CliRunner().invoke(ml, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "--tpot-repeats" in result.output
        assert "--calibrate-threshold" in result.output
        assert "--corr-threshold" in result.output


class TestBenchAndMlBenchmarkEquivalence:
    """`qr bench` never applies a correlation pre-filter; `qr ml benchmark`
    must reproduce identical CV numbers when its --corr-threshold is
    disabled (1.0) so it degenerates to the same feature set as `bench`."""

    def _common_flags(self, csv_path, output_dir):
        return [
            "--input", str(csv_path),
            "--outcome", "event",
            "--models", "LR",
            "--cv", "3",
            "--inner-cv", "2",
            "--seed", "7",
            "--output-dir", str(output_dir),
        ]

    def test_cv_results_match(self, tiny_classification_csv, tmp_path):
        runner = CliRunner()

        bench_out = tmp_path / "bench_out"
        r_bench = runner.invoke(bench, self._common_flags(tiny_classification_csv, bench_out))
        assert r_bench.exit_code == 0, r_bench.output

        benchmark_out = tmp_path / "benchmark_out"
        r_benchmark = runner.invoke(
            ml,
            ["benchmark", *self._common_flags(tiny_classification_csv, benchmark_out),
             "--corr-threshold", "1.0"],
        )
        assert r_benchmark.exit_code == 0, r_benchmark.output

        bench_summary = pd.read_csv(bench_out / "bench_cv_results.csv")
        benchmark_summary = pd.read_csv(benchmark_out / "benchmark_cv_results.csv")

        pd.testing.assert_frame_equal(
            bench_summary.reset_index(drop=True),
            benchmark_summary.reset_index(drop=True),
        )


class TestMlClassifyAndMlBenchmarkEquivalence:
    """`qr ml classify` and `qr ml benchmark` share the same feature
    pre-processing (patient_id/PatientID exclude + median fillna +
    correlation pre-filter); with the same --corr-threshold they must
    produce identical CV numbers."""

    def _common_flags(self, csv_path, output_dir):
        return [
            "--input", str(csv_path),
            "--outcome", "event",
            "--models", "LR",
            "--cv", "3",
            "--inner-cv", "2",
            "--seed", "7",
            "--corr-threshold", "0.95",
            "--output-dir", str(output_dir),
        ]

    def test_cv_results_match(self, tiny_classification_csv, tmp_path):
        runner = CliRunner()

        classify_out = tmp_path / "classify_out"
        r_classify = runner.invoke(
            ml, ["classify", *self._common_flags(tiny_classification_csv, classify_out)]
        )
        assert r_classify.exit_code == 0, r_classify.output

        benchmark_out = tmp_path / "benchmark_out"
        r_benchmark = runner.invoke(
            ml, ["benchmark", *self._common_flags(tiny_classification_csv, benchmark_out)]
        )
        assert r_benchmark.exit_code == 0, r_benchmark.output

        classify_summary = pd.read_csv(classify_out / "classify_cv_results.csv")
        benchmark_summary = pd.read_csv(benchmark_out / "benchmark_cv_results.csv")

        pd.testing.assert_frame_equal(
            classify_summary.reset_index(drop=True),
            benchmark_summary.reset_index(drop=True),
        )


class TestArtefactsProduced:
    def test_bench_writes_html_report(self, tiny_classification_csv, tmp_path):
        out = tmp_path / "bench_out"
        result = CliRunner().invoke(
            bench,
            [
                "--input", str(tiny_classification_csv),
                "--outcome", "event",
                "--models", "LR",
                "--cv", "3",
                "--inner-cv", "2",
                "--output-dir", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / "bench_report.html").exists()
        assert "Artefacts" in result.output

    def test_ml_benchmark_writes_html_report_with_own_title(self, tiny_classification_csv, tmp_path):
        out = tmp_path / "benchmark_out"
        result = CliRunner().invoke(
            ml,
            [
                "benchmark",
                "--input", str(tiny_classification_csv),
                "--outcome", "event",
                "--models", "LR",
                "--cv", "3",
                "--inner-cv", "2",
                "--output-dir", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        report = out / "benchmark_report.html"
        assert report.exists()
        assert "qr ml benchmark" in report.read_text()

    def test_ml_classify_does_not_relabel_html_title(self, tiny_classification_csv, tmp_path):
        """Regression guard: `qr ml classify`'s report keeps its historical
        "qr bench"-branded title (title_prefix default) — changing this
        would alter classify's existing output."""
        out = tmp_path / "classify_out"
        result = CliRunner().invoke(
            ml,
            [
                "classify",
                "--input", str(tiny_classification_csv),
                "--outcome", "event",
                "--models", "LR",
                "--cv", "3",
                "--inner-cv", "2",
                "--output-dir", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        report = out / "classify_report.html"
        assert report.exists()
        assert "qr bench" in report.read_text()
