"""Tests for `qr convert manifest-from-dir` — both directory layouts."""

import csv

from click.testing import CliRunner

from qradiomics.cli.commands.convert import convert


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_manifest_from_per_patient_subdirs(tmp_path):
    """Layout 1: one subdirectory per patient."""
    for pid in ("P001", "P002"):
        d = tmp_path / pid
        d.mkdir()
        (d / f"{pid}_CT.nrrd").write_text("x")
        (d / f"{pid}_CT_mask-label.nrrd").write_text("x")

    out = tmp_path / "manifest.csv"
    result = CliRunner().invoke(
        convert,
        [
            "manifest-from-dir",
            "-d", str(tmp_path),
            "--image-glob", "*_CT.nrrd",
            "--mask-glob", "*-label.nrrd",
            "-o", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = _read(out)
    assert {r["patient_id"] for r in rows} == {"P001", "P002"}
    assert all(r["image_path"].endswith("_CT.nrrd") for r in rows)
    assert all(r["mask_path"].endswith("-label.nrrd") for r in rows)


def test_manifest_from_flat_layout(tmp_path):
    """Layout 2 (fallback): flat files sharing a per-patient prefix.

    Mirrors the Colab LUNG1 notebook, which writes {pid}_image.nrrd and
    {pid}_mask.nrrd side by side in one directory. Before the flat-layout
    fallback this exited 1 with "No image/mask pairs found".
    """
    for pid in ("LUNG1-001", "LUNG1-002", "LUNG1-003"):
        (tmp_path / f"{pid}_image.nrrd").write_text("x")
        (tmp_path / f"{pid}_mask.nrrd").write_text("x")
    # A patient missing its mask must be skipped, not paired.
    (tmp_path / "LUNG1-004_image.nrrd").write_text("x")

    out = tmp_path / "manifest.csv"
    result = CliRunner().invoke(
        convert,
        [
            "manifest-from-dir",
            "-d", str(tmp_path),
            "--image-glob", "*_image.nrrd",
            "--mask-glob", "*_mask.nrrd",
            "-o", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = _read(out)
    assert {r["patient_id"] for r in rows} == {"LUNG1-001", "LUNG1-002", "LUNG1-003"}
    for r in rows:
        assert r["image_path"].endswith(f"{r['patient_id']}_image.nrrd")
        assert r["mask_path"].endswith(f"{r['patient_id']}_mask.nrrd")


def test_manifest_from_dir_no_pairs_errors(tmp_path):
    """Neither layout matches -> non-zero exit with a clear message."""
    (tmp_path / "unrelated.txt").write_text("x")
    out = tmp_path / "manifest.csv"
    result = CliRunner().invoke(
        convert,
        [
            "manifest-from-dir",
            "-d", str(tmp_path),
            "--image-glob", "*_image.nrrd",
            "--mask-glob", "*_mask.nrrd",
            "-o", str(out),
        ],
    )
    assert result.exit_code != 0
    assert "No image/mask pairs found" in result.output
