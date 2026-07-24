"""Tests for RtoolsExtractor (qradiomics/rtools_extractor.py)."""

import csv
import subprocess
from unittest.mock import patch
from uuid import uuid4

from qradiomics.rtools_extractor import RtoolsExtractor, parse_feature_txt


def _write_manifest(path, rows):
    fieldnames = ["patient_id", "modality", "image_path", "mask_path"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_binary_not_found(tmp_path, monkeypatch):
    missing_bin = tmp_path / "no" / "such" / "FeatureExtraction"
    monkeypatch.setenv("QRADIOMICS_RTOOLS_BIN", str(missing_bin))

    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "img1.nrrd", "mask_path": "mask1.nrrd"},
    ])
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    extractor = RtoolsExtractor()
    result = extractor.run_extraction(uuid4(), manifest, job_dir, {}, jobs=1)

    assert result["status"] == "error"
    assert "error" in result
    assert str(missing_bin) in result["error"]


def _make_fake_binary(tmp_path):
    fake_bin = tmp_path / "FeatureExtraction"
    fake_bin.write_text("#!/bin/sh\n")
    return fake_bin


def test_happy_path_two_patients(tmp_path, monkeypatch):
    fake_bin = _make_fake_binary(tmp_path)
    monkeypatch.setenv("QRADIOMICS_RTOOLS_BIN", str(fake_bin))

    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "img1.nrrd", "mask_path": "mask1.nrrd"},
        {"patient_id": "P002", "modality": "CT", "image_path": "img2.nrrd", "mask_path": "mask2.nrrd"},
    ])
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    sample_text = (
        "firstorder_Mean=5.5\n"
        "PrincipalAxes=1 0 0\n"
        "Eigenvectors=0 1 0\n"
        "shape_Volume=100.0[extra]\n"
        "bad_value=notanumber\n"
    )

    def fake_run(args, capture_output=None, timeout=None, check=None):
        out_txt = args[3]
        with open(out_txt, "w") as f:
            f.write(sample_text)
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        extractor = RtoolsExtractor()
        result = extractor.run_extraction(uuid4(), manifest, job_dir, {}, jobs=1)

    assert mock_run.call_count == 2
    assert result["status"] == "extracted"
    assert result["patients_processed"] == 2
    assert result["patients_failed"] == 0
    assert result["patients_skipped"] == 0

    features_csv = job_dir / "features.csv"
    assert features_csv.exists()
    rows = {r["patient_id"]: r for r in _read_csv(features_csv)}
    assert set(rows) == {"P001", "P002"}
    assert "PrincipalAxes" not in rows["P001"]
    assert "Eigenvectors" not in rows["P001"]
    assert "bad_value" not in rows["P001"]
    assert float(rows["P001"]["firstorder_Mean"]) == 5.5
    assert float(rows["P001"]["shape_Volume"]) == 100.0


def test_subprocess_failure_isolated(tmp_path, monkeypatch):
    fake_bin = _make_fake_binary(tmp_path)
    monkeypatch.setenv("QRADIOMICS_RTOOLS_BIN", str(fake_bin))

    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "bad.nrrd", "mask_path": "mask1.nrrd"},
        {"patient_id": "P002", "modality": "CT", "image_path": "good.nrrd", "mask_path": "mask2.nrrd"},
    ])
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    def fake_run(args, capture_output=None, timeout=None, check=None):
        image_path = args[1]
        if image_path == "bad.nrrd":
            raise subprocess.CalledProcessError(1, args, stderr=b"boom")
        out_txt = args[3]
        with open(out_txt, "w") as f:
            f.write("firstorder_Mean=1.0\n")
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    with patch("subprocess.run", side_effect=fake_run):
        extractor = RtoolsExtractor()
        result = extractor.run_extraction(uuid4(), manifest, job_dir, {}, jobs=1)

    assert result["patients_failed"] == 1
    assert result["patients_processed"] == 1

    rows = _read_csv(job_dir / "features.csv")
    assert len(rows) == 1
    assert rows[0]["patient_id"] == "P002"


def test_skip_empty_mask(tmp_path, monkeypatch):
    fake_bin = _make_fake_binary(tmp_path)
    monkeypatch.setenv("QRADIOMICS_RTOOLS_BIN", str(fake_bin))

    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "img1.nrrd", "mask_path": ""},
        {"patient_id": "P002", "modality": "CT", "image_path": "img2.nrrd", "mask_path": "mask2.nrrd"},
    ])
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    def fake_run(args, capture_output=None, timeout=None, check=None):
        out_txt = args[3]
        with open(out_txt, "w") as f:
            f.write("firstorder_Mean=1.0\n")
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        extractor = RtoolsExtractor()
        result = extractor.run_extraction(uuid4(), manifest, job_dir, {}, jobs=1)

    assert result["patients_skipped"] == 1
    assert result["patients_processed"] == 1
    assert mock_run.call_count == 1
    called_image_paths = [c.args[0][1] for c in mock_run.call_args_list]
    assert "img1.nrrd" not in called_image_paths
    assert "img2.nrrd" in called_image_paths


def test_parse_feature_txt():
    sample = "\n".join([
        "firstorder_Mean=5.5",
        "PrincipalAxes=1 0 0",
        "Eigenvectors=0 1 0",
        "shape_Volume=1.5[extra stuff]",
        "bad_key=notanumber",
        "",
        "no_equals_sign_here",
    ])

    feats = parse_feature_txt(sample)

    assert feats == {"firstorder_Mean": 5.5, "shape_Volume": 1.5}
    assert "PrincipalAxes" not in feats
    assert "Eigenvectors" not in feats
    assert "bad_key" not in feats
    assert "no_equals_sign_here" not in feats


def test_jobs_parallel_path(tmp_path, monkeypatch):
    fake_bin = _make_fake_binary(tmp_path)
    monkeypatch.setenv("QRADIOMICS_RTOOLS_BIN", str(fake_bin))

    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "img1.nrrd", "mask_path": "mask1.nrrd"},
        {"patient_id": "P002", "modality": "CT", "image_path": "img2.nrrd", "mask_path": "mask2.nrrd"},
    ])
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    def fake_run(args, capture_output=None, timeout=None, check=None):
        out_txt = args[3]
        with open(out_txt, "w") as f:
            f.write("firstorder_Mean=1.0\n")
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    with patch("subprocess.run", side_effect=fake_run):
        extractor = RtoolsExtractor()
        result = extractor.run_extraction(uuid4(), manifest, job_dir, {}, jobs=2)

    assert result["patients_processed"] == 2
    assert result["patients_failed"] == 0
    rows = _read_csv(job_dir / "features.csv")
    assert {r["patient_id"] for r in rows} == {"P001", "P002"}
