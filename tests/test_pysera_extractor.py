"""Tests for PyseraExtractor (qradiomics/pysera_extractor.py).

pysera is not an installed dependency in this environment, so pysera.process_batch
is mocked by patching sys.modules['pysera'] with a fake module.
"""

import csv
import sys
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pandas as pd

from qradiomics.pysera_extractor import PyseraExtractor


def _write_manifest(path, rows):
    fieldnames = ["patient_id", "modality", "image_path", "mask_path"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fake_pysera_module(process_batch):
    module = MagicMock()
    module.process_batch = process_batch
    return module


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_happy_path(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "img1.nrrd", "mask_path": "mask1.nrrd"},
        {"patient_id": "P002", "modality": "CT", "image_path": "img2.nrrd", "mask_path": "mask2.nrrd"},
    ])
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    features_df = pd.DataFrame([{"firstorder_Mean": 5.0, "glcm_Contrast": 1.2}])
    process_batch = MagicMock(return_value={"success": True, "features_extracted": features_df})
    fake_module = _fake_pysera_module(process_batch)

    with patch.dict(sys.modules, {"pysera": fake_module}):
        extractor = PyseraExtractor()
        result = extractor.run_extraction(uuid4(), manifest, job_dir, {}, jobs=1)

    assert result["status"] == "extracted"
    assert result["patients_processed"] == 2
    assert result["patients_failed"] == 0
    assert result["patients_skipped"] == 0
    assert process_batch.call_count == 2

    features_csv = job_dir / "features.csv"
    assert features_csv.exists()
    rows = _read_csv(features_csv)
    assert len(rows) == 2
    assert "patient_id" in rows[0]
    assert "firstorder_Mean" in rows[0]
    assert "glcm_Contrast" in rows[0]
    assert {r["patient_id"] for r in rows} == {"P001", "P002"}


def test_skip_empty_mask(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "img1.nrrd", "mask_path": ""},
        {"patient_id": "P002", "modality": "CT", "image_path": "img2.nrrd", "mask_path": "mask2.nrrd"},
    ])
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    features_df = pd.DataFrame([{"firstorder_Mean": 5.0}])
    process_batch = MagicMock(return_value={"success": True, "features_extracted": features_df})
    fake_module = _fake_pysera_module(process_batch)

    with patch.dict(sys.modules, {"pysera": fake_module}):
        extractor = PyseraExtractor()
        result = extractor.run_extraction(uuid4(), manifest, job_dir, {}, jobs=1)

    assert result["patients_skipped"] == 1
    assert result["patients_processed"] == 1
    assert process_batch.call_count == 1


def test_partial_failure_success_flag(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "bad.nrrd", "mask_path": "mask1.nrrd"},
        {"patient_id": "P002", "modality": "CT", "image_path": "good.nrrd", "mask_path": "mask2.nrrd"},
    ])
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    features_df = pd.DataFrame([{"firstorder_Mean": 5.0}])

    def fake_process_batch(**kwargs):
        if kwargs["image_input"] == "bad.nrrd":
            return {"success": False, "error": "boom"}
        return {"success": True, "features_extracted": features_df}

    fake_module = _fake_pysera_module(MagicMock(side_effect=fake_process_batch))

    with patch.dict(sys.modules, {"pysera": fake_module}):
        extractor = PyseraExtractor()
        result = extractor.run_extraction(uuid4(), manifest, job_dir, {}, jobs=1)

    assert result["patients_failed"] == 1
    assert result["patients_processed"] == 1

    rows = _read_csv(job_dir / "features.csv")
    assert len(rows) == 1
    assert rows[0]["patient_id"] == "P002"


def test_process_batch_exception_isolated(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "boom.nrrd", "mask_path": "mask1.nrrd"},
        {"patient_id": "P002", "modality": "CT", "image_path": "good.nrrd", "mask_path": "mask2.nrrd"},
    ])
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    features_df = pd.DataFrame([{"firstorder_Mean": 5.0}])

    def fake_process_batch(**kwargs):
        if kwargs["image_input"] == "boom.nrrd":
            raise RuntimeError("kaboom")
        return {"success": True, "features_extracted": features_df}

    fake_module = _fake_pysera_module(MagicMock(side_effect=fake_process_batch))

    with patch.dict(sys.modules, {"pysera": fake_module}):
        extractor = PyseraExtractor()
        result = extractor.run_extraction(uuid4(), manifest, job_dir, {}, jobs=1)

    assert result["patients_failed"] == 1
    assert result["patients_processed"] == 1
    assert result["status"] == "extracted"


def test_pysera_import_error_returns_error_status(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "img1.nrrd", "mask_path": "mask1.nrrd"},
    ])
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    # pysera is not installed in this environment; make sure no leftover
    # mock module from another test lingers in sys.modules either.
    monkeypatch.delitem(sys.modules, "pysera", raising=False)

    extractor = PyseraExtractor()
    result = extractor.run_extraction(uuid4(), manifest, job_dir, {}, jobs=1)

    assert result["status"] == "error"
    assert "error" in result
