"""Tests for the extraction engine registry (qradiomics/extractor_registry.py)."""

import csv
from unittest.mock import patch
from uuid import uuid4

import pytest

import qradiomics.extractor_registry as extractor_registry
from qradiomics.extractor import RadiomicsExtractor
from qradiomics.extractor_registry import (
    ALL_ENGINES,
    DEFAULT_ENGINE,
    EXTRACTOR_REGISTRY,
    build_extractor,
    resolve_engines,
    run_multi_extraction,
)
from qradiomics.pysera_extractor import PyseraExtractor
from qradiomics.rtools_extractor import RtoolsExtractor


def _write_manifest(path, rows):
    fieldnames = ["patient_id", "modality", "image_path", "mask_path"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def test_registry_contents():
    assert set(EXTRACTOR_REGISTRY) == {"pyradiomics", "pysera", "rtools"}
    assert EXTRACTOR_REGISTRY["pyradiomics"] is RadiomicsExtractor
    assert EXTRACTOR_REGISTRY["pysera"] is PyseraExtractor
    assert EXTRACTOR_REGISTRY["rtools"] is RtoolsExtractor


def test_build_extractor():
    extractor = build_extractor("pyradiomics")
    assert isinstance(extractor, RadiomicsExtractor)

    with pytest.raises(KeyError):
        build_extractor("bogus")


def test_resolve_engines():
    assert resolve_engines(None) == [DEFAULT_ENGINE]
    assert resolve_engines("all") == list(ALL_ENGINES)
    assert resolve_engines("pyradiomics, pysera ,pyradiomics") == ["pyradiomics", "pysera"]
    assert resolve_engines(None, pattern_extractor="pysera") == ["pysera"]

    with pytest.raises(KeyError):
        resolve_engines("bogus")


def test_run_multi_extraction_single_engine_passthrough(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "img1.nrrd", "mask_path": "mask1.nrrd"},
    ])

    fixed_result = {
        "features_uri": f"file://{job_dir.resolve()}/features.csv",
        "feature_count": 2,
        "patients_processed": 1,
        "patients_failed": 0,
        "patients_skipped": 0,
        "status": "extracted",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }

    with patch.object(RadiomicsExtractor, "run_extraction", return_value=fixed_result) as mock_run:
        result = run_multi_extraction(
            ["pyradiomics"], uuid4(), manifest, job_dir, {}, jobs=1,
        )

    mock_run.assert_called_once()
    assert result == fixed_result
    assert not (job_dir / "_engine_pyradiomics").exists()


def test_run_multi_extraction_multi_engine_merge(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "img1.nrrd", "mask_path": "mask1.nrrd"},
        {"patient_id": "P002", "modality": "CT", "image_path": "img2.nrrd", "mask_path": "mask2.nrrd"},
    ])

    class FakeExtractorA:
        def run_extraction(self, job_id, manifest_path, job_dir, extraction_settings, jobs=1):
            with open(job_dir / "features.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["patient_id", "feat1"])
                writer.writeheader()
                writer.writerow({"patient_id": "P001", "feat1": 1.0})
                writer.writerow({"patient_id": "P002", "feat1": 2.0})
            return {
                "feature_count": 1, "patients_processed": 2,
                "patients_failed": 0, "patients_skipped": 0, "status": "extracted",
            }

    class FakeExtractorB:
        def run_extraction(self, job_id, manifest_path, job_dir, extraction_settings, jobs=1):
            with open(job_dir / "features.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["patient_id", "feat2"])
                writer.writeheader()
                # P002 overlaps with engA, P003 does NOT overlap.
                writer.writerow({"patient_id": "P002", "feat2": 3.0})
                writer.writerow({"patient_id": "P003", "feat2": 4.0})
            return {
                "feature_count": 1, "patients_processed": 2,
                "patients_failed": 0, "patients_skipped": 0, "status": "extracted",
            }

    fake_registry = {"engA": FakeExtractorA, "engB": FakeExtractorB}

    with patch.object(extractor_registry, "EXTRACTOR_REGISTRY", fake_registry):
        result = run_multi_extraction(
            ["engA", "engB"], uuid4(), manifest, job_dir, {}, jobs=1,
        )

    assert result["status"] == "extracted"
    assert "per_engine" in result
    assert set(result["per_engine"]) == {"engA", "engB"}

    features_csv = job_dir / "features.csv"
    assert features_csv.exists()
    rows = {r["patient_id"]: r for r in _read_csv(features_csv)}

    assert set(rows) == {"P001", "P002", "P003"}
    assert "engA_feat1" in rows["P001"]
    assert "engB_feat2" in rows["P001"]

    # P001 only came from engA -> engB_feat2 missing/NaN.
    assert rows["P001"]["engB_feat2"] in ("", "nan")
    # P003 only came from engB -> engA_feat1 missing/NaN.
    assert rows["P003"]["engA_feat1"] in ("", "nan")
    # P002 overlaps both engines -> both columns populated.
    assert float(rows["P002"]["engA_feat1"]) == 2.0
    assert float(rows["P002"]["engB_feat2"]) == 3.0

    assert not (job_dir / "_engine_engA").exists()
    assert not (job_dir / "_engine_engB").exists()


def test_run_multi_extraction_unknown_engine_raises_keyerror(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"patient_id": "P001", "modality": "CT", "image_path": "img1.nrrd", "mask_path": "mask1.nrrd"},
    ])

    with pytest.raises(KeyError):
        run_multi_extraction(["bogus"], uuid4(), manifest, job_dir, {}, jobs=1)
