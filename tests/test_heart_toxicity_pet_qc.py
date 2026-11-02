"""Tests for pipelines/heart_toxicity/pet_suv_qc.py (IBSI-SUV v3.1 QC ledger)."""

import csv
import json

import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

qc = pytest.importorskip("pipelines.heart_toxicity.pet_suv_qc")

PET_SOP = "1.2.840.10008.5.1.4.1.1.128"


def _write_series(root, name, *, weight=70.0, dose=370_000_000.0):
    d = root / name
    d.mkdir(parents=True)
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = PET_SOP
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = PET_SOP
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Units = "BQML"
    ds.DecayCorrection = "ADMIN"
    ds.SeriesTime = "120000"
    ds.AcquisitionTime = "120030"
    if weight is not None:
        ds.PatientWeight = weight
    ri = Dataset()
    ri.RadiopharmaceuticalStartTime = "110000"
    ri.RadionuclideHalfLife = 6588.0
    if dose is not None:
        ri.RadionuclideTotalDose = dose
    ds.RadiopharmaceuticalInformationSequence = [ri]
    ds.save_as(str(d / "s1.dcm"), enforce_file_format=True)
    return d


def test_valid_series_passes(tmp_path):
    row = qc.qc_series(_write_series(tmp_path, "ok"))
    assert row.status == "PASS" and row.flags == ""
    assert row.suv_factor and row.suv_factor > 0


def test_missing_weight_flagged(tmp_path):
    row = qc.qc_series(_write_series(tmp_path, "nw", weight=None))
    assert row.status == "FAIL"
    assert "missing_patient_weight" in row.flags
    assert "PatientWeight" in row.message


def test_missing_dose_flagged(tmp_path):
    row = qc.qc_series(_write_series(tmp_path, "nd", dose=None))
    assert row.status == "FAIL"
    assert "missing_total_dose" in row.flags


def test_mbq_scaling_anomaly_warns(tmp_path):
    row = qc.qc_series(_write_series(tmp_path, "mbq", dose=300.0))
    assert row.status == "WARN"  # pet_suv auto-scales, but ledger flags it
    assert row.flags == "mbq_unit_scaling"


def test_unreadable_series(tmp_path):
    (tmp_path / "empty").mkdir()
    assert qc.qc_series(tmp_path / "empty").flags == "unreadable"


def test_ledger_and_manifest_cli(tmp_path):
    ok = _write_series(tmp_path, "ok")
    bad = _write_series(tmp_path, "bad", weight=None)
    man = tmp_path / "coh" / "manifest.csv"
    man.parent.mkdir()
    man.write_text(f"patient_id,pet_dir\na,{ok}\nb,{bad}\n")
    out = tmp_path / "out"
    assert qc.main(["--manifest", str(man), "--out", str(out)]) == 0
    assert qc.main(["--manifest", str(man), "--out", str(out), "--fail-on-fail"]) == 1
    rows = list(csv.DictReader((out / "pet_suv_qc.csv").open()))
    assert [r["status"] for r in rows] == ["PASS", "FAIL"]
    summ = json.loads((out / "pet_suv_qc_summary.json").read_text())
    assert summ["n_fail"] == 1 and summ["flag_counts"]["missing_patient_weight"] == 1
