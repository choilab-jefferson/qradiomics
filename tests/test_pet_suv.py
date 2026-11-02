"""Tests for qradiomics.io.dicom.pet_suv — IBSI-SUV v3.1 & QIBA SUV computation."""

import csv
import datetime
import math
import os
from pathlib import Path

import pytest
from pydicom.dataset import Dataset

from qradiomics.io.dicom.pet_suv import (
    _geometry,
    _parse_time,
    _slice_z,
    compute_body_norm_factor_g,
    compute_suv_factor,
    read_pet_suv,
)


def _radiopharm(injection_time="110000", half_life=6588.0, dose=370_000_000.0):
    ri = Dataset()
    ri.RadiopharmaceuticalStartTime = injection_time
    ri.RadionuclideHalfLife = half_life
    ri.RadionuclideTotalDose = dose
    return ri


class TestSUVBranches:
    def test_bqml_full_metadata(self):
        ds = Dataset()
        ds.PatientWeight = 70.0
        ds.Units = "BQML"
        ds.SeriesTime = "120000"
        ds.AcquisitionTime = "120030"
        ds.RadiopharmaceuticalInformationSequence = [_radiopharm()]
        factor, estimated = compute_suv_factor(ds)
        expected = 70000 / (370_000_000 * math.exp(-math.log(2) * 3600 / 6588))
        assert abs(factor - expected) / expected < 1e-9
        assert estimated is False

    def test_gml_pass_through(self):
        ds = Dataset()
        ds.PatientWeight = 70.0
        ds.Units = "GML"
        f, _ = compute_suv_factor(ds)
        assert f == 1.0

    def test_cnts_vendor_private_tag(self):
        ds = Dataset()
        ds.PatientWeight = 70.0
        ds.Units = "CNTS"
        ds.add_new((0x7053, 0x1000), "DS", "2.5e-5")
        f, _ = compute_suv_factor(ds)
        assert abs(f - 2.5e-5) < 1e-12

    def test_cnts_without_vendor_tag_falls_back(self):
        ds = Dataset()
        ds.PatientWeight = 70.0
        ds.Units = "CNTS"
        ds.SeriesTime = "120000"
        ds.AcquisitionTime = "120000"
        ds.RadiopharmaceuticalInformationSequence = [_radiopharm()]
        f, _ = compute_suv_factor(ds)
        assert f > 0

    def test_unknown_units_returns_identity(self):
        ds = Dataset()
        ds.PatientWeight = 70.0
        ds.Units = "FOO"
        f, est = compute_suv_factor(ds)
        assert f == 1.0
        assert est is True

    def test_missing_weight_uses_fallback(self):
        ds = Dataset()
        ds.Units = "BQML"
        ds.SeriesTime = "120000"
        ds.AcquisitionTime = "120000"
        ds.RadiopharmaceuticalInformationSequence = [_radiopharm()]
        _, est = compute_suv_factor(ds)
        assert est is True  # fallback weight = 75 kg

    def test_zero_weight_uses_fallback(self):
        ds = Dataset()
        ds.PatientWeight = 0.0
        ds.Units = "BQML"
        ds.SeriesTime = "120000"
        ds.AcquisitionTime = "120000"
        ds.RadiopharmaceuticalInformationSequence = [_radiopharm()]
        _, est = compute_suv_factor(ds)
        assert est is True

    def test_missing_radiopharm_seq_uses_fallback(self):
        ds = Dataset()
        ds.PatientWeight = 70.0
        ds.Units = "BQML"
        ds.SeriesTime = "120000"
        ds.AcquisitionTime = "120000"
        _, est = compute_suv_factor(ds)
        assert est is True


class TestIBSISUVStandard:
    """Tests covering IBSI-SUV v3.1 rules (https://theibsi.github.io/ibsi-suv/)."""

    def test_weight_in_grams_auto_converted(self):
        ds = Dataset()
        ds.PatientWeight = 75000.0  # DRO_1_1: weight recorded in grams
        assert compute_body_norm_factor_g(ds, "BW") == pytest.approx(75000.0)

    def test_body_norm_equations_male_female_other(self):
        ds = Dataset()
        ds.PatientWeight = 75.0
        ds.PatientSize = 1.75
        ds.PatientSex = "M"
        assert compute_body_norm_factor_g(ds, "LBM") == pytest.approx(
            (1.10 * 75.0 - 120.0 * (75.0 / 175.0) ** 2) * 1000.0
        )
        assert compute_body_norm_factor_g(ds, "LBMJAMES128") == pytest.approx(
            (1.10 * 75.0 - 128.0 * (75.0 / 175.0) ** 2) * 1000.0
        )
        bmi = 75.0 / (1.75**2)
        assert compute_body_norm_factor_g(ds, "LBMJANMA") == pytest.approx(
            ((9270.0 * 75.0) / (6680.0 + 216.0 * bmi)) * 1000.0
        )
        assert compute_body_norm_factor_g(ds, "IBW") == pytest.approx(
            (48.0 + 1.06 * (175.0 - 152.0)) * 1000.0
        )
        assert compute_body_norm_factor_g(ds, "BSA") == pytest.approx(
            0.007184 * (175.0**0.725) * (75.0**0.425) * 10000.0
        )

        ds.PatientSex = "F"
        assert compute_body_norm_factor_g(ds, "LBM") == pytest.approx(
            (1.07 * 75.0 - 148.0 * (75.0 / 175.0) ** 2) * 1000.0
        )

        ds.PatientSex = "O"
        m_ibw = (48.0 + 1.06 * (175.0 - 152.0)) * 1000.0
        f_ibw = (45.5 + 0.91 * (175.0 - 152.0)) * 1000.0
        assert compute_body_norm_factor_g(ds, "IBW") == pytest.approx(0.5 * (m_ibw + f_ibw))

    def test_gml_lbm_to_suvbw_conversion(self):
        ds = Dataset()
        ds.PatientWeight = 75.0
        ds.PatientSize = 1.75
        ds.PatientSex = "M"
        ds.Units = "GML"
        ds.SUVType = "LBM"
        factor, est = compute_suv_factor(ds, strict=True)
        lbm_g = (1.10 * 75.0 - 120.0 * (75.0 / 175.0) ** 2) * 1000.0
        assert est is False
        assert factor == pytest.approx(75000.0 / lbm_g)

    def test_cm2ml_bsa_to_suvbw_conversion(self):
        ds = Dataset()
        ds.PatientWeight = 75.0
        ds.PatientSize = 1.75
        ds.Units = "CM2ML"
        ds.SUVType = "BSA"
        factor, est = compute_suv_factor(ds, strict=True)
        bsa_cm2 = 0.007184 * (175.0**0.725) * (75.0**0.425) * 10000.0
        assert est is False
        assert factor == pytest.approx(75000.0 / bsa_cm2)

    def test_mbq_dose_auto_scaled_to_bq(self):
        ds = Dataset()
        ds.PatientWeight = 75.0
        ds.Units = "BQML"
        ds.DecayCorrection = "ADMIN"
        ds.RadiopharmaceuticalInformationSequence = [
            _radiopharm(dose=300.0)  # 300 MBq (< 10^4) -> 300e6 Bq
        ]
        factor, est = compute_suv_factor(ds, strict=True)
        assert est is False
        assert factor == pytest.approx(75000.0 / 300_000_000.0)

    def test_philips_cnts_activity_factor_preferred_over_suv_factor(self):
        ds = Dataset()
        ds.Manufacturer = "Philips Medical Systems"
        ds.PatientWeight = 75.0
        ds.Units = "CNTS"
        ds.DecayCorrection = "ADMIN"
        ds.RadiopharmaceuticalInformationSequence = [_radiopharm(dose=300_000_000.0)]
        ds.add_new((0x7053, 0x1009), "DS", "2.0")  # Activity scale factor
        ds.add_new((0x7053, 0x1000), "DS", "999.0")  # Should be ignored when 7053,1009 > 0
        factor, est = compute_suv_factor(ds, strict=True)
        assert est is False
        assert factor == pytest.approx(2.0 * 75000.0 / 300_000_000.0)

    def test_decay_correction_none_uses_frame_average_time(self):
        ds = Dataset()
        ds.PatientWeight = 75.0
        ds.Units = "BQML"
        ds.DecayCorrection = "NONE"
        ds.AcquisitionDate = "20210407"
        ds.AcquisitionTime = "120000"
        ds.ActualFrameDuration = 300000  # 300 s = 5 min
        ds.RadiopharmaceuticalInformationSequence = [
            _radiopharm(injection_time="110000", half_life=6588.0, dose=300_000_000.0)
        ]
        factor, est = compute_suv_factor(ds, strict=True)
        assert est is False
        lam = math.log(2.0) / 6588.0
        t_ave = (1.0 / lam) * math.log((lam * 300.0) / (1.0 - math.exp(-lam * 300.0)))
        expected = 75000.0 / (300_000_000.0 * math.exp(-lam * (3600.0 + t_ave)))
        assert factor == pytest.approx(expected, rel=1e-9)

    def test_slice_z_prioritizes_image_position_patient_over_slice_location(self):
        """Regression test for ACRIN-NSCLC-FDG-PET-157 (Validation_09) where
        SliceLocation (-1406.7) has opposite sign to ImagePositionPatient[2] (+1406.7).
        """
        ds = Dataset()
        ds.SliceLocation = -1406.7
        ds.ImagePositionPatient = ["-344.8", "-350.0", "1406.7"]
        assert _slice_z(ds) == pytest.approx(1406.7)

    def test_strict_mode_raises_on_missing_or_invalid_tags(self):
        ds = Dataset()
        ds.Units = "BQML"
        with pytest.raises(ValueError, match="PatientWeight"):
            compute_suv_factor(ds, strict=True)


def _make_ds(iop, ipp, pixel_spacing=(1.0, 1.0)):
    ds = Dataset()
    ds.ImageOrientationPatient = [str(v) for v in iop]
    ds.ImagePositionPatient = [str(v) for v in ipp]
    ds.PixelSpacing = [str(pixel_spacing[0]), str(pixel_spacing[1])]
    return ds


class TestGeometrySliceDirection:
    """slice_dir in _geometry must agree with the physical stacking order."""

    def test_hfs_normal_no_flip(self):
        # HFS: row=[1,0,0], col=[0,1,0] → cross=[0,0,1] (same as ascending z)
        iop = [1, 0, 0, 0, 1, 0]
        rows = [
            (_make_ds(iop, [0, 0, -50]), -50.0),
            (_make_ds(iop, [0, 0, 0]), 0.0),
            (_make_ds(iop, [0, 0, 50]), 50.0),
        ]
        _, _, direction = _geometry(rows)
        # Third column of the direction matrix is slice_dir
        slice_dir_z = direction[8]  # direction[2], direction[5], direction[8]
        assert slice_dir_z > 0, "HFS: slice_dir should point toward +z (head)"

    def test_hfp_cross_product_flipped(self):
        # HFP: row=[1,0,0], col=[0,-1,0] → cross=[0,0,-1] but stacking is ascending z
        # The fix should negate slice_dir so it points toward +z.
        iop = [1, 0, 0, 0, -1, 0]
        rows = [
            (_make_ds(iop, [0, 0, -50]), -50.0),
            (_make_ds(iop, [0, 0, 0]), 0.0),
            (_make_ds(iop, [0, 0, 50]), 50.0),
        ]
        _, _, direction = _geometry(rows)
        slice_dir_z = direction[8]
        assert slice_dir_z > 0, "HFP: slice_dir must be flipped to agree with +z stacking"


class TestTimeParser:
    def test_parses_hhmmss(self):
        assert _parse_time("120000") == datetime.datetime.strptime("120000", "%H%M%S")

    def test_parses_hhmmss_fff(self):
        t = _parse_time("120000.123")
        assert t.microsecond == 123000

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_time("garbage")


@pytest.mark.skipif(
    not os.environ.get("IBSI_SUV_DRO_DIR"),
    reason="Set IBSI_SUV_DRO_DIR to run full 58-DRO IBSI-SUV v3.1 reference suite",
)
def test_ibsi_suv_full_58_dro_suite():
    import numpy as np
    import SimpleITK as sitk

    dro_root = Path(os.environ["IBSI_SUV_DRO_DIR"])
    with open(dro_root / "DRO_list.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 58
    for r in rows:
        did = r["ID"]
        pt_dir = dro_root / "data" / did / "PT"
        mask_files = list((dro_root / "data" / did / "mask").glob("*.nii*"))
        exp = r["SUVmax_expected"]
        if exp == "ERROR":
            with pytest.raises(ValueError):
                read_pet_suv(pt_dir, strict=True)
        else:
            res = read_pet_suv(pt_dir, strict=True)
            assert res.estimated is False
            suv_arr = sitk.GetArrayFromImage(res.image)
            mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_files[0]))) > 0
            roi = suv_arr[mask_arr]
            assert f"{float(np.min(roi)):.2f}" == "0.20", f"{did} min mismatch"
            assert f"{float(np.median(roi)):.2f}" == "1.00", f"{did} median mismatch"
            assert f"{float(np.max(roi)):.2f}" == "4.00", f"{did} max mismatch"


@pytest.mark.skipif(
    not os.environ.get("IBSI_SUV_VALIDATION_DIR"),
    reason="Set IBSI_SUV_VALIDATION_DIR to run 16-series clinical validation suite",
)
def test_ibsi_suv_full_16_clinical_validation_suite():
    import numpy as np
    import SimpleITK as sitk

    val_root = Path(os.environ["IBSI_SUV_VALIDATION_DIR"])
    data_dir = val_root / "data" if (val_root / "data").is_dir() else val_root
    val_dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and p.name.lower().startswith("validation_")
    )

    assert len(val_dirs) == 16
    error_ids = {"validation_06", "validation_08", "validation_12", "validation_14"}
    for vdir in val_dirs:
        vid = vdir.name.lower()
        pt_dir = vdir / "PT"
        mask_files = list((vdir / "mask").glob("*.nii*"))
        if vid in error_ids:
            with pytest.raises(ValueError):
                read_pet_suv(pt_dir, strict=True)
        else:
            res = read_pet_suv(pt_dir, strict=True)
            assert res.estimated is False
            suv_arr = sitk.GetArrayFromImage(res.image)
            mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_files[0]))) > 0
            roi = suv_arr[mask_arr]
            assert float(np.max(roi)) > 0.5, f"{vid} unexpected SUVmax"
            assert float(np.mean(roi)) > 0.2, f"{vid} unexpected SUVmean"
