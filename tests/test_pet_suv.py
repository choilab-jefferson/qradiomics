"""Tests for qradiomics.io.dicom.pet_suv — SUVbw factor computation."""
import datetime
import math

import pytest
from pydicom.dataset import Dataset

from qradiomics.io.dicom.pet_suv import compute_suv_factor, _parse_time, _geometry


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
            (_make_ds(iop, [0, 0,   0]),   0.0),
            (_make_ds(iop, [0, 0,  50]),  50.0),
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
            (_make_ds(iop, [0, 0,   0]),   0.0),
            (_make_ds(iop, [0, 0,  50]),  50.0),
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
