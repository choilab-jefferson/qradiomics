"""Tests for qradiomics.io.dicom.pet_suv — SUVbw factor computation."""
import datetime
import math

import pytest
from pydicom.dataset import Dataset

from qradiomics.io.dicom.pet_suv import compute_suv_factor, _parse_time


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


class TestTimeParser:
    def test_parses_hhmmss(self):
        assert _parse_time("120000") == datetime.datetime.strptime("120000", "%H%M%S")

    def test_parses_hhmmss_fff(self):
        t = _parse_time("120000.123")
        assert t.microsecond == 123000

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_time("garbage")
