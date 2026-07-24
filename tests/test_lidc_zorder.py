"""Regression test: LIDC mask z-index must match the CT volume's z-index.

_uid_to_z used to sort slices by ImagePositionPatient[2] DESCENDING while the CT
volume is built by SimpleITK's ImageSeriesReader (ASCENDING). Since the mask is
written with CopyInformation(ct), that mismatch mirrored every nodule along z
onto the wrong slices. This test builds a synthetic series with a distinct pixel
value per slice and asserts the two orderings agree.
"""
import numpy as np
import pytest

pytest.importorskip("pydicom")
pytest.importorskip("SimpleITK")

import SimpleITK as sitk  # noqa: E402
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid  # noqa: E402

from qradiomics.io.dicom import load_dicom_series  # noqa: E402
from qradiomics.io.lidc.extract import _uid_to_z  # noqa: E402


def _write_series(directory):
    """3 axial slices at z = 0/2/4 mm with pixel value 10/20/30. Returns {z: sop}."""
    series, study = generate_uid(), generate_uid()
    sops = {}
    for i, (z, val) in enumerate([(0.0, 10), (2.0, 20), (4.0, 30)]):
        ds = Dataset()
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = CTImageStorage
        sop = generate_uid()
        fm.MediaStorageSOPInstanceUID = sop
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = fm
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = sop
        sops[z] = sop
        ds.SeriesInstanceUID, ds.StudyInstanceUID, ds.Modality = series, study, "CT"
        ds.ImagePositionPatient = [0.0, 0.0, z]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.Rows = ds.Columns = 4
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 2.0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.InstanceNumber = i + 1
        ds.PixelData = np.full((4, 4), val, np.int16).tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(str(directory / f"s{i}.dcm"), enforce_file_format=True)
    return sops


def test_uid_to_z_matches_ct_volume_ordering(tmp_path):
    sops = _write_series(tmp_path)
    arr = sitk.GetArrayFromImage(load_dicom_series(str(tmp_path)))  # (z, y, x)

    # CT volume is ascending: index 0 is the z=0 slice (value 10).
    assert arr[0].mean() == 10
    assert arr[-1].mean() == 30

    uid_to_z = _uid_to_z(tmp_path)
    # For each physical slice, the mask index _uid_to_z assigns must point at the
    # CT array plane that actually holds that slice's pixels.
    for z, val in [(0.0, 10), (2.0, 20), (4.0, 30)]:
        idx = uid_to_z[sops[z]]
        assert arr[idx].mean() == val, (z, idx, arr[idx].mean())
