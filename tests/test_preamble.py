"""Tests for the DICOM preamble repair utility."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError
import pytest

from qradiomics.io.dicom import fix_dicom_preamble


def test_fix_dicom_preamble_reconstructs_header():
    """Verify that fix_dicom_preamble repairs a DICOM file missing its preamble."""
    # Create a mock DICOM dataset
    ds = pydicom.Dataset()
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"  # RT Structure Set Storage
    ds.SOPInstanceUID = "1.2.3.4.5.6"
    ds.is_implicit_VR = True
    ds.is_little_endian = True

    # Save to a temporary file without preamble (no file_meta)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "corrupt.dcm"
        ds.save_as(str(test_path), write_like_original=False)

        # Strip the first 132 bytes (128-byte preamble + 4-byte 'DICM' prefix)
        # to simulate a TCIA file missing its DICOM header.
        with open(test_path, "rb") as f:
            data = f.read()
        with open(test_path, "wb") as f:
            f.write(data[132:])

        # 1. Verify that a standard read fails with InvalidDicomError
        with pytest.raises(InvalidDicomError):
            pydicom.dcmread(str(test_path))

        # 2. Run our preamble fixer
        success = fix_dicom_preamble(test_path, verbose=False)
        assert success is True

        # 3. Verify that the file can now be read successfully by standard pydicom.dcmread
        fixed_ds = pydicom.dcmread(str(test_path))
        assert fixed_ds.PatientName == "Test^Patient"
        assert fixed_ds.PatientID == "12345"
        assert fixed_ds.file_meta.MediaStorageSOPClassUID == ds.SOPClassUID
        assert fixed_ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID


def test_fix_dicom_preamble_on_already_valid_file():
    """Verify that fix_dicom_preamble leaves an already valid DICOM file untouched."""
    # Create a mock DICOM dataset with proper file meta and preamble
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6"
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    ds = pydicom.Dataset()
    ds.file_meta = file_meta
    ds.PatientName = "Valid^Patient"
    ds.PatientID = "67890"

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "valid.dcm"
        ds.save_as(str(test_path), write_like_original=False)

        # Standard read should succeed
        original_ds = pydicom.dcmread(str(test_path))
        assert original_ds.PatientID == "67890"

        # Run our preamble fixer (should return True immediately without error)
        success = fix_dicom_preamble(test_path, verbose=False)
        assert success is True

        # Verify it is still readable and identical
        fixed_ds = pydicom.dcmread(str(test_path))
        assert fixed_ds.PatientID == "67890"
        assert fixed_ds.PatientName == "Valid^Patient"
