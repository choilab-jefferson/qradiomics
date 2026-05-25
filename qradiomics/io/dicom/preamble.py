"""DICOM preamble repair utility."""

from __future__ import annotations

from pathlib import Path
import click

def fix_dicom_preamble(dicom_path: str | Path, verbose: bool = True) -> bool:
    """Verify and fix in-place the DICOM preamble of a file if missing.
    
    Returns:
        True if the file was repaired or already valid, False if it failed.
    """
    import pydicom
    from pydicom.errors import InvalidDicomError
    
    rt_path = Path(dicom_path)
    try:
        pydicom.dcmread(str(rt_path))
        # File is already valid
        return True
    except InvalidDicomError:
        if verbose:
            click.echo(f"  DICOM file is missing standard preamble. Fixing in-place: {rt_path}")
        try:
            ds = pydicom.dcmread(str(rt_path), force=True)
            from pydicom.dataset import FileMetaDataset
            if not hasattr(ds, "file_meta") or ds.file_meta is None:
                ds.file_meta = FileMetaDataset()
            if "SOPClassUID" in ds:
                ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            if "SOPInstanceUID" in ds:
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            ds.save_as(str(rt_path), write_like_original=False)
            return True
        except Exception as e:
            if verbose:
                click.echo(f"Warning: Failed to fix DICOM preamble for {rt_path}: {e}", err=True)
            return False
