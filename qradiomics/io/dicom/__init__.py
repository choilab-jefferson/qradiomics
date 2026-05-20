"""qradiomics.io.dicom — DICOM series + RTSTRUCT + seg.nrrd loaders.

Loaders return :class:`SimpleITK.Image` objects so they plug directly
into :mod:`qradiomics.atomic`. The DICOM series loader supports
``SeriesInstanceUID`` filtering — required for 4D-CT directories that
mix ten breathing phases + MIP + AvgIP into a single folder.
"""

from .pet_suv import compute_suv_factor, read_pet_suv
from .rtstruct import load_rtstruct_roi
from .segmentations import list_segments_in_seg_nrrd, load_segment_from_seg_nrrd
from .series import list_series_in_directory, load_dicom_series

__all__ = [
    "compute_suv_factor",
    "list_segments_in_seg_nrrd",
    "list_series_in_directory",
    "load_dicom_series",
    "load_rtstruct_roi",
    "load_segment_from_seg_nrrd",
    "read_pet_suv",
]
