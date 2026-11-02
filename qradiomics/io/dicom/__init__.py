"""qradiomics.io.dicom — DICOM series + RTSTRUCT + seg.nrrd loaders + PET SUV.

Loaders return :class:`SimpleITK.Image` objects so they plug directly
into :mod:`qradiomics.atomic`. The DICOM series loader supports
``SeriesInstanceUID`` filtering — required for 4D-CT directories that
mix ten breathing phases + MIP + AvgIP into a single folder.

For PET (Modality == "PT") series, :func:`read_pet_suv` converts raw
DICOM activity values (Classic PET ``1.2.840.10008.5.1.4.1.1.128`` and
Enhanced PET ``1.2.840.10008.5.1.4.1.1.130``) to SUV following the
IBSI-SUV v3.1 reference standard and QIBA vendor-neutral fallback rules.
"""

from .pet_suv import (
    SUPPORTED_SUV_TYPES,
    SUVResult,
    compute_body_norm_factor_g,
    compute_suv_factor,
    read_pet_suv,
)
from .preamble import fix_dicom_preamble
from .rtstruct import load_rtstruct_roi
from .segmentations import list_segments_in_seg_nrrd, load_segment_from_seg_nrrd
from .series import list_series_in_directory, load_dicom_series

__all__ = [
    "SUPPORTED_SUV_TYPES",
    "SUVResult",
    "compute_body_norm_factor_g",
    "compute_suv_factor",
    "fix_dicom_preamble",
    "list_segments_in_seg_nrrd",
    "list_series_in_directory",
    "load_dicom_series",
    "load_rtstruct_roi",
    "load_segment_from_seg_nrrd",
    "read_pet_suv",
]
