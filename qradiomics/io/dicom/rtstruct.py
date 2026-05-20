"""RTSTRUCT → mask rasterization, **hole-aware**.

This is an alternative to ``rt_utils`` for cases where an ROI on a
single slice is described by *nested contours* — e.g. a donut-shaped
PTV that wraps around an OAR, or a hollow target volume with an inner
exclusion. ``rt_utils`` rasterizes each contour independently and
unions the results, which fills the hole. This module instead builds a
single :class:`matplotlib.path.Path` per slice containing every
sub-polygon and rasterizes with the **even-odd fill rule**: a pixel
inside an odd number of nested polygons becomes foreground, inside an
even number becomes background. This is the standard DICOM RT
convention for representing holes and is what clinical TPS exports
expect.

The implementation depends only on ``pydicom``, ``numpy``,
``SimpleITK``, and ``matplotlib`` — no rt_utils, no OpenCV. The
trade-off: it requires an explicit reference image so it can map
``(x_mm, y_mm)`` contour points to pixel indices.

Other libraries that handle holes correctly if you want to delegate
instead: ``dcmrtstruct2nii``, ``platipy``, ``pyplastimatch``,
``dcmqi``. We pick the matplotlib path here because it adds no new
heavy dependency.
"""

from __future__ import annotations

from pathlib import Path as _PathlibPath
from typing import List, Optional, Union

import numpy as np
import SimpleITK as sitk
from matplotlib.path import Path

__all__ = ["load_rtstruct_roi"]


def load_rtstruct_roi(
    rt_dicom_path: Union[str, _PathlibPath],
    reference_image: sitk.Image,
    roi_name: str,
    *,
    output_label: int = 1,
    case_insensitive: bool = True,
) -> sitk.Image:
    """Rasterize one ROI from an RTSTRUCT into a 3-D mask aligned to ``reference_image``.

    Args:
        rt_dicom_path: Path to the RTSTRUCT DICOM file.
        reference_image: SimpleITK image whose grid the mask is built
            on. Typically the planning CT this RTSTRUCT was contoured
            on. The mask inherits ``reference_image``'s size, spacing,
            origin, and direction.
        roi_name: ROI name as written in ``StructureSetROISequence``.
        output_label: Foreground value in the returned mask.
        case_insensitive: When True (default), the ROI name match
            ignores case and surrounding whitespace.

    Returns:
        3-D :class:`SimpleITK.Image` (``uint8``) with foreground voxels
        set to ``output_label`` and background to 0.

    Raises:
        ValueError: If the ROI is not found or contains no
            ``CLOSED_PLANAR`` contours.
    """
    import pydicom  # local import keeps qradiomics.io importable without pydicom

    rt = pydicom.dcmread(str(rt_dicom_path))

    roi_number = _find_roi_number(rt, roi_name, case_insensitive=case_insensitive)
    if roi_number is None:
        available = [str(r.ROIName) for r in getattr(rt, "StructureSetROISequence", [])]
        raise ValueError(
            f"ROI {roi_name!r} not found in {rt_dicom_path}. Available: {available}"
        )

    contour_seq = _contour_sequence_for_roi(rt, roi_number)
    if not contour_seq:
        raise ValueError(
            f"ROI {roi_name!r} (#{roi_number}) has no contour sequence."
        )

    by_slice = _group_polygons_by_slice(contour_seq)
    if not by_slice:
        raise ValueError(
            f"ROI {roi_name!r} contains no CLOSED_PLANAR contours."
        )

    size = reference_image.GetSize()  # (X, Y, Z)
    mask_arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)

    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    sample_points = np.column_stack([xx.ravel(), yy.ravel()])
    for z_mm, polygons in by_slice.items():
        z_idx = _z_index_for(reference_image, z_mm)
        if z_idx is None or z_idx < 0 or z_idx >= size[2]:
            continue
        slice_mask = _rasterize_even_odd(
            reference_image, polygons, sample_points, size[1], size[0]
        )
        if slice_mask is None:
            continue
        mask_arr[z_idx] |= slice_mask

    if int(output_label) != 1:
        mask_arr = mask_arr * int(output_label)

    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(reference_image)
    return mask


def _find_roi_number(rt, roi_name: str, *, case_insensitive: bool) -> Optional[int]:
    needle = roi_name.strip().lower() if case_insensitive else roi_name
    for entry in getattr(rt, "StructureSetROISequence", []):
        name = str(entry.ROIName)
        token = name.strip().lower() if case_insensitive else name
        if token == needle:
            return int(entry.ROINumber)
    return None


def _contour_sequence_for_roi(rt, roi_number: int):
    for entry in getattr(rt, "ROIContourSequence", []):
        if int(entry.ReferencedROINumber) == int(roi_number):
            return getattr(entry, "ContourSequence", [])
    return []


def _group_polygons_by_slice(contour_seq) -> "dict[float, List[np.ndarray]]":
    out: "dict[float, List[np.ndarray]]" = {}
    for contour in contour_seq:
        if str(getattr(contour, "ContourGeometricType", "")).upper() != "CLOSED_PLANAR":
            continue
        data = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
        if data.size == 0:
            continue
        # CLOSED_PLANAR contours are planar — all vertices share a z.
        z_mm = round(float(data[0, 2]), 5)
        out.setdefault(z_mm, []).append(data)
    return out


def _z_index_for(reference_image: sitk.Image, z_mm: float) -> Optional[int]:
    """Convert a physical z (mm) to the nearest reference image z-index."""
    # Use SimpleITK's transform so it stays correct for non-identity
    # direction matrices (oblique CT, head-first vs feet-first, …).
    origin = np.asarray(reference_image.GetOrigin())
    # Probe the centre of the image plane in x/y; we only need the z
    # index regardless of in-plane position.
    point = (float(origin[0]), float(origin[1]), float(z_mm))
    try:
        idx = reference_image.TransformPhysicalPointToContinuousIndex(point)
    except RuntimeError:
        return None
    return int(round(idx[2]))


def _rasterize_even_odd(
    reference_image: sitk.Image,
    polygons: List[np.ndarray],
    sample_points: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> Optional[np.ndarray]:
    """Per-polygon XOR rasterization — implements the even-odd fill rule.

    Matplotlib's ``Path.contains_points`` uses the *nonzero winding*
    rule by default. We bypass that by testing each polygon
    independently and XOR-ing the boolean inclusion masks: a pixel ends
    up foreground iff it lies inside an *odd* number of subpolygons,
    which is exactly the even-odd rule. This handles donut-shaped ROIs
    correctly regardless of contour winding direction.
    """
    accumulated: Optional[np.ndarray] = None
    for poly in polygons:
        if len(poly) < 3:
            continue
        xy_idx: List[List[float]] = []
        for pt_mm in poly:
            ix = reference_image.TransformPhysicalPointToContinuousIndex(
                (float(pt_mm[0]), float(pt_mm[1]), float(pt_mm[2]))
            )
            xy_idx.append([float(ix[0]), float(ix[1])])
        sub_path = Path(np.asarray(xy_idx, dtype=float))
        inside = sub_path.contains_points(sample_points).reshape(n_rows, n_cols)
        if accumulated is None:
            accumulated = inside.copy()
        else:
            accumulated ^= inside
    if accumulated is None:
        return None
    return accumulated.astype(np.uint8)
