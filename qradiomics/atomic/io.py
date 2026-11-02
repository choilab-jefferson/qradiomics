"""Image + mask loading primitives for the atomic layer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk

__all__ = ["check_geometry", "load_image_and_mask", "save_image_and_mask"]


def load_image_and_mask(
    image_path: str | Path,
    mask_path: str | Path,
    *,
    require_compatible_geometry: bool = True,
    spacing_tolerance: float = 1e-3,
    origin_tolerance: float = 1e-3,
) -> Tuple[sitk.Image, sitk.Image]:
    """Load an image and a mask from disk and validate their geometry.

    Args:
        image_path: NRRD/NIfTI/MHA path for the image volume.
        mask_path: NRRD/NIfTI/MHA path for the binary or label mask.
        require_compatible_geometry: If True (default), raise when image and
            mask differ in size/spacing/origin beyond the tolerances. Set to
            False when callers want to resample themselves.
        spacing_tolerance: Per-axis absolute tolerance for spacing equality.
        origin_tolerance: Per-axis absolute tolerance for origin equality.

    Returns:
        ``(image, mask)`` as SimpleITK Image objects.
    """
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if require_compatible_geometry:
        check_geometry(
            image,
            mask,
            spacing_tolerance=spacing_tolerance,
            origin_tolerance=origin_tolerance,
        )
    return image, mask


def check_geometry(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    spacing_tolerance: float = 1e-3,
    origin_tolerance: float = 1e-3,
) -> None:
    """Raise ``ValueError`` if image and mask are not geometrically aligned.

    Compares size, spacing, and origin. Direction matrix mismatches are
    common across DICOM converters and are left to the caller to resolve;
    PyRadiomics handles small direction differences via ``geometryTolerance``.
    """
    if image.GetSize() != mask.GetSize():
        raise ValueError(f"Image/mask size mismatch: image={image.GetSize()} mask={mask.GetSize()}")
    img_spacing = image.GetSpacing()
    msk_spacing = mask.GetSpacing()
    if any(abs(a - b) > spacing_tolerance for a, b in zip(img_spacing, msk_spacing)):
        raise ValueError(
            f"Image/mask spacing mismatch beyond {spacing_tolerance}: "
            f"image={img_spacing} mask={msk_spacing}"
        )
    img_origin = image.GetOrigin()
    msk_origin = mask.GetOrigin()
    if any(abs(a - b) > origin_tolerance for a, b in zip(img_origin, msk_origin)):
        raise ValueError(
            f"Image/mask origin mismatch beyond {origin_tolerance}: "
            f"image={img_origin} mask={msk_origin}"
        )


def save_image_and_mask(
    image: Union[sitk.Image, np.ndarray],
    mask: Union[sitk.Image, np.ndarray],
    image_path: str | Path,
    mask_path: str | Path,
    *,
    spacing: Optional[Sequence[float]] = None,
    origin: Optional[Sequence[float]] = None,
    direction: Optional[Sequence[float]] = None,
    compress: bool = True,
) -> Tuple[Path, Path]:
    """Write an ``(image, mask)`` pair to disk with consistent geometry."""
    img_sitk = _coerce_sitk(image, spacing=spacing, origin=origin, direction=direction)
    msk_sitk = _coerce_sitk(mask, spacing=spacing, origin=origin, direction=direction, is_mask=True)
    if spacing is None and isinstance(image, sitk.Image) and not isinstance(mask, sitk.Image):
        msk_sitk.CopyInformation(img_sitk)
    check_geometry(img_sitk, msk_sitk)
    ip, mp = Path(image_path), Path(mask_path)
    sitk.WriteImage(img_sitk, str(ip), bool(compress))
    sitk.WriteImage(msk_sitk, str(mp), bool(compress))
    return ip, mp


def _coerce_sitk(
    vol: Union[sitk.Image, np.ndarray],
    *,
    spacing: Optional[Sequence[float]] = None,
    origin: Optional[Sequence[float]] = None,
    direction: Optional[Sequence[float]] = None,
    is_mask: bool = False,
) -> sitk.Image:
    if isinstance(vol, sitk.Image):
        out = sitk.Image(vol)
    else:
        arr = np.asarray(vol)
        if is_mask:
            arr = arr.astype(np.uint8, copy=False)
        out = sitk.GetImageFromArray(arr)
    if spacing is not None:
        out.SetSpacing(tuple(float(s) for s in spacing))
    if origin is not None:
        out.SetOrigin(tuple(float(o) for o in origin))
    if direction is not None:
        out.SetDirection(tuple(float(d) for d in direction))
    return out
