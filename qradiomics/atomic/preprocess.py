"""Bounding-box crop + optional isotropic resample for an (image, mask) pair.

The radiomics workflow needs two related preprocessing primitives:

* **Bounding-box crop** — restrict the image and mask to the smallest
  axis-aligned region that encloses the foreground voxels of the mask,
  optionally with a padding margin. Reduces downstream filter cost and
  keeps PyRadiomics' own padding semantics predictable.
* **Isotropic resample** — bring the cropped pair onto a uniform voxel
  spacing (e.g. 1×1×1 mm) so texture features compare across scanners.

These are paired here because most workflows do them together (crop then
resample). Both are pure SimpleITK operations on in-memory ``Image``
handles, so the function is friendly to wrappers that already have
images loaded.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import SimpleITK as sitk

__all__ = ["preprocess_pair"]


def preprocess_pair(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    pad_mm: float = 0.0,
    resample_mm: Optional[Sequence[float]] = None,
    image_interpolator: int = sitk.sitkLinear,
    mask_interpolator: int = sitk.sitkNearestNeighbor,
    label: int = 1,
) -> Tuple[sitk.Image, sitk.Image]:
    """Crop ``(image, mask)`` to the mask's bounding box and optionally resample.

    Args:
        image: Input image (any modality).
        mask: Binary or label mask geometrically aligned with ``image``.
            Caller is responsible for upstream alignment — see
            :func:`qradiomics.atomic.check_geometry`.
        pad_mm: Symmetric padding (in millimeters) to add around the mask
            bounding box on each axis. Values are converted to voxels using
            ``image.GetSpacing()`` and clamped to the image extent.
        resample_mm: If given, resample the cropped image and mask to this
            isotropic spacing (sequence of length 3, in mm). Pass
            ``(1.0, 1.0, 1.0)`` for the conventional 1 mm isotropic
            target.
        image_interpolator: SimpleITK interpolation enum for the image.
        mask_interpolator: SimpleITK interpolation enum for the mask
            (default: nearest neighbor — preserves label integers).
        label: Label value whose bounding box defines the crop.

    Returns:
        ``(cropped_image, cropped_mask)``. Spacing/direction/origin are
        consistent across the pair.
    """
    if image.GetSize() != mask.GetSize():
        raise ValueError(
            f"Image/mask size mismatch: image={image.GetSize()} mask={mask.GetSize()}"
        )

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    if stats.GetNumberOfLabels() == 0:
        raise ValueError("Mask contains no foreground voxels — cannot compute bounding box.")
    if label not in stats.GetLabels():
        raise ValueError(
            f"Requested label {label} not present in mask labels {stats.GetLabels()}."
        )

    # bbox is (x0, y0, z0, sx, sy, sz)
    bbox = stats.GetBoundingBox(label)
    spacing = image.GetSpacing()
    pad_voxels = tuple(max(0, int(round(pad_mm / s))) for s in spacing)

    size_total = image.GetSize()
    start = [max(0, bbox[i] - pad_voxels[i]) for i in range(3)]
    end = [min(size_total[i], bbox[i] + bbox[i + 3] + pad_voxels[i]) for i in range(3)]
    size = [end[i] - start[i] for i in range(3)]
    if any(s <= 0 for s in size):
        raise ValueError(f"Computed crop has non-positive size: {size}")

    cropped_image = sitk.RegionOfInterest(image, size=size, index=start)
    cropped_mask = sitk.RegionOfInterest(mask, size=size, index=start)

    if resample_mm is not None:
        if len(resample_mm) != 3:
            raise ValueError(
                f"resample_mm must have length 3 (got {len(resample_mm)})."
            )
        cropped_image = _resample(cropped_image, resample_mm, image_interpolator)
        cropped_mask = _resample(cropped_mask, resample_mm, mask_interpolator)

    return cropped_image, cropped_mask


def _resample(
    image: sitk.Image,
    target_spacing: Sequence[float],
    interpolator: int,
) -> sitk.Image:
    src_spacing = image.GetSpacing()
    src_size = image.GetSize()
    new_size = [
        max(1, int(round(src_size[i] * src_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(float(s) for s in target_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)
