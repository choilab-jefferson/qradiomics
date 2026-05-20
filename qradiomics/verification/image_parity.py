"""Image-pair parity helpers for preprocess_pair / load_image_and_mask
verification (ADR-005 Stage 3b).

When two implementations return SimpleITK images (e.g. cropped+resampled
output of a `preprocess_pair` swap), `compare_image_pair` reports their
geometric and pixel-wise differences in a structure that the `run_ab_sweep`
harness can summarise across a manifest.

Returned `ImageDiff` is empty (all-zero) when the images are
byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class ImageDiff:
    """Summary of how two SimpleITK images differ.

    Geometry fields are exact comparisons (mismatch = True or non-zero).
    Pixel-array fields are computed only when geometry matches.
    """
    size_diff: bool = False
    spacing_max_abs: float = 0.0
    origin_max_abs: float = 0.0
    direction_max_abs: float = 0.0
    pixel_max_abs: float = 0.0
    pixel_mean_abs: float = 0.0
    pixel_n_above_tol: int = 0          # count of voxels exceeding pixel_tol
    pixel_total: int = 0                # total voxel count compared
    note: str = ""

    @property
    def is_identical(self) -> bool:
        return (
            not self.size_diff
            and self.spacing_max_abs == 0.0
            and self.origin_max_abs == 0.0
            and self.direction_max_abs == 0.0
            and self.pixel_max_abs == 0.0
        )


def compare_image_pair(
    a, b,
    geom_tol: float = 0.0,
    pixel_tol: float = 1e-9,
) -> ImageDiff:
    """Compare two SimpleITK images and return an `ImageDiff` summary.

    Args:
        a, b: SimpleITK images.
        geom_tol: tolerance for spacing/origin/direction element diffs
            (0.0 = exact). Size mismatch is always strict (cannot tolerate).
        pixel_tol: absolute tolerance per voxel.

    Returns:
        ImageDiff with diff statistics. `is_identical` is True iff every
        check is within tolerance.
    """
    import numpy as np

    diff = ImageDiff()
    if tuple(a.GetSize()) != tuple(b.GetSize()):
        diff.size_diff = True
        diff.note = f"size {a.GetSize()} vs {b.GetSize()}"
        return diff

    sa = np.asarray(a.GetSpacing()); sb = np.asarray(b.GetSpacing())
    diff.spacing_max_abs = float(np.max(np.abs(sa - sb)))
    if diff.spacing_max_abs > geom_tol:
        diff.note = f"spacing diff: {sa} vs {sb}"

    oa = np.asarray(a.GetOrigin()); ob = np.asarray(b.GetOrigin())
    diff.origin_max_abs = float(np.max(np.abs(oa - ob)))

    da = np.asarray(a.GetDirection()); db = np.asarray(b.GetDirection())
    diff.direction_max_abs = float(np.max(np.abs(da - db)))

    # Voxel-wise comparison (cast to float so int mask diffs are still numeric)
    import SimpleITK as sitk
    arr_a = sitk.GetArrayFromImage(a).astype(np.float64)
    arr_b = sitk.GetArrayFromImage(b).astype(np.float64)
    delta = np.abs(arr_a - arr_b)
    diff.pixel_total = int(delta.size)
    diff.pixel_max_abs = float(delta.max()) if delta.size else 0.0
    diff.pixel_mean_abs = float(delta.mean()) if delta.size else 0.0
    diff.pixel_n_above_tol = int((delta > pixel_tol).sum())
    return diff


def compare_image_pair_dict(
    legacy: Tuple[Any, Any],
    candidate: Tuple[Any, Any],
    geom_tol: float = 0.0,
    pixel_tol: float = 1e-9,
) -> Dict[str, ImageDiff]:
    """Apply compare_image_pair to (image, mask) tuples.

    Returns {"image": ImageDiff, "mask": ImageDiff}.
    """
    return {
        "image": compare_image_pair(legacy[0], candidate[0],
                                     geom_tol=geom_tol, pixel_tol=pixel_tol),
        "mask":  compare_image_pair(legacy[1], candidate[1],
                                     geom_tol=geom_tol, pixel_tol=pixel_tol),
    }


