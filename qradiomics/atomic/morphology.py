"""Morphological region-of-interest primitives for the atomic layer.

Provides physical-mm peritumoral ring shells (:func:`peritumoral_rim`) and
morphological boundary bands (:func:`boundary_band`) shared across
``JeffLungRadiomics`` and ``penumbra``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

__all__ = ["boundary_band", "peritumoral_rim"]


def peritumoral_rim(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    rim_mm: float = 5.0,
    spacing_mm: Optional[Sequence[float]] = None,
    hu_floor: Optional[float] = -400.0,
    hu_ceil: Optional[float] = None,
) -> np.ndarray:
    """Compute a 3D peritumoral shell around ``mask > 0`` of thickness ``rim_mm``.

    Args:
        mask: 3D binary/integer mask array ``(Z, Y, X)``.
        image: Optional 3D image array aligned with ``mask`` for HU gating.
        rim_mm: Physical shell thickness in millimeters (must be > 0).
        spacing_mm: Voxel spacing along array axes ``(sz, sy, sx)`` in mm.
            Defaults to isotropic ``(1.0, 1.0, 1.0)`` when ``None``.
        hu_floor: Optional lower intensity threshold (e.g. ``-400.0`` HU to
            restrict the peritumoral ring to lung parenchyma / soft tissue and
            exclude air). Pass ``None`` to disable.
        hu_ceil: Optional upper intensity threshold (e.g. to exclude bone).

    Returns:
        ``uint8`` array of identical shape with ``1`` inside the peritumoral
        ring and ``0`` elsewhere (core foreground voxels are always ``0``).
    """
    if rim_mm <= 0:
        raise ValueError(f"rim_mm must be > 0, got {rim_mm}")
    fg = np.asarray(mask) > 0
    if fg.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {fg.shape}")
    if not fg.any():
        return np.zeros_like(fg, dtype=np.uint8)

    sampling = (1.0, 1.0, 1.0) if spacing_mm is None else tuple(float(s) for s in spacing_mm)
    if len(sampling) != 3 or any(s <= 0 for s in sampling):
        raise ValueError(f"spacing_mm must have 3 positive floats, got {spacing_mm}")

    dist_out = distance_transform_edt(~fg, sampling=sampling)
    rim = (~fg) & (dist_out <= float(rim_mm))

    if image is not None:
        img = np.asarray(image)
        if img.shape != fg.shape:
            raise ValueError(f"image/mask shape mismatch: image={img.shape} mask={fg.shape}")
        if hu_floor is not None:
            rim &= img > float(hu_floor)
        if hu_ceil is not None:
            rim &= img <= float(hu_ceil)

    return rim.astype(np.uint8)


def boundary_band(
    mask: np.ndarray,
    contour_type: str = "inner_outer",
    width_mm: float = 2.0,
    spacing_mm: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Compute a morphological boundary band around a 3D mask.

    Args:
        mask: 3D binary/integer mask array ``(Z, Y, X)``.
        contour_type: One of ``"full"`` (or ``"ptv"``), ``"inner"``,
            ``"outer"``, ``"inner_outer"`` (or ``"gtv"`` / ``"itv"``).
        width_mm: Band half-width in millimeters (must be > 0).
        spacing_mm: Voxel spacing ``(sz, sy, sx)`` in mm; defaults to 1 mm.

    Returns:
        ``uint8`` binary mask of the requested band.
    """
    if width_mm <= 0:
        raise ValueError(f"width_mm must be > 0, got {width_mm}")
    fg = np.asarray(mask) > 0
    if fg.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {fg.shape}")
    if not fg.any():
        return np.zeros_like(fg, dtype=np.uint8)

    ct = contour_type.lower().strip()
    if ct in ("full", "ptv"):
        return fg.astype(np.uint8)

    sampling = (1.0, 1.0, 1.0) if spacing_mm is None else tuple(float(s) for s in spacing_mm)
    if len(sampling) != 3 or any(s <= 0 for s in sampling):
        raise ValueError(f"spacing_mm must have 3 positive floats, got {spacing_mm}")

    if ct == "outer":
        dist_out = distance_transform_edt(~fg, sampling=sampling)
        band = (~fg) & (dist_out <= float(width_mm))
    elif ct == "inner":
        dist_in = distance_transform_edt(fg, sampling=sampling)
        band = fg & (dist_in <= float(width_mm))
    elif ct in ("inner_outer", "gtv", "itv"):
        dist_out = distance_transform_edt(~fg, sampling=sampling)
        dist_in = distance_transform_edt(fg, sampling=sampling)
        band = ((~fg) & (dist_out <= float(width_mm))) | (fg & (dist_in <= float(width_mm)))
    else:
        raise ValueError(
            f"Unknown contour_type {contour_type!r}; expected one of "
            "('full', 'inner', 'outer', 'inner_outer', 'gtv', 'itv', 'ptv')"
        )
    return band.astype(np.uint8)
