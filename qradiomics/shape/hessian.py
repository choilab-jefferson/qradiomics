"""Hessian eigendecomposition and surface elements.

Implements §2.2.1 of Choi & Choi 2014 CMPB.

Convention: eigenvalues sorted so |λ1| ≥ |λ2| ≥ |λ3|. Then:
  surfaceness = (|λ1| − |λ2|),  normal = e1
  curvedness  = (|λ2| − |λ3|),  tangent = e3
  pointedness =  |λ3|
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class SurfaceElements:
    """Per-voxel surface descriptor from Hessian eigendecomposition."""

    surfaceness: np.ndarray   # (Z,Y,X) ≥ 0
    curvedness: np.ndarray    # (Z,Y,X) ≥ 0
    pointedness: np.ndarray   # (Z,Y,X) ≥ 0
    normal: np.ndarray        # (Z,Y,X,3) — e1 (largest |λ|)
    tangent: np.ndarray       # (Z,Y,X,3) — e3 (smallest |λ|)
    eigvals: np.ndarray       # (Z,Y,X,3) sorted by descending |λ|

    @property
    def shape(self) -> tuple[int, ...]:
        return self.surfaceness.shape


def hessian_3d(volume: np.ndarray, sigma: float) -> np.ndarray:
    """Compute scale-normalized 3D Hessian via Gaussian derivatives.

    Returns array of shape (Z, Y, X, 3, 3). Scale normalization (×σ²)
    makes responses comparable across scales (Lindeberg 1998).
    """
    v = volume.astype(np.float32, copy=False)
    s = float(sigma)
    # Second-order Gaussian derivatives via gaussian_filter with `order`
    H = np.empty((*v.shape, 3, 3), dtype=np.float32)
    axes = (0, 1, 2)  # Z, Y, X
    for i, ai in enumerate(axes):
        for j, aj in enumerate(axes):
            if j < i:
                H[..., i, j] = H[..., j, i]
                continue
            order = [0, 0, 0]
            order[ai] += 1
            order[aj] += 1
            H[..., i, j] = gaussian_filter(v, sigma=s, order=tuple(order),
                                           mode="nearest")
    H *= s * s  # scale normalization
    return H


def hessian_eigendecomp(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition per voxel, sorted by descending |eigenvalue|.

    Args:
        H: (Z, Y, X, 3, 3) symmetric Hessian.
    Returns:
        evals: (Z, Y, X, 3) sorted |λ1| ≥ |λ2| ≥ |λ3| (signed values kept)
        evecs: (Z, Y, X, 3, 3) corresponding eigenvectors as columns
    """
    # np.linalg.eigh returns ascending eigenvalues
    evals_asc, evecs_asc = np.linalg.eigh(H)
    # Sort by |λ| descending
    order = np.argsort(-np.abs(evals_asc), axis=-1)
    take = np.take_along_axis
    evals = take(evals_asc, order, axis=-1)
    # Reorder eigenvectors: evecs[..., :, i] = i-th eigenvector (columns)
    evecs = take(evecs_asc, order[..., None, :], axis=-1)
    return evals.astype(np.float32), evecs.astype(np.float32)


def surface_elements(volume: np.ndarray, sigma: float) -> SurfaceElements:
    """Compute surface saliency, normal, tangent at every voxel.

    Eq. 6 of paper:
        H = (λ1 − λ2) e1·e1ᵀ + (λ2 − λ3)(e1·e1ᵀ + e2·e2ᵀ) + λ3 · I

    With |λ1| ≥ |λ2| ≥ |λ3|:
        surfaceness = |λ1| − |λ2|;  e1 is surface normal
        curvedness  = |λ2| − |λ3|;  e3 is curve tangent
        pointedness = |λ3|
    """
    H = hessian_3d(volume, sigma)
    evals, evecs = hessian_eigendecomp(H)
    absL = np.abs(evals)
    surf = np.maximum(absL[..., 0] - absL[..., 1], 0.0)
    curv = np.maximum(absL[..., 1] - absL[..., 2], 0.0)
    pt = absL[..., 2]
    n = evecs[..., :, 0]  # normal = column 0
    t = evecs[..., :, 2]  # tangent = column 2
    return SurfaceElements(
        surfaceness=surf, curvedness=curv, pointedness=pt,
        normal=n, tangent=t, eigvals=evals,
    )
