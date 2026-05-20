"""Multi-scale dot enhancement filter (§2.2.2 of Choi & Choi 2014).

Detects sphere-like (nodule) blobs. Uses Li-Sato dot enhancement:
    dot(λ) = |λ3| * (λ3/λ1)²   if  λ1,λ2,λ3 < 0  (bright blob)
           = 0                  otherwise.

Eigenvalues here use the paper's convention |λ1| ≥ |λ2| ≥ |λ3|; for
the dot filter we use the signed values directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

from .hessian import hessian_3d, hessian_eigendecomp


@dataclass(frozen=True)
class Candidate:
    z: int
    y: int
    x: int
    scale: float
    diameter: float  # mm or voxel
    dot: float


def dot_value(evals_signed: np.ndarray) -> np.ndarray:
    """Per-voxel Sato/Li dot enhancement.

    Args:
        evals_signed: (..., 3) sorted by descending |λ|.
    """
    l1 = evals_signed[..., 0]
    l3 = evals_signed[..., 2]
    bright = (evals_signed < 0).all(axis=-1)
    safe_l1 = np.where(np.abs(l1) > 1e-8, l1, 1e-8)
    score = np.abs(l3) * (l3 / safe_l1) ** 2
    return np.where(bright, score, 0.0).astype(np.float32)


def make_scales(d_min: float, d_max: float, n: int = 5) -> list[float]:
    """Eq. 7 — log-spaced σ values in range [d_min/4, d_max/4]."""
    s0, s1 = d_min / 4.0, d_max / 4.0
    r = (s1 / s0) ** (1.0 / (n - 1))
    return [s0 * r ** k for k in range(n)]


def detect_candidates(volume: np.ndarray, lung_mask: np.ndarray | None = None,
                       d_min: float = 3.0, d_max: float = 30.0,
                       n_scales: int = 5,
                       nms_radius: int = 3,
                       threshold: float | None = None) -> list[Candidate]:
    """Multi-scale dot enhancement + per-scale local-max NMS.

    Args:
        volume: (Z, Y, X) CT (HU). Will be smoothed at each scale.
        lung_mask: (Z, Y, X) bool — only detect inside.
        d_min, d_max: nodule diameter range in voxels (or mm if isotropic 1mm).
        n_scales: number of Gaussian scales.
        nms_radius: voxel radius for per-scale non-max suppression.
        threshold: per-scale dot threshold; if None, uses mean of local maxima.
    """
    scales = make_scales(d_min, d_max, n_scales)
    out: list[Candidate] = []
    for sigma in scales:
        H = hessian_3d(volume, sigma)
        evals, _ = hessian_eigendecomp(H)
        d = dot_value(evals)
        if lung_mask is not None:
            d = d * lung_mask
        # Per-scale NMS via maximum_filter
        loc = (d == maximum_filter(d, size=2 * nms_radius + 1)) & (d > 0)
        if threshold is None:
            vals = d[loc]
            thr = float(vals.mean()) if vals.size > 0 else 0.0
        else:
            thr = float(threshold)
        hits = loc & (d >= thr)
        zz, yy, xx = np.where(hits)
        diameter = 4.0 * sigma
        for z, y, x in zip(zz, yy, xx):
            out.append(Candidate(int(z), int(y), int(x),
                                 float(sigma), float(diameter), float(d[z, y, x])))
    return out


def extract_block(volume: np.ndarray, c: Candidate,
                  boundary: int = 2) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Extract centered cubic block of size K = ceil(4σ) + 2b (Eq. 9).

    Returns (block, top_left_corner).
    """
    K = int(np.ceil(4 * c.scale)) + 2 * boundary
    if K % 2 == 0:
        K += 1
    half = K // 2
    z0, y0, x0 = c.z - half, c.y - half, c.x - half
    z1, y1, x1 = z0 + K, y0 + K, x0 + K
    Z, Y, X = volume.shape
    # pad if out of bounds
    pad = [(max(0, -z0), max(0, z1 - Z)),
           (max(0, -y0), max(0, y1 - Y)),
           (max(0, -x0), max(0, x1 - X))]
    v_pad = np.pad(volume, pad, mode="edge")
    z0p, y0p, x0p = z0 + pad[0][0], y0 + pad[1][0], x0 + pad[2][0]
    blk = v_pad[z0p:z0p + K, y0p:y0p + K, x0p:x0p + K]
    return blk.astype(np.float32, copy=False), (z0, y0, x0)
