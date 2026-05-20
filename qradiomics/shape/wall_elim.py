"""Iterative wall detection & elimination (Algorithm 1 of paper §2.3.2).

Walls = large connected surface patches whose normals point in a similar
direction (tall peak in the AHSN). Iteratively detected and masked out.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from .ahsn import AHSNConfig, _ahsn_from_elements, find_tall_peak, normal_spherical
from .hessian import surface_elements


def _orientation_similar(theta: np.ndarray, phi: np.ndarray,
                          th0: float, ph0: float,
                          theta_tol_deg: float = 22.5,
                          phi_tol_deg: float = 22.5) -> np.ndarray:
    """Boolean mask: voxels whose normal orientation is within ±tol of (th0,ph0)."""
    dth = np.abs(theta - th0)
    dph = np.abs(phi - ph0)
    dph = np.minimum(dph, 360.0 - dph)
    return (dth <= theta_tol_deg) & (dph <= phi_tol_deg)


def wall_eliminate(block: np.ndarray, cfg: AHSNConfig | None = None,
                    area_threshold: int | None = None,
                    z_score: float = 3.0,
                    max_iter: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Refine AHSN descriptor by removing wall-like surfaces.

    Returns:
        descriptor (after WE), keep_mask (bool, same shape as block).
    """
    cfg = cfg or AHSNConfig()
    se = surface_elements(block, cfg.sigma)
    keep = np.ones(block.shape, dtype=bool)
    if area_threshold is None:
        # heuristic: 10% of block voxels
        area_threshold = max(8, int(0.1 * block.size))

    for _ in range(max_iter):
        desc = _ahsn_from_elements(se, keep, cfg)
        peak = find_tall_peak(desc, cfg, z_score=z_score)
        if peak is None:
            break
        th0, ph0 = peak
        theta, phi = normal_spherical(se.normal)
        sim = _orientation_similar(theta, phi, th0, ph0) & keep
        if sim.sum() == 0:
            break
        labels, n = label(sim)
        removed_any = False
        for k in range(1, n + 1):
            comp = labels == k
            if comp.sum() >= area_threshold:
                keep &= ~comp
                removed_any = True
        if not removed_any:
            break
    desc = _ahsn_from_elements(se, keep, cfg)
    return desc, keep
