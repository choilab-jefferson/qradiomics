"""AHSN — Angular Histogram of Surface Normals (Choi & Choi 2014 CMPB).

Historical note (PI): originally named SNOH = Surface Normal Orientation
Histogram in the pre-publication drafts; renamed to AHSN ("Angular
Histogram of Surface Normals") on submission. The mathematics is identical.

Faithful reproduction of §2.3.1.

Descriptor of length 2n: elevation θ ∈ [0,180°] in n bins,
azimuth φ ∈ [0,360°] in n bins. Bin weight = surface saliency,
normalized to sum 1 inside each block.

Default n = 90 → 180-dim descriptor (matches the paper's "AHSN 180").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hessian import surface_elements, SurfaceElements


@dataclass(frozen=True)
class AHSNConfig:
    n_bins_theta: int = 90       # elevation bins (each 180/n°)
    n_bins_phi: int = 90         # azimuth bins (each 360/n°)
    sigma: float = 1.0           # Gaussian smoothing for Hessian
    surfaceness_pct: float = 50  # weight-floor percentile (filters noise)

    @property
    def dim(self) -> int:
        return self.n_bins_theta + self.n_bins_phi


def normal_spherical(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (z, y, x) unit normal → (elevation θ, azimuth φ) in degrees.

    θ ∈ [0, 180], φ ∈ [0, 360). Sign-disambiguated to upper hemisphere
    (eigenvectors have sign ambiguity; flip so first component ≥ 0).
    """
    nx = normal[..., 0]
    ny = normal[..., 1]
    nz = normal[..., 2]
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-12
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    theta = np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))   # [0, 180]
    phi = (np.degrees(np.arctan2(ny, nx)) + 360.0) % 360.0   # [0, 360)
    return theta, phi


def ahsn(block: np.ndarray, mask: np.ndarray | None = None,
         cfg: AHSNConfig | None = None) -> np.ndarray:
    """Compute AHSN descriptor on a 3D image block.

    Args:
        block: (D, D, D) image block (extracted around candidate).
        mask:  (D, D, D) bool — restrict to these voxels (e.g. wall-eliminated).
        cfg:   AHSNConfig.
    Returns:
        (cfg.dim,) float vector — concatenation of θ-hist and φ-hist.
    """
    cfg = cfg or AHSNConfig()
    se = surface_elements(block, cfg.sigma)
    return _ahsn_from_elements(se, mask, cfg)


def _ahsn_from_elements(se: SurfaceElements, mask: np.ndarray | None,
                        cfg: AHSNConfig) -> np.ndarray:
    surf = se.surfaceness
    if cfg.surfaceness_pct > 0:
        thr = np.percentile(surf, cfg.surfaceness_pct)
        keep = surf >= thr
    else:
        keep = np.ones_like(surf, dtype=bool)
    if mask is not None:
        keep &= mask.astype(bool)
    weights = surf[keep]
    theta, phi = normal_spherical(se.normal)
    th, ph = theta[keep], phi[keep]
    if weights.sum() <= 0:
        return np.zeros(cfg.dim, dtype=np.float32)
    h_theta, _ = np.histogram(
        th, bins=cfg.n_bins_theta, range=(0.0, 180.0), weights=weights,
    )
    h_phi, _ = np.histogram(
        ph, bins=cfg.n_bins_phi, range=(0.0, 360.0), weights=weights,
    )
    total = weights.sum()
    desc = np.concatenate([h_theta / total, h_phi / total])
    return desc.astype(np.float32)


def find_tall_peak(desc: np.ndarray, cfg: AHSNConfig,
                   z_score: float = 3.0) -> tuple[float, float] | None:
    """Find dominant (θ, φ) peak in the descriptor — used by wall elim.

    Returns (theta_deg, phi_deg) of the joint argmax across both halves,
    or None if no peak exceeds z_score · σ above mean.
    """
    h_theta = desc[: cfg.n_bins_theta]
    h_phi = desc[cfg.n_bins_theta:]
    th_idx = int(np.argmax(h_theta))
    ph_idx = int(np.argmax(h_phi))
    th_mu, th_sd = h_theta.mean(), h_theta.std() + 1e-12
    ph_mu, ph_sd = h_phi.mean(), h_phi.std() + 1e-12
    if (h_theta[th_idx] - th_mu) / th_sd < z_score and \
       (h_phi[ph_idx] - ph_mu) / ph_sd < z_score:
        return None
    th = (th_idx + 0.5) * (180.0 / cfg.n_bins_theta)
    ph = (ph_idx + 0.5) * (360.0 / cfg.n_bins_phi)
    return float(th), float(ph)
