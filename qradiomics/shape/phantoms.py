"""Synthetic 3-D lung phantoms for spiculation / shape testing.

Generates realistic-ish mini CT volumes with combinations of:
  - smooth pulmonary nodule (benign-like)
  - spiculated lung cancer (~2 cm) with controllable spike count/length
  - cylindrical vessels
  - hollow cylindrical airway
  - curved lung wall (juxta-pleural setting)
  - lobulated bumps

Plus pure geometric primitives (sphere, cylinder, plane, saddle).

Each phantom returns:
    `(volume, mask, attachment_mask, meta)` where
      - `volume`     : float32 HU-like (-1000 air, -700 parenchyma,
                       -100..+30 soft tissue, ~+200 calcified vessel/airway wall)
      - `mask`       : bool nodule core
      - `attachment_mask` : bool voxels where vessel/wall touches the nodule
      - `meta`       : dict describing what's in the phantom
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =====================================================================
# Background lung parenchyma
# =====================================================================

HU_AIR = -1000.0
HU_PARENCHYMA = -700.0
HU_SOFT = -100.0
HU_TUMOR = 30.0
HU_VESSEL = 100.0
HU_WALL = 200.0


def _empty(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return (volume initialized to parenchyma, all-False mask)."""
    vol = np.full((size, size, size), HU_PARENCHYMA, dtype=np.float32)
    mask = np.zeros((size, size, size), dtype=bool)
    return vol, mask


def _grid_centered(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = (size - 1) / 2
    z, y, x = np.indices((size, size, size), dtype=np.float32)
    return z - c, y - c, x - c


# =====================================================================
# Geometric primitives
# =====================================================================

def add_sphere(vol: np.ndarray, mask: np.ndarray, center: tuple[float, float, float],
                radius: float, hu: float = HU_TUMOR) -> None:
    cz, cy, cx = center
    z, y, x = np.indices(vol.shape, dtype=np.float32)
    rho = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
    region = rho <= radius
    vol[region] = hu
    mask |= region


def add_cylinder(vol: np.ndarray, mask: np.ndarray | None,
                  start: tuple[float, float, float],
                  end: tuple[float, float, float],
                  radius: float, hu: float = HU_VESSEL,
                  hollow_inner_radius: float = 0.0,
                  hollow_hu: float = HU_AIR) -> None:
    """Cylinder between two endpoints; optional hollow core (airway)."""
    z, y, x = np.indices(vol.shape, dtype=np.float32)
    s = np.asarray(start, dtype=np.float32)
    e = np.asarray(end, dtype=np.float32)
    axis = e - s
    length = np.linalg.norm(axis)
    axis_unit = axis / (length + 1e-12)
    rel = np.stack([z - s[0], y - s[1], x - s[2]], axis=-1)
    along = rel @ axis_unit
    perp = rel - along[..., None] * axis_unit
    perp_dist = np.linalg.norm(perp, axis=-1)
    in_length = (along >= 0) & (along <= length)
    in_radius = perp_dist <= radius
    region = in_length & in_radius
    vol[region] = hu
    if hollow_inner_radius > 0:
        inner = in_length & (perp_dist <= hollow_inner_radius)
        vol[inner] = hollow_hu
    if mask is not None:
        mask |= region


def add_spike(vol: np.ndarray, mask: np.ndarray,
               nodule_center: tuple[float, float, float],
               nodule_radius: float, direction: tuple[float, float, float],
               length: float, base_radius: float = 1.5,
               tip_radius: float = 0.5,
               hu: float = HU_TUMOR) -> None:
    """A tapered spike protruding from a nodule along `direction`."""
    z, y, x = np.indices(vol.shape, dtype=np.float32)
    c = np.asarray(nodule_center, dtype=np.float32)
    d = np.asarray(direction, dtype=np.float32)
    d = d / (np.linalg.norm(d) + 1e-12)
    rel = np.stack([z - c[0], y - c[1], x - c[2]], axis=-1)
    along = rel @ d
    perp = rel - along[..., None] * d
    perp_dist = np.linalg.norm(perp, axis=-1)
    t = (along - nodule_radius + 1) / max(length, 1e-6)
    # Linear taper from base to tip
    r_t = base_radius + (tip_radius - base_radius) * np.clip(t, 0, 1)
    region = (along >= nodule_radius - 1) & (along <= nodule_radius + length) \
             & (perp_dist <= r_t)
    vol[region] = hu
    mask |= region


def add_lobulation(vol: np.ndarray, mask: np.ndarray,
                    nodule_center: tuple[float, float, float],
                    nodule_radius: float, direction: tuple[float, float, float],
                    bulge_radius: float = 3.0, hu: float = HU_TUMOR) -> None:
    """A rounded bulge (lobulation) on the nodule surface."""
    d = np.asarray(direction, dtype=np.float32)
    d = d / (np.linalg.norm(d) + 1e-12)
    offset = (nodule_radius - bulge_radius / 2) * d
    bulge_center = np.asarray(nodule_center, dtype=np.float32) + offset
    add_sphere(vol, mask, tuple(bulge_center.tolist()), bulge_radius, hu)


def add_plane(vol: np.ndarray, mask: np.ndarray | None,
               position: float, axis: int, hu: float = HU_WALL,
               thickness: float = 3.0) -> None:
    """Flat slab along given axis (0=z, 1=y, 2=x)."""
    coord = np.indices(vol.shape)[axis].astype(np.float32)
    region = np.abs(coord - position) <= thickness / 2
    vol[region] = hu
    if mask is not None:
        mask |= region


def add_curved_wall(vol: np.ndarray, mask: np.ndarray | None,
                     center: tuple[float, float, float],
                     radius: float, thickness: float = 2.5,
                     hu: float = HU_WALL) -> None:
    """Spherical shell (mimics a portion of the lung wall)."""
    z, y, x = np.indices(vol.shape, dtype=np.float32)
    cz, cy, cx = center
    rho = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
    region = np.abs(rho - radius) <= thickness / 2
    vol[region] = hu
    if mask is not None:
        mask |= region


# =====================================================================
# Named phantom presets
# =====================================================================

@dataclass
class Phantom:
    name: str
    volume: np.ndarray
    mask: np.ndarray
    attachment_mask: np.ndarray
    meta: dict = field(default_factory=dict)


def benign_nodule(size: int = 64, radius: float = 6.0) -> Phantom:
    """Smooth round nodule (~12 mm), no attachments."""
    vol, mask = _empty(size)
    c = (size / 2, size / 2, size / 2)
    add_sphere(vol, mask, c, radius)
    return Phantom("benign_nodule", vol, mask, np.zeros_like(mask),
                   meta={"radius_vox": radius, "center": c})


def lung_cancer_spiculated(size: int = 64, radius: float = 10.0,
                             n_spikes: int = 6, spike_len: float = 5.0
                             ) -> Phantom:
    """~2 cm spiculated cancer (radius ≈ 10 vox → 20 mm at 1 mm/vox)."""
    vol, mask = _empty(size)
    c = (size / 2, size / 2, size / 2)
    add_sphere(vol, mask, c, radius)
    # Distribute spikes on roughly orthogonal directions
    base_dirs = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        (1, 0, 1), (-1, 0, 1),
    ][:n_spikes]
    rng = np.random.default_rng(0)
    for d in base_dirs:
        jitter = rng.normal(0, 0.05, size=3)
        dir_j = np.array(d, dtype=np.float32) + jitter
        add_spike(vol, mask, c, radius, tuple(dir_j.tolist()),
                  length=spike_len + rng.uniform(-1, 1),
                  base_radius=1.5, tip_radius=0.4)
    return Phantom("lung_cancer_spiculated", vol, mask,
                   np.zeros_like(mask),
                   meta={"radius_vox": radius, "n_spikes": n_spikes,
                         "spike_len": spike_len})


def lobulated_nodule(size: int = 64, radius: float = 7.0,
                      n_lobes: int = 3) -> Phantom:
    """Nodule with lobulated bumps (no spiculation)."""
    vol, mask = _empty(size)
    c = (size / 2, size / 2, size / 2)
    add_sphere(vol, mask, c, radius)
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (-1, 1, 0), (1, -1, 1)][:n_lobes]
    for d in directions:
        add_lobulation(vol, mask, c, radius, d, bulge_radius=3.0)
    return Phantom("lobulated_nodule", vol, mask, np.zeros_like(mask),
                   meta={"radius_vox": radius, "n_lobes": n_lobes})


def juxta_vascular_cancer(size: int = 64, radius: float = 10.0,
                            n_spikes: int = 4, vessel_radius: float = 2.0
                            ) -> Phantom:
    """Spiculated cancer touching a vessel — tests attachment detection."""
    vol, mask = _empty(size)
    c = (size / 2, size / 2, size / 2)
    add_sphere(vol, mask, c, radius)
    # Vessel runs along x-axis through the nodule edge
    v_start = (c[0], c[1] - radius, 0)
    v_end = (c[0], c[1] - radius, size - 1)
    attachment = np.zeros_like(mask)
    add_cylinder(vol, attachment, v_start, v_end, vessel_radius, hu=HU_VESSEL)
    # Attachment = vessel ∩ dilated-nodule
    from scipy.ndimage import binary_dilation
    near_nodule = binary_dilation(mask, iterations=2)
    attachment = attachment & near_nodule
    # Add spikes
    for d in [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)][:n_spikes]:
        add_spike(vol, mask, c, radius, d, length=5)
    return Phantom("juxta_vascular_cancer", vol, mask, attachment,
                   meta={"radius_vox": radius, "vessel_radius": vessel_radius,
                         "n_spikes": n_spikes})


def juxta_pleural_cancer(size: int = 64, radius: float = 10.0,
                           n_spikes: int = 4) -> Phantom:
    """Spiculated cancer touching a curved lung wall."""
    vol, mask = _empty(size)
    c = (size / 2, size / 2, size / 2)
    add_sphere(vol, mask, c, radius)
    # Curved wall on +y side of nodule
    wall_center = (c[0], c[1] + 30, c[2])
    attachment = np.zeros_like(mask)
    add_curved_wall(vol, attachment, wall_center, radius=30,
                     thickness=2.5, hu=HU_WALL)
    from scipy.ndimage import binary_dilation
    attachment = attachment & binary_dilation(mask, iterations=2)
    for d in [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)][:n_spikes]:
        add_spike(vol, mask, c, radius, d, length=5)
    return Phantom("juxta_pleural_cancer", vol, mask, attachment,
                   meta={"radius_vox": radius, "n_spikes": n_spikes})


def airway_only(size: int = 64, radius: float = 4.0,
                 wall: float = 1.0) -> Phantom:
    """Hollow cylindrical airway (no nodule) — should not be classified as one."""
    vol, mask = _empty(size)
    c = size / 2
    add_cylinder(vol, None, (c, c, 0), (c, c, size - 1), radius,
                  hu=HU_WALL, hollow_inner_radius=radius - wall,
                  hollow_hu=HU_AIR)
    # Mask = entire airway wall ring (no nodule)
    return Phantom("airway_only", vol, mask, np.zeros_like(mask),
                   meta={"airway_radius": radius})


def vessel_only(size: int = 64, radius: float = 2.0) -> Phantom:
    """Solid cylindrical vessel."""
    vol, mask = _empty(size)
    c = size / 2
    add_cylinder(vol, mask, (c, c, 0), (c, c, size - 1), radius, hu=HU_VESSEL)
    return Phantom("vessel_only", vol, mask, np.zeros_like(mask),
                   meta={"vessel_radius": radius})


def simple_sphere(size: int = 64, radius: float = 8.0) -> Phantom:
    """Pure geometric sphere (no surrounding parenchyma)."""
    vol, mask = _empty(size)
    add_sphere(vol, mask, (size / 2,) * 3, radius)
    return Phantom("simple_sphere", vol, mask, np.zeros_like(mask),
                   meta={"radius_vox": radius})


def simple_cylinder(size: int = 64, radius: float = 4.0) -> Phantom:
    vol, mask = _empty(size)
    c = size / 2
    add_cylinder(vol, mask, (c, c, 0), (c, c, size - 1), radius, hu=HU_TUMOR)
    return Phantom("simple_cylinder", vol, mask, np.zeros_like(mask),
                   meta={"radius_vox": radius})


def simple_curved_surface(size: int = 64, radius: float = 22.0) -> Phantom:
    """Portion of a curved surface (subset of large sphere)."""
    vol, mask = _empty(size)
    add_curved_wall(vol, mask, (size + 8, size / 2, size / 2),
                     radius, thickness=3.0, hu=HU_TUMOR)
    return Phantom("simple_curved_surface", vol, mask, np.zeros_like(mask),
                   meta={"sphere_radius": radius})


# =====================================================================
# Soft (Gaussian / sigmoid-edge) variants
# =====================================================================
#
# Real CT exhibits partial-volume effects: lesion boundaries are not
# step functions. The binary-mask phantoms above tend to push every
# boundary voxel into the "plate" class. The soft variants below model
# a smooth HU transition over a 2-3 voxel edge band, producing a much
# more biology-like Hessian response and a more diverse class
# distribution.

def _sigmoid_edge(d: np.ndarray, edge_softness: float) -> np.ndarray:
    """Smooth 0→1 transition at d=0 with characteristic width `edge_softness`.

    d > 0 inside, d < 0 outside.
    """
    return 1.0 / (1.0 + np.exp(-d / max(edge_softness, 1e-6)))


def add_soft_sphere(vol: np.ndarray, mask: np.ndarray,
                     center: tuple[float, float, float], radius: float,
                     fg: float = HU_TUMOR, bg: float | None = None,
                     edge_softness: float = 1.2) -> None:
    """Sphere with sigmoid HU transition at radius.

    `mask` is updated to include voxels where the foreground weight
    exceeds 0.5 (so the "ground truth" tracks the half-amplitude
    contour — closer to how a radiologist would draw the boundary).
    """
    if bg is None:
        bg = float(vol.min())   # leave parenchyma value
    cz, cy, cx = center
    z, y, x = np.indices(vol.shape, dtype=np.float32)
    rho = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
    w = _sigmoid_edge(radius - rho, edge_softness)
    new_val = bg + (fg - bg) * w
    update = w > 0.05            # only touch voxels with appreciable weight
    vol[update] = np.maximum(vol[update], new_val[update])
    mask |= w >= 0.5


def add_soft_spike(vol: np.ndarray, mask: np.ndarray,
                    center: tuple[float, float, float], radius: float,
                    direction: tuple[float, float, float],
                    length: float, base_radius: float = 1.6,
                    tip_radius: float = 0.5,
                    fg: float = HU_TUMOR, bg: float | None = None,
                    edge_softness: float = 0.8) -> None:
    """Tapered spike with sigmoid edge along + perpendicular to axis."""
    if bg is None:
        bg = float(vol.min())
    z, y, x = np.indices(vol.shape, dtype=np.float32)
    c = np.asarray(center, dtype=np.float32)
    d = np.asarray(direction, dtype=np.float32)
    d /= (np.linalg.norm(d) + 1e-12)
    rel = np.stack([z - c[0], y - c[1], x - c[2]], axis=-1)
    along = rel @ d
    perp = rel - along[..., None] * d
    perp_dist = np.linalg.norm(perp, axis=-1)
    # along weight: 1 inside [radius-1, radius+length], 0 outside
    s_along = _sigmoid_edge(along - (radius - 1), edge_softness) \
              * _sigmoid_edge((radius + length) - along, edge_softness)
    # perpendicular tapered radius
    t = np.clip((along - radius + 1) / max(length, 1e-6), 0, 1)
    r_t = base_radius + (tip_radius - base_radius) * t
    s_perp = _sigmoid_edge(r_t - perp_dist, edge_softness)
    w = s_along * s_perp
    new_val = bg + (fg - bg) * w
    update = w > 0.05
    vol[update] = np.maximum(vol[update], new_val[update])
    mask |= w >= 0.5


def add_soft_cylinder(vol: np.ndarray, mask: np.ndarray | None,
                       start: tuple[float, float, float],
                       end: tuple[float, float, float],
                       radius: float, fg: float = HU_VESSEL,
                       bg: float | None = None,
                       edge_softness: float = 1.0) -> None:
    """Cylinder with sigmoid radial fall-off (vessel-like)."""
    if bg is None:
        bg = float(vol.min())
    z, y, x = np.indices(vol.shape, dtype=np.float32)
    s = np.asarray(start, dtype=np.float32)
    e = np.asarray(end, dtype=np.float32)
    axis = e - s
    length = np.linalg.norm(axis)
    axis_unit = axis / (length + 1e-12)
    rel = np.stack([z - s[0], y - s[1], x - s[2]], axis=-1)
    along = rel @ axis_unit
    perp = rel - along[..., None] * axis_unit
    perp_dist = np.linalg.norm(perp, axis=-1)
    s_along = _sigmoid_edge(along, edge_softness) \
              * _sigmoid_edge(length - along, edge_softness)
    s_perp = _sigmoid_edge(radius - perp_dist, edge_softness)
    w = s_along * s_perp
    new_val = bg + (fg - bg) * w
    update = w > 0.05
    vol[update] = np.maximum(vol[update], new_val[update])
    if mask is not None:
        mask |= w >= 0.5


# =====================================================================
# Soft phantom presets — drop-in replacements with realistic boundaries
# =====================================================================

def soft_benign_nodule(size: int = 64, radius: float = 6.0) -> Phantom:
    vol, mask = _empty(size)
    c = (size / 2, size / 2, size / 2)
    add_soft_sphere(vol, mask, c, radius)
    return Phantom("soft_benign_nodule", vol, mask, np.zeros_like(mask),
                   meta={"radius_vox": radius, "edge": "sigmoid"})


def soft_lung_cancer_spiculated(size: int = 64, radius: float = 10.0,
                                  n_spikes: int = 6, spike_len: float = 5.0
                                  ) -> Phantom:
    vol, mask = _empty(size)
    c = (size / 2, size / 2, size / 2)
    add_soft_sphere(vol, mask, c, radius, edge_softness=1.2)
    rng = np.random.default_rng(0)
    base_dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                 (0, 0, 1), (0, 0, -1),
                 (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                 (1, 0, 1), (-1, 0, 1)][:n_spikes]
    for d in base_dirs:
        jitter = rng.normal(0, 0.05, size=3)
        dir_j = np.array(d, dtype=np.float32) + jitter
        add_soft_spike(vol, mask, c, radius, tuple(dir_j.tolist()),
                       length=spike_len + rng.uniform(-1, 1),
                       base_radius=1.5, tip_radius=0.4)
    return Phantom("soft_lung_cancer_spiculated", vol, mask,
                   np.zeros_like(mask),
                   meta={"radius_vox": radius, "n_spikes": n_spikes,
                         "spike_len": spike_len, "edge": "sigmoid"})


def soft_lobulated_nodule(size: int = 64, radius: float = 7.0,
                            n_lobes: int = 3) -> Phantom:
    vol, mask = _empty(size)
    c = (size / 2, size / 2, size / 2)
    add_soft_sphere(vol, mask, c, radius, edge_softness=1.2)
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (-1, 1, 0), (1, -1, 1)][:n_lobes]
    for d in directions:
        dn = np.asarray(d, dtype=np.float32)
        dn = dn / (np.linalg.norm(dn) + 1e-12)
        bulge_center = np.asarray(c) + (radius - 1.5) * dn
        add_soft_sphere(vol, mask, tuple(bulge_center.tolist()),
                         radius=3.0, edge_softness=1.0)
    return Phantom("soft_lobulated_nodule", vol, mask,
                   np.zeros_like(mask),
                   meta={"radius_vox": radius, "n_lobes": n_lobes,
                         "edge": "sigmoid"})


def soft_juxta_vascular_cancer(size: int = 64, radius: float = 10.0,
                                  n_spikes: int = 4,
                                  vessel_radius: float = 2.0) -> Phantom:
    vol, mask = _empty(size)
    c = (size / 2, size / 2, size / 2)
    add_soft_sphere(vol, mask, c, radius, edge_softness=1.2)
    v_start = (c[0], c[1] - radius, 0)
    v_end = (c[0], c[1] - radius, size - 1)
    attachment = np.zeros_like(mask)
    add_soft_cylinder(vol, attachment, v_start, v_end, vessel_radius,
                       edge_softness=0.8)
    from scipy.ndimage import binary_dilation
    attachment = attachment & binary_dilation(mask, iterations=2)
    for d in [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)][:n_spikes]:
        add_soft_spike(vol, mask, c, radius, d, length=5)
    return Phantom("soft_juxta_vascular_cancer", vol, mask, attachment,
                   meta={"radius_vox": radius, "vessel_radius": vessel_radius,
                         "n_spikes": n_spikes, "edge": "sigmoid"})


def soft_juxta_pleural_cancer(size: int = 64, radius: float = 10.0,
                                 n_spikes: int = 4) -> Phantom:
    vol, mask = _empty(size)
    c = (size / 2, size / 2, size / 2)
    add_soft_sphere(vol, mask, c, radius, edge_softness=1.2)
    # Soft pleural wall — large-radius sphere shell with sigmoid edge
    wall_center = (c[0], c[1] + 30, c[2])
    attachment = np.zeros_like(mask)
    z, y, x = np.indices(vol.shape, dtype=np.float32)
    cz, cy, cx = wall_center
    rho = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
    w = _sigmoid_edge(1.5 - np.abs(rho - 30.0), 0.8)
    vol[w > 0.05] = np.maximum(vol[w > 0.05],
                                HU_PARENCHYMA + (HU_WALL - HU_PARENCHYMA) * w[w > 0.05])
    attachment_full = w >= 0.5
    from scipy.ndimage import binary_dilation
    attachment = attachment_full & binary_dilation(mask, iterations=2)
    for d in [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)][:n_spikes]:
        add_soft_spike(vol, mask, c, radius, d, length=5)
    return Phantom("soft_juxta_pleural_cancer", vol, mask, attachment,
                   meta={"radius_vox": radius, "n_spikes": n_spikes,
                         "edge": "sigmoid"})


def soft_vessel_only(size: int = 64, radius: float = 2.0) -> Phantom:
    vol, mask = _empty(size)
    c = size / 2
    add_soft_cylinder(vol, mask, (c, c, 0), (c, c, size - 1), radius)
    return Phantom("soft_vessel_only", vol, mask, np.zeros_like(mask),
                   meta={"vessel_radius": radius, "edge": "sigmoid"})


def soft_sphere(size: int = 64, radius: float = 8.0) -> Phantom:
    """Pure soft-Gaussian sphere — the cleanest comparison to simple_sphere."""
    vol, mask = _empty(size)
    add_soft_sphere(vol, mask, (size / 2,) * 3, radius, edge_softness=1.0)
    return Phantom("soft_sphere", vol, mask, np.zeros_like(mask),
                   meta={"radius_vox": radius, "edge": "sigmoid"})


# =====================================================================
# Bulk registry
# =====================================================================

ALL_PHANTOMS: dict[str, callable] = {
    "benign_nodule": benign_nodule,
    "lung_cancer_spiculated": lung_cancer_spiculated,
    "lobulated_nodule": lobulated_nodule,
    "juxta_vascular_cancer": juxta_vascular_cancer,
    "juxta_pleural_cancer": juxta_pleural_cancer,
    "airway_only": airway_only,
    "vessel_only": vessel_only,
    "simple_sphere": simple_sphere,
    "simple_cylinder": simple_cylinder,
    "simple_curved_surface": simple_curved_surface,
    # soft Gaussian / sigmoid-edge variants (realistic CT boundaries)
    "soft_benign_nodule": soft_benign_nodule,
    "soft_lobulated_nodule": soft_lobulated_nodule,
    "soft_lung_cancer_spiculated": soft_lung_cancer_spiculated,
    "soft_juxta_vascular_cancer": soft_juxta_vascular_cancer,
    "soft_juxta_pleural_cancer": soft_juxta_pleural_cancer,
    "soft_vessel_only": soft_vessel_only,
    "soft_sphere": soft_sphere,
}

SOFT_PHANTOMS = (
    "soft_benign_nodule", "soft_lobulated_nodule",
    "soft_lung_cancer_spiculated",
    "soft_juxta_vascular_cancer", "soft_juxta_pleural_cancer",
    "soft_vessel_only", "soft_sphere",
)


def make_all(size: int = 64) -> list[Phantom]:
    """Generate every phantom at the default settings."""
    return [factory(size=size) for factory in ALL_PHANTOMS.values()]


def make(name: str, **kwargs) -> Phantom:
    if name not in ALL_PHANTOMS:
        raise KeyError(f"unknown phantom: {name}; available: {sorted(ALL_PHANTOMS)}")
    return ALL_PHANTOMS[name](**kwargs)
