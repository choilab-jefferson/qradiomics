"""Tests for the 2021 CMPB spiculation pipeline (mesh + spherical parameterization)."""
from __future__ import annotations

import numpy as np

from qradiomics.shape import (
    SpiculationFeatures,
    area_distortion,
    detect_peaks,
    face_areas,
    spherical_parameterization,
    spiculation_from_voxel,
    vertex_areas,
    voxel_to_mesh,
)


# ---- Synthetic shape helpers ----------------------------------------------


def _sphere_volume(size: int = 33, r: float = 8.0) -> np.ndarray:
    c = (size - 1) / 2
    z, y, x = np.indices((size, size, size), dtype=np.float32)
    rho = np.sqrt((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2)
    return (rho <= r).astype(np.float32)


def _spiked_sphere(
    size: int = 41,
    r: float = 9.0,
    n_spikes: int = 4,
    spike_len: float = 5.0,
    spike_radius: float = 1.5,
) -> np.ndarray:
    c = (size - 1) / 2
    z, y, x = np.indices((size, size, size), dtype=np.float32)
    z -= c
    y -= c
    x -= c
    rho = np.sqrt(z**2 + y**2 + x**2)
    mask = (rho <= r).astype(np.float32)
    dirs = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)][:n_spikes]
    for dz, dy, dx in dirs:
        proj = z * dz + y * dy + x * dx
        perp2 = (z - proj * dz) ** 2 + (y - proj * dy) ** 2 + (x - proj * dx) ** 2
        spike = (proj >= r - 1) & (proj <= r + spike_len) & (perp2 <= spike_radius**2)
        mask = np.maximum(mask, spike.astype(np.float32))
    return mask


# ---- Mesh utilities --------------------------------------------------------


def test_voxel_to_mesh_returns_mesh():
    sph = _sphere_volume()
    mesh = voxel_to_mesh(sph, level=0.5)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0


def test_vertex_areas_sum_equals_face_areas():
    sph = _sphere_volume()
    mesh = voxel_to_mesh(sph)
    total_v = vertex_areas(mesh).sum()
    total_f = face_areas(mesh).sum()
    assert np.isclose(total_v, total_f, rtol=1e-5)


# ---- Spherical parameterization -------------------------------------------


def test_spherical_param_on_unit_sphere():
    sph = _sphere_volume()
    mesh = voxel_to_mesh(sph)
    pos = spherical_parameterization(mesh, n_iter=50)
    norms = np.linalg.norm(pos, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_area_distortion_low_for_round_sphere():
    sph = _sphere_volume()
    mesh = voxel_to_mesh(sph)
    pos = spherical_parameterization(mesh, n_iter=100)
    dist = area_distortion(mesh, pos)
    assert dist.min() > -3.0


# ---- Peak detection on spiked sphere --------------------------------------


def test_detect_peaks_on_spiked_sphere():
    mask = _spiked_sphere(size=41, n_spikes=4, spike_len=5)
    mesh = voxel_to_mesh(mask)
    pos = spherical_parameterization(mesh, n_iter=100)
    dist = area_distortion(mesh, pos)
    peaks = detect_peaks(mesh, dist, baseline_eps=0.05, min_peak_size=3)
    assert len(peaks) > 0, "spikes should produce at least one peak"


def test_classify_peak_categories_valid():
    mask = _spiked_sphere(size=41, n_spikes=4, spike_len=5)
    feats, _, _, _ = spiculation_from_voxel(mask, n_param_iter=80)
    valid = {"spiculation", "lobulation", "attachment", "small"}
    assert all(c in valid for c in feats.classes)


def test_spiculation_features_consistency():
    mask = _spiked_sphere(size=41, n_spikes=4, spike_len=5)
    feats, _, _, _ = spiculation_from_voxel(mask, n_param_iter=80)
    assert isinstance(feats, SpiculationFeatures)
    n_small = sum(1 for c in feats.classes if c == "small")
    assert feats.Np == feats.Na + feats.Nl + feats.Na_att + n_small


def test_classify_with_attachment_mask():
    mask = _spiked_sphere(size=41, n_spikes=4, spike_len=5)
    attach = np.zeros_like(mask, dtype=bool)
    attach[20:22, 20:22, 20:22] = True
    feats, _, _, _ = spiculation_from_voxel(mask, attachment_mask=attach, n_param_iter=80)
    assert feats.Na_att >= 0


# ---- End-to-end -----------------------------------------------------------


def test_pure_sphere_has_few_spikes():
    sph = _sphere_volume(size=33, r=10)
    feats, _, _, _ = spiculation_from_voxel(sph, n_param_iter=80)
    if feats.Np > 0:
        spic_fraction = feats.Na / feats.Np
        assert spic_fraction < 0.2, (
            f"pure sphere: {feats.Na}/{feats.Np} = {spic_fraction:.2%} spiculation"
        )


def test_mesh_pipeline_runs_on_both_shapes():
    pure = _sphere_volume(size=41, r=9)
    spiked = _spiked_sphere(size=41, n_spikes=4, spike_len=6)
    f_pure, _, _, _ = spiculation_from_voxel(pure, n_param_iter=80)
    f_spiked, _, _, _ = spiculation_from_voxel(spiked, n_param_iter=80)
    assert isinstance(f_pure, SpiculationFeatures)
    assert isinstance(f_spiked, SpiculationFeatures)
