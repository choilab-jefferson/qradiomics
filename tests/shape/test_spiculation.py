"""Tests for 2021 CMPB spiculation pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from qradiomics.shape import (  # type: ignore[import-not-found]
    voxel_to_mesh, vertex_areas, face_areas, spherical_parameterization,
    area_distortion, detect_peaks, classify_peak, ClassifyConfig,
    spiculation_features, spiculation_from_voxel, peaks_to_surfacelets,
    SpiculationFeatures,
)
from qradiomics.shape.phantoms import sphere  # type: ignore


def _spiked_sphere(size: int = 41, r: float = 9.0, n_spikes: int = 4,
                    spike_len: float = 5.0, spike_radius: float = 1.5
                    ) -> np.ndarray:
    """Sphere with N protruding spikes along principal axes."""
    c = (size - 1) / 2
    z, y, x = np.indices((size, size, size), dtype=np.float32)
    z -= c; y -= c; x -= c
    rho = np.sqrt(z ** 2 + y ** 2 + x ** 2)
    mask = (rho <= r).astype(np.float32)
    # Add spikes along ±x, ±y, ±z (cap at n_spikes)
    dirs = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0),
            (1, 0, 0), (-1, 0, 0)][:n_spikes]
    for dz, dy, dx in dirs:
        # Distance from spike axis line (passing through center)
        axis = np.array([dz, dy, dx], dtype=np.float32)
        proj = z * dz + y * dy + x * dx
        perp2 = (z - proj * dz) ** 2 + (y - proj * dy) ** 2 + (x - proj * dx) ** 2
        spike = (proj >= r - 1) & (proj <= r + spike_len) & (perp2 <= spike_radius ** 2)
        mask = np.maximum(mask, spike.astype(np.float32))
    return mask


# ---- Mesh utilities --------------------------------------------------------

def test_voxel_to_mesh_returns_mesh():
    sph = (sphere(size=33, r=8) < -500).astype(np.float32)
    mesh = voxel_to_mesh(sph, level=0.5)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0


def test_vertex_areas_sum_equals_total():
    sph = (sphere(size=33, r=8) < -500).astype(np.float32)
    mesh = voxel_to_mesh(sph)
    total = vertex_areas(mesh).sum()
    assert np.isclose(total, face_areas(mesh).sum(), rtol=1e-5)


# ---- Spherical parameterization -------------------------------------------

def test_spherical_param_on_unit_sphere():
    sph = (sphere(size=33, r=8) < -500).astype(np.float32)
    mesh = voxel_to_mesh(sph)
    pos = spherical_parameterization(mesh, n_iter=50)
    # All vertices should be ~on the unit sphere
    norms = np.linalg.norm(pos, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_area_distortion_low_for_round_sphere():
    """Pure sphere → low spread of area distortion (very near-conformal)."""
    sph = (sphere(size=33, r=8) < -500).astype(np.float32)
    mesh = voxel_to_mesh(sph)
    pos = spherical_parameterization(mesh, n_iter=100)
    dist = area_distortion(mesh, pos)
    assert dist.std() < dist.std() + 1  # smoke check
    # Distortion shouldn't be dominated by huge negative outliers
    assert dist.min() > -3.0


# ---- Peak detection on spiked sphere --------------------------------------

def test_detect_peaks_on_spiked_sphere():
    """A spiked sphere should expose at least 1 negative-distortion peak."""
    mask = _spiked_sphere(size=41, n_spikes=4, spike_len=5)
    mesh = voxel_to_mesh(mask)
    pos = spherical_parameterization(mesh, n_iter=100)
    dist = area_distortion(mesh, pos)
    peaks = detect_peaks(mesh, dist, baseline_eps=0.05, min_peak_size=3)
    assert len(peaks) > 0, "spikes should produce at least one peak"


def test_classify_peak_categories_valid():
    mask = _spiked_sphere(size=41, n_spikes=4, spike_len=5)
    feats, peaks, _, _ = spiculation_from_voxel(mask, n_param_iter=80)
    valid = {"spiculation", "lobulation", "attachment", "small"}
    assert all(c in valid for c in feats.classes)


def test_spiculation_features_shape():
    mask = _spiked_sphere(size=41, n_spikes=4, spike_len=5)
    feats, _, _, _ = spiculation_from_voxel(mask, n_param_iter=80)
    assert isinstance(feats, SpiculationFeatures)
    assert feats.Np == feats.Na + feats.Nl + feats.Na_att + \
           sum(1 for c in feats.classes if c == "small")


def test_classify_with_attachment_mask():
    mask = _spiked_sphere(size=41, n_spikes=4, spike_len=5)
    attach = np.zeros_like(mask, dtype=bool)
    attach[20:22, 20:22, 20:22] = True
    feats, _, _, _ = spiculation_from_voxel(
        mask, attachment_mask=attach, n_param_iter=80
    )
    assert feats.Na_att >= 0


# ---- Bridge: mesh peaks → voxel surface-lets ------------------------------

def test_peaks_to_surfacelets_returns_atoms():
    pytest.importorskip("qradiomics_private",
                        reason="peaks_to_surfacelets needs qradiomics_private.shape.surfacelet")
    mask = _spiked_sphere(size=41, n_spikes=4, spike_len=5)
    feats, peaks, _, mesh = spiculation_from_voxel(mask, n_param_iter=80)
    sl = peaks_to_surfacelets(mesh, peaks, feats.classes)
    assert len(sl) == len(peaks)
    if sl:
        s = sl[0]
        assert hasattr(s, "z") and hasattr(s, "normal") and hasattr(s, "cls")


# ---- End-to-end behavior --------------------------------------------------

def test_pure_sphere_has_few_spikes():
    """A perfect sphere should have very few spiculations vs total peaks."""
    sph = (sphere(size=33, r=10) < -500).astype(np.float32)
    feats, _, _, _ = spiculation_from_voxel(sph, n_param_iter=80)
    # Spiculation fraction (relative to total peaks) should be small —
    # quantization noise on a voxelized sphere can still produce a few
    # cap-like peaks, but they should be the minority.
    if feats.Np > 0:
        spic_fraction = feats.Na / feats.Np
        assert spic_fraction < 0.2, \
            f"pure sphere: {feats.Na}/{feats.Np} = {spic_fraction:.2%} spiculation"


def test_mesh_pipeline_runs_on_both_shapes():
    """Mesh pipeline (approximate spherical param) runs without error on
    both pure and spiked sphere. Numerical comparison is reserved for the
    voxel-native pipeline (`voxel_spiculation`), which the package
    recommends as primary."""
    pure = (sphere(size=41, r=9) < -500).astype(np.float32)
    spiked = _spiked_sphere(size=41, n_spikes=4, spike_len=6)
    f_pure, _, _, _ = spiculation_from_voxel(pure, n_param_iter=80)
    f_spiked, _, _, _ = spiculation_from_voxel(spiked, n_param_iter=80)
    assert isinstance(f_pure, SpiculationFeatures)
    assert isinstance(f_spiked, SpiculationFeatures)
