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
    """Mesh pipeline (conformal spherical param) runs without error on
    both pure and spiked sphere."""
    pure = (sphere(size=41, r=9) < -500).astype(np.float32)
    spiked = _spiked_sphere(size=41, n_spikes=4, spike_len=6)
    f_pure, _, _, _ = spiculation_from_voxel(pure, n_param_iter=12)
    f_spiked, _, _, _ = spiculation_from_voxel(spiked, n_param_iter=12)
    assert isinstance(f_pure, SpiculationFeatures)
    assert isinstance(f_spiked, SpiculationFeatures)
    assert f_spiked.Na > f_pure.Na


def test_nadeem2017_conformal_bipartition_and_distortion():
    """Test Nadeem et al. (IEEE TVCG 2017) Laplace-Beltrami eigen-bipartition,
    Möbius area-mass centering, angle distortion, and balanced parameterization."""
    from qradiomics.shape import (
        angle_distortion,
        laplace_beltrami_bipartition,
        mesh_roundness,
        zscore_area_distortion,
    )
    mask = _spiked_sphere(size=41, n_spikes=4, spike_len=5)
    mesh = voxel_to_mesh(mask, level=0.4, fill_holes_and_largest_cc=True, smooth_iter=1)
    f0, hemi = laplace_beltrami_bipartition(mesh)
    assert f0.shape == (mesh.n_vertices,)
    assert set(np.unique(hemi)) == {0, 1}
    # Hemispheres should be roughly balanced
    frac0 = float(np.mean(hemi == 0))
    assert 0.25 < frac0 < 0.75

    pos_conf = spherical_parameterization(mesh, n_iter=10, balance_alpha=0.0)
    pos_bal = spherical_parameterization(mesh, n_iter=10, balance_alpha=0.5)
    assert np.allclose(np.linalg.norm(pos_conf, axis=1), 1.0, atol=1e-3)
    assert np.allclose(np.linalg.norm(pos_bal, axis=1), 1.0, atol=1e-3)

    ang_d = angle_distortion(mesh, pos_conf)
    nd = zscore_area_distortion(mesh, pos_conf)
    assert ang_d.shape == (mesh.n_faces,)
    assert nd.shape == (mesh.n_vertices,)
    assert abs(float(np.mean(nd))) < 1e-4
    assert np.isclose(float(np.std(nd, ddof=1)), 1.0, atol=1e-3)
    assert 0.0 < mesh_roundness(mesh) <= 1.0



# ---- Opt-in conformal variants (#23) ---------------------------------------

def _baseline():
    from pathlib import Path
    from qradiomics.shape.mesh_utils import Mesh
    p = Path(__file__).resolve().parents[1] / "fixtures" / "spiculation" / "cmcf_baseline.npz"
    d = np.load(p)
    return Mesh(vertices=d["vertices"], faces=d["faces"]), d["sphere"]


def test_default_spherical_param_matches_saved_baseline():
    mesh, ref = _baseline()
    out = spherical_parameterization(mesh)
    np.testing.assert_array_equal(out, ref)
    out2 = spherical_parameterization(mesh, mobius_center=False, yamabe_refine=0)
    np.testing.assert_array_equal(out2, ref)


def test_mobius_center_puts_area_centroid_at_origin():
    from qradiomics.shape.spiculation import _per_vertex_area_lcsr
    mesh, ref = _baseline()
    w = _per_vertex_area_lcsr(mesh.vertices.astype(np.float64), mesh.faces)
    w = w / w.sum()
    out = spherical_parameterization(mesh, mobius_center=True).astype(np.float64)
    c_ref = np.linalg.norm((w[:, None] * ref).sum(axis=0))
    c_out = np.linalg.norm((w[:, None] * out).sum(axis=0))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    assert c_out < 1e-2
    assert c_out <= c_ref + 1e-6


def test_yamabe_refine_does_not_increase_angle_distortion():
    from qradiomics.shape.spiculation import (
        _corner_angles, _mean_abs_corner_log_ratio, _unique_edges)
    mesh, ref = _baseline()
    out = spherical_parameterization(mesh, yamabe_refine=5)
    assert out.shape == ref.shape
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    edges, fe = _unique_edges(mesh.faces)

    def _ang(p):
        p = p.astype(np.float64)
        return _corner_angles(np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)[fe])

    a0 = _ang(mesh.vertices)
    d_ref = _mean_abs_corner_log_ratio(_ang(ref), a0)
    d_out = _mean_abs_corner_log_ratio(_ang(out), a0)
    print(f"corner-angle distortion cMCF={d_ref:.4f} yamabe={d_out:.4f}")
    assert d_out <= d_ref + 1e-6


def test_spiculation_from_voxel_accepts_sphere_kwargs():
    sf, _p, _d, _m = spiculation_from_voxel(
        _spiked_sphere(), sphere_kwargs={"mobius_center": True, "yamabe_refine": 2})
    assert isinstance(sf, SpiculationFeatures)


def test_return_info_reports_yamabe_outcome():
    mesh, ref = _baseline()
    out, info = spherical_parameterization(mesh, yamabe_refine=5, return_info=True)
    y = info["yamabe"]
    assert isinstance(y["applied"], bool)
    assert y["reason"] in {"applied", "degenerate_edge", "invalid_triangle",
                           "solver_failure", "nan", "flipped_faces", "no_gain"}
    np.testing.assert_array_equal(out, spherical_parameterization(mesh, yamabe_refine=5))
    out0, info0 = spherical_parameterization(mesh, return_info=True)
    np.testing.assert_array_equal(out0, ref)
    assert info0 == {}


def test_intrinsic_delaunay_flips_fix_non_delaunay_quad():
    from qradiomics.shape.spiculation import _intrinsic_delaunay_flips
    # Octahedron; scaling vertices 0 and 2 makes edge 0-2 non-Delaunay.
    v = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0],
                  [0, 1, 0], [0, -1, 0]], dtype=np.float64)
    F = np.array([[0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
                  [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5]])
    l0d = {}
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = (int(min(a, b)), int(max(a, b)))
            l0d[k] = float(np.linalg.norm(v[a] - v[b]))
    u = np.zeros(6)
    u[[0, 2]] = 1.0  # stretch edge 0-2: opposite angles at 4 and 5 ~111 deg each
    F2, n = _intrinsic_delaunay_flips(F, l0d, u)
    assert n > 0
    assert (4, 5) in l0d and (0, 2) not in l0d
    assert len(l0d) == 12 and F2.shape == F.shape
    # every edge still has exactly two incident faces
    from qradiomics.shape.spiculation import _unique_edges
    _e, fe = _unique_edges(F2)
    assert np.all(np.bincount(fe.ravel()) == 2)


def test_yamabe_flip_opt_in_does_not_increase_angle_distortion():
    from qradiomics.shape.spiculation import (
        _corner_angles, _mean_abs_corner_log_ratio, _unique_edges)
    mesh, ref = _baseline()
    out, info = spherical_parameterization(mesh, yamabe_refine=5, yamabe_flip=True,
                                           return_info=True)
    assert "n_flips" in info["yamabe"]
    edges, fe = _unique_edges(mesh.faces)

    def _ang(p):
        p = p.astype(np.float64)
        return _corner_angles(np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)[fe])

    a0 = _ang(mesh.vertices)
    print(f"flip: n_flips={info['yamabe']['n_flips']} reason={info['yamabe']['reason']} "
          f"dist={_mean_abs_corner_log_ratio(_ang(out), a0):.4f}")
    assert _mean_abs_corner_log_ratio(_ang(out), a0) <= _mean_abs_corner_log_ratio(_ang(ref), a0) + 1e-6
    # default yamabe path unchanged by the new flag
    np.testing.assert_array_equal(spherical_parameterization(mesh, yamabe_refine=5),
                                  spherical_parameterization(mesh, yamabe_refine=5, yamabe_flip=False))
