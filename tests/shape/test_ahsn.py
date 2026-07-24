"""Smoke + sanity tests for AHSN (2014) and SNoH (2026)."""
from __future__ import annotations

import numpy as np
import pytest

# This test file imports symbols from the qradiomics_private overlay
# (SNoH, spiculation_from_voxel_native). Skip the whole module in public-only
# installs where the overlay is not present.
pytest.importorskip("qradiomics_private", reason="requires qradiomics_private overlay")

from qradiomics.shape import (  # type: ignore[import-not-found]
    AHSNConfig, SNoHConfig, ahsn, snoh, dot_value, make_scales,
    surface_elements,
)
from qradiomics.shape.phantoms import sphere, cylinder, plane, saddle  # type: ignore


# ---- AHSN (2014) ----------------------------------------------------------

def test_ahsn_dim_default():
    desc = ahsn(sphere())
    assert desc.shape == (AHSNConfig().dim,)
    assert desc.dtype == np.float32


def test_ahsn_normalization():
    desc = ahsn(sphere(), cfg=AHSNConfig(n_bins_theta=10, n_bins_phi=10))
    assert np.isclose(desc[:10].sum(), 1.0, atol=1e-5)
    assert np.isclose(desc[10:].sum(), 1.0, atol=1e-5)


# ---- SNoH (2026) ----------------------------------------------------------

def test_snoh_dim_default():
    desc = snoh(sphere())
    cfg = SNoHConfig()
    assert desc.shape == (cfg.dim,)


def test_snoh_dim_is_thousand_class():
    """SNoH default should be in the thousand-dim regime, not SIFT-sized."""
    assert SNoHConfig().dim >= 1000


def _cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_snoh_separates_shapes():
    """Sphere/cylinder/plane/saddle should have low pairwise cosine sim."""
    cfg = SNoHConfig(n_scales=2, d_min=4.0, d_max=12.0,
                     n_bins_theta=18, n_bins_phi=18,
                     n_bins_shape_index=8, n_bins_curvedness=8)
    sigs = {name: snoh(make(), cfg=cfg)
            for name, make in {"sphere": sphere, "cylinder": cylinder,
                                "plane": plane, "saddle": saddle}.items()}
    cos = _cosine(sigs["sphere"], sigs["plane"])
    assert cos < 0.95, f"sphere vs plane too similar: {cos:.3f}"
    cos = _cosine(sigs["sphere"], sigs["cylinder"])
    assert cos < 0.99, f"sphere vs cylinder too similar: {cos:.3f}"


# ---- Dot enhancement filter ------------------------------------------------

def test_dot_value_bright_blob_positive():
    v = sphere()
    se = surface_elements(v, sigma=2.0)
    d = dot_value(se.eigvals)
    c = v.shape[0] // 2
    assert d[c, c, c] > 0


def test_dot_value_zero_on_background():
    v = np.full((17, 17, 17), -900, dtype=np.float32)
    se = surface_elements(v, sigma=2.0)
    d = dot_value(se.eigvals)
    assert np.allclose(d, 0.0)


def test_make_scales_endpoints():
    s = make_scales(3.0, 30.0, 5)
    assert pytest.approx(s[0]) == 3.0 / 4.0
    assert pytest.approx(s[-1]) == 30.0 / 4.0
    assert len(s) == 5


# ---- Wall elimination ------------------------------------------------------

def test_wall_eliminate_runs_and_reduces_mask():
    from qradiomics.shape import wall_eliminate  # type: ignore[import-not-found]
    v = plane(size=33, thickness=4)
    _, keep = wall_eliminate(v, cfg=AHSNConfig(n_bins_theta=18, n_bins_phi=18),
                              area_threshold=20)
    assert keep.sum() < v.size
