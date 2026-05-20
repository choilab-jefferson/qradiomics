"""Smoke + sanity tests for the 2014 AHSN pipeline (hessian / detection / ahsn / wall_elim)."""
from __future__ import annotations

import numpy as np
import pytest

from qradiomics.shape import (
    AHSNConfig,
    ahsn,
    dot_value,
    make_scales,
    surface_elements,
    wall_eliminate,
)


# ---- Synthetic volumes (analytic shapes for which AHSN behaviour is known) -------


def _grid(size: int = 33) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = (size - 1) / 2
    z, y, x = np.indices((size, size, size), dtype=np.float32)
    return z - c, y - c, x - c


def sphere(size: int = 33, r: float = 8.0, bright: float = -200, bg: float = -900) -> np.ndarray:
    z, y, x = _grid(size)
    rho = np.sqrt(z**2 + y**2 + x**2)
    return np.where(rho <= r, bright, bg).astype(np.float32)


def cylinder(size: int = 33, r: float = 4.0, bright: float = -200, bg: float = -900) -> np.ndarray:
    _, y, x = _grid(size)
    rho = np.sqrt(y**2 + x**2)
    return np.where(rho <= r, bright, bg).astype(np.float32)


def plane(
    size: int = 33, thickness: float = 4.0, bright: float = -200, bg: float = -900
) -> np.ndarray:
    z, _, _ = _grid(size)
    return np.where(np.abs(z) <= thickness / 2, bright, bg).astype(np.float32)


def saddle(
    size: int = 33, scale: float = 12.0, bright: float = -200, bg: float = -900
) -> np.ndarray:
    z, y, x = _grid(size)
    surf = (x**2 - y**2) / scale
    return np.where(np.abs(z - surf) <= 2.0, bright, bg).astype(np.float32)


# ---- AHSN descriptor (paper §2.3.1) -------------------------------------------------


def test_ahsn_dim_default():
    desc = ahsn(sphere())
    assert desc.shape == (AHSNConfig().dim,)
    assert desc.dtype == np.float32


def test_ahsn_normalization():
    desc = ahsn(sphere(), cfg=AHSNConfig(n_bins_theta=10, n_bins_phi=10))
    assert np.isclose(desc[:10].sum(), 1.0, atol=1e-5)
    assert np.isclose(desc[10:].sum(), 1.0, atol=1e-5)


# ---- Dot enhancement filter (paper §2.2.2) -----------------------------------------


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


# ---- Wall elimination (paper §2.3.2, Algorithm 1) ----------------------------------


def test_wall_eliminate_runs_and_reduces_mask():
    v = plane(size=33, thickness=4)
    _, keep = wall_eliminate(
        v,
        cfg=AHSNConfig(n_bins_theta=18, n_bins_phi=18),
        area_threshold=20,
    )
    assert keep.sum() < v.size
