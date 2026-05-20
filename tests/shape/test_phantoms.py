"""Construction tests for synthetic lung phantoms."""
from __future__ import annotations

import numpy as np

from qradiomics.shape import ALL_PHANTOMS, Phantom, make, make_all
from qradiomics.shape.phantoms import HU_TUMOR


def test_all_phantoms_buildable():
    """Every phantom factory produces a valid Phantom object."""
    phantoms = make_all(size=48)
    assert len(phantoms) == len(ALL_PHANTOMS)
    for p in phantoms:
        assert isinstance(p, Phantom)
        assert p.volume.shape == (48, 48, 48)
        assert p.mask.shape == (48, 48, 48)
        assert p.mask.dtype == bool


def test_phantom_volume_is_in_hu_range():
    p = make("lung_cancer_spiculated", size=48)
    assert p.volume.min() >= -1100
    assert p.volume.max() <= 300


def test_spiculated_has_tumor_voxels():
    p = make("lung_cancer_spiculated", size=48)
    assert (p.volume == HU_TUMOR).any()
    assert p.mask.sum() > 0


def test_juxta_vascular_has_attachment():
    p = make("juxta_vascular_cancer", size=64)
    assert p.attachment_mask.sum() > 0


def test_ahsn_runs_on_each_phantom():
    """AHSN descriptor produces correct shape on every phantom."""
    from qradiomics.shape import AHSNConfig, ahsn

    cfg = AHSNConfig()
    for name in ALL_PHANTOMS:
        p = make(name, size=33)
        desc = ahsn(p.volume, cfg=cfg)
        assert desc.shape == (cfg.dim,), f"{name}: expected {cfg.dim}, got {desc.shape}"
        assert desc.dtype == np.float32
