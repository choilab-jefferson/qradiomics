"""Tests for upstreamed P0/P1 primitives (#19):
- qradiomics.atomic.morphology (peritumoral_rim, boundary_band)
- qradiomics.atomic.io.save_image_and_mask
- qradiomics.atomic.preprocess.preprocess_pair 100%-filled ROI guard
- qradiomics.contract (Provenance, Encoding, roi_hash, check_fusable, gate_encodings_by_icc)
- qradiomics.analytics.robustness.volume_correlation_screen
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from qradiomics.analytics.robustness import volume_correlation_screen
from qradiomics.atomic import (
    boundary_band,
    load_image_and_mask,
    peritumoral_rim,
    preprocess_pair,
    save_image_and_mask,
)
from qradiomics.contract import (
    Encoding,
    FusionError,
    Provenance,
    check_fusable,
    gate_encodings_by_icc,
    roi_hash,
)


def test_peritumoral_rim_and_hu_floor():
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[6:10, 6:10, 6:10] = 1
    img = np.full((16, 16, 16), -100.0, dtype=np.float32)
    img[:8, :, :] = -800.0  # air on upper half

    rim = peritumoral_rim(mask, image=img, rim_mm=2.0, spacing_mm=(1.0, 1.0, 1.0), hu_floor=-400.0)
    assert rim.dtype == np.uint8
    assert rim.shape == mask.shape
    assert not np.any((mask > 0) & (rim > 0))
    assert not np.any(rim[:8, :, :] > 0)
    assert np.any(rim[8:, :, :] > 0)


def test_boundary_band_modes():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:15, 5:15, 5:15] = 1

    outer = boundary_band(mask, contour_type="outer", width_mm=2.0)
    inner = boundary_band(mask, contour_type="inner", width_mm=2.0)
    both = boundary_band(mask, contour_type="inner_outer", width_mm=2.0)
    full = boundary_band(mask, contour_type="full", width_mm=2.0)

    assert not np.any((mask > 0) & (outer > 0))
    assert np.all(inner[inner > 0] == mask[inner > 0])
    assert int(both.sum()) == int(outer.sum() + inner.sum())
    assert np.array_equal(full, mask)


def test_save_and_load_image_and_mask(tmp_path):
    img = np.random.default_rng(0).normal(size=(10, 12, 14)).astype(np.float32)
    msk = np.zeros((10, 12, 14), dtype=np.uint8)
    msk[2:6, 3:7, 4:8] = 1

    ip, mp = save_image_and_mask(
        img, msk, tmp_path / "img.nrrd", tmp_path / "msk.nrrd", spacing=(1.5, 1.5, 2.0)
    )
    loaded_img, loaded_msk = load_image_and_mask(ip, mp)
    assert loaded_img.GetSpacing() == pytest.approx((1.5, 1.5, 2.0))
    assert np.allclose(sitk.GetArrayFromImage(loaded_img), img)
    assert np.array_equal(sitk.GetArrayFromImage(loaded_msk), msk)


def test_preprocess_pair_refuses_100_percent_filled_crop():
    img = sitk.GetImageFromArray(np.ones((8, 8, 8), dtype=np.float32))
    msk = sitk.GetImageFromArray(np.ones((8, 8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="fills 100% of the cropped volume"):
        preprocess_pair(img, msk, pad_mm=0.0)


def test_contract_and_volume_correlation_screen():
    m = np.zeros((8, 8, 8), dtype=np.uint8)
    m[2:6, 2:6, 2:6] = 1
    h = roi_hash(m, (1.0, 1.0, 1.0))

    p1 = Provenance(
        encoder="e", encoder_version="1", modality="CT", scope="lesion", roi_role="gtv", roi=h
    )
    e1 = Encoding(
        "P1", np.array([1.0, 2.0], dtype=np.float32), p1, names=("f1", "f2"), object_id="L1"
    )
    e2 = Encoding(
        "P1", np.array([3.0, 4.0], dtype=np.float32), p1, names=("f1", "f2"), object_id="L1"
    )
    check_fusable([e1, e2])

    p_ptv = Provenance(
        encoder="e", encoder_version="1", modality="CT", scope="lesion", roi_role="ptv", roi=h
    )
    e_ptv = Encoding(
        "P1", np.array([3.0, 4.0], dtype=np.float32), p_ptv, names=("f1", "f2"), object_id="L1"
    )
    with pytest.raises(FusionError, match="multiple region roles"):
        check_fusable([e1, e_ptv])

    with pytest.raises(ValueError, match="at least 2 unique patients"):
        gate_encodings_by_icc(
            [e1], pd.DataFrame({"f1": [1.0, 1.1], "f2": [2.0, 2.1]}), ["P1", "P1"]
        )

    rng = np.random.default_rng(42)
    vols = np.exp(rng.normal(5.0, 1.0, size=30))
    df = pd.DataFrame(
        {
            "volume": vols,
            "size_confounded": np.log(vols) * 2.0 + rng.normal(0, 0.05, size=30),
            "independent": rng.normal(0, 1.0, size=30),
        }
    )
    screen = volume_correlation_screen(df, "volume", max_abs_r=0.5)
    by_feat = screen.set_index("feature")
    assert not bool(by_feat.loc["size_confounded", "passes_gate"])
    assert bool(by_feat.loc["independent", "passes_gate"])


def test_longitudinal_in_memory_rasterize_without_dicom_reread():
    from qradiomics.longitudinal.contours import RoiContours, rasterize

    outer = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]])
    inner = np.array([[3.0, 3.0, 0.0], [7.0, 3.0, 0.0], [7.0, 7.0, 0.0], [3.0, 7.0, 0.0]])
    roi = RoiContours(name="GTV_donut", slices={0.0: [outer, inner], 2.0: [outer + [0, 0, 2.0]]})
    mask, spacing, _ = rasterize("/nonexistent/rtstruct.dcm", roi, xy_mm=1.0, z_mm=2.0)
    assert spacing == (2.0, 1.0, 1.0)
    assert mask.shape[0] > 2
    assert int(mask.sum()) > 0
