"""Tests for qradiomics.atomic — single (image, mask) pair operations.

Signatures (JLR-aligned, mirrored from qradiomics-dev):

  load_image_and_mask(image_path, mask_path, *,
                      require_compatible_geometry=True,
                      spacing_tolerance=1e-3, origin_tolerance=1e-3)
    → (sitk.Image, sitk.Image)

  check_geometry(image, mask, *, spacing_tolerance, origin_tolerance) → None
    (raises ValueError on mismatch)

  preprocess_pair(image, mask, *, pad_mm, resample_mm: Sequence[float]|None,
                  image_interpolator, mask_interpolator, label)
    → (sitk.Image, sitk.Image)

  extract_features(image, mask, *, params_file=None, label=1,
                   geometry_tolerance=1e-3, include_diagnostics=False)
    → dict[str, Any]

  register_pair(fixed, moving, *, transform="rigid", metric="mattes", …)
    → (sitk.Transform, sitk.Image)

  resample_to_fixed(fixed, moving, transform=None, *, interpolator, default_pixel_value)
    → sitk.Image
"""
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from qradiomics.atomic import (
    check_geometry,
    extract_features,
    histogram_match_hu,
    load_image_and_mask,
    preprocess_pair,
    register_pair,
    resample_to_fixed,
)


def _sphere_image_and_mask(size=(40, 40, 40), radius=8.0,
                            spacing=(1.0, 1.0, 1.0)):
    zc, yc, xc = [s / 2 for s in size]
    zz, yy, xx = np.meshgrid(
        np.arange(size[2]) - zc,
        np.arange(size[1]) - yc,
        np.arange(size[0]) - xc,
        indexing="ij",
    )
    dist2 = (xx * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (zz * spacing[2]) ** 2
    intensity = np.where(dist2 < radius ** 2, 100.0, -200.0).astype(np.float32)
    mask_arr = (dist2 < radius ** 2).astype(np.uint8)
    img = sitk.GetImageFromArray(intensity); img.SetSpacing(spacing)
    msk = sitk.GetImageFromArray(mask_arr); msk.SetSpacing(spacing)
    return img, msk


# ─── io ─────────────────────────────────────────────────────────────────────

class TestLoadAndGeometry:
    def test_round_trip(self, tmp_path):
        img, msk = _sphere_image_and_mask()
        img_p = tmp_path / "img.nrrd"
        msk_p = tmp_path / "msk.nrrd"
        sitk.WriteImage(img, str(img_p))
        sitk.WriteImage(msk, str(msk_p))
        img2, msk2 = load_image_and_mask(img_p, msk_p)
        # Should not raise — geometry compatible by construction
        check_geometry(img2, msk2)

    def test_missing_image_raises(self, tmp_path):
        msk_path = tmp_path / "msk.nrrd"
        msk_path.touch()
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            load_image_and_mask(tmp_path / "missing.nrrd", msk_path)

    def test_check_geometry_raises_on_mismatch(self):
        img, _ = _sphere_image_and_mask()
        msk2 = sitk.Image((50, 50, 50), sitk.sitkUInt8)
        with pytest.raises(ValueError):
            check_geometry(img, msk2)


# ─── preprocess ─────────────────────────────────────────────────────────────

class TestPreprocessPair:
    def test_crop_produces_smaller_or_equal_volume(self):
        img, msk = _sphere_image_and_mask()
        img_c, msk_c = preprocess_pair(img, msk, pad_mm=5.0, resample_mm=None)
        assert img_c.GetSize() == msk_c.GetSize()
        assert any(a < b for a, b in zip(img_c.GetSize(), img.GetSize())) \
            or img_c.GetSize() == img.GetSize()  # tiny phantom may not actually shrink

    def test_resample_tuple(self):
        """JLR call site passes resample_mm as (sx, sy, sz)."""
        img, msk = _sphere_image_and_mask(spacing=(0.5, 0.5, 2.0))
        img_r, msk_r = preprocess_pair(
            img, msk, pad_mm=5.0, resample_mm=(1.0, 1.0, 1.0)
        )
        assert all(abs(s - 1.0) < 1e-6 for s in img_r.GetSpacing())
        vals = set(np.unique(sitk.GetArrayFromImage(msk_r)))
        assert vals.issubset({0, 1})

    def test_empty_mask_raises(self):
        img, _ = _sphere_image_and_mask()
        empty = sitk.Image(img.GetSize(), sitk.sitkUInt8)
        empty.CopyInformation(img)
        with pytest.raises(ValueError):
            preprocess_pair(img, empty, pad_mm=5.0)

    def test_size_mismatch_raises(self):
        img, _ = _sphere_image_and_mask()
        smaller_mask = sitk.Image((20, 20, 20), sitk.sitkUInt8)
        smaller_mask[:] = 1
        with pytest.raises(ValueError):
            preprocess_pair(img, smaller_mask, pad_mm=5.0)


# ─── feature extraction ────────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_basic_extraction(self):
        img, msk = _sphere_image_and_mask()
        feats = extract_features(img, msk)
        assert len(feats) > 0
        assert not any(k.startswith("diagnostics_") for k in feats)

    def test_deterministic(self):
        img, msk = _sphere_image_and_mask()
        f1 = extract_features(img, msk)
        f2 = extract_features(img, msk)
        for k in f1:
            if isinstance(f1[k], (int, float)):
                assert abs(float(f1[k]) - float(f2[k])) < 1e-12

    def test_include_diagnostics_flag(self):
        img, msk = _sphere_image_and_mask()
        with_diag = extract_features(img, msk, include_diagnostics=True)
        without_diag = extract_features(img, msk, include_diagnostics=False)
        assert any(k.startswith("diagnostics_") for k in with_diag)
        assert not any(k.startswith("diagnostics_") for k in without_diag)


# ─── registration ───────────────────────────────────────────────────────────

class TestRegistration:
    def test_self_register_returns_transform_and_resampled(self):
        img, _ = _sphere_image_and_mask()
        transform, resampled = register_pair(img, img)
        assert isinstance(transform, sitk.Transform)
        # Self-pair: resampled should approximate the input
        a = sitk.GetArrayFromImage(img)
        b = sitk.GetArrayFromImage(resampled)
        assert np.mean(np.abs(a - b)) < 5.0

    def test_resample_to_fixed_signature(self):
        """resample_to_fixed(fixed, moving, transform=None) - new arg order."""
        img, _ = _sphere_image_and_mask()
        # Identity transform: identity mapping
        identity = sitk.Euler3DTransform()
        warped = resample_to_fixed(img, img, identity)
        assert warped.GetSize() == img.GetSize()


# ─── HU correction ──────────────────────────────────────────────────────────

class TestHUCorrection:
    def test_self_match_preserves_mean(self):
        img, _ = _sphere_image_and_mask()
        matched = histogram_match_hu(img, img)
        ob = sitk.GetArrayFromImage(img).mean()
        mb = sitk.GetArrayFromImage(matched).mean()
        assert abs(mb - ob) < 0.5

    def test_shift_corrected_toward_reference(self):
        ref, _ = _sphere_image_and_mask()
        shifted = sitk.ShiftScale(ref, shift=200.0, scale=1.0)
        ref_mean = sitk.GetArrayFromImage(ref).mean()
        sh_mean = sitk.GetArrayFromImage(shifted).mean()
        matched = histogram_match_hu(shifted, ref)
        m_mean = sitk.GetArrayFromImage(matched).mean()
        assert abs(m_mean - ref_mean) < abs(sh_mean - ref_mean)


# ─── integration with real HeartCB cohort (skip if absent) ─────────────────

_HEARTCB = Path("/data/users/wxc151/HeartToxicity/HeartCB/HeartB10")


@pytest.mark.skipif(
    not _HEARTCB.exists(),
    reason="HeartCB cohort fixtures absent",
)
class TestIntegrationHeartCB:
    img_p = _HEARTCB / "HeartB10_planCT_cropped-1mm.nrrd"
    msk_p = _HEARTCB / "HeartB10_planCT_manual_cropped-1mm-label.nrrd"

    def test_load_and_extract(self):
        img, msk = load_image_and_mask(self.img_p, self.msk_p)
        feats = extract_features(img, msk)
        assert len(feats) > 100
