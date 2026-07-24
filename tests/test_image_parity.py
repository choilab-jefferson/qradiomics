"""Tests for qradiomics.verification.image_parity — for Stage 3b sweeps."""
from pathlib import Path

import pytest
import SimpleITK as sitk

from qradiomics.verification import compare_image_pair, compare_image_pair_dict


def _make_image(size=(16, 16, 16), value=42.0, spacing=(1.0, 1.0, 1.0)):
    img = sitk.Image(size, sitk.sitkFloat32)
    img.SetSpacing(spacing)
    img[:] = value
    return img


class TestCompareImagePair:
    def test_identical_images_zero_diff(self):
        a = _make_image()
        b = sitk.Image(a)  # deep copy
        d = compare_image_pair(a, b)
        assert d.is_identical
        assert d.pixel_max_abs == 0.0

    def test_size_mismatch_short_circuits(self):
        a = _make_image(size=(10, 10, 10))
        b = _make_image(size=(12, 10, 10))
        d = compare_image_pair(a, b)
        assert d.size_diff is True
        assert d.pixel_total == 0  # not computed when size differs

    def test_spacing_diff_captured(self):
        a = _make_image()
        b = _make_image(spacing=(1.0, 1.0, 2.0))
        d = compare_image_pair(a, b)
        assert d.spacing_max_abs == 1.0

    def test_pixel_diff_above_tol(self):
        a = _make_image(value=10.0)
        b = _make_image(value=10.0 + 1e-6)
        d = compare_image_pair(a, b, pixel_tol=1e-9)
        assert d.pixel_max_abs > 1e-9
        assert d.pixel_n_above_tol == d.pixel_total

    def test_pixel_diff_below_tol_passes(self):
        a = _make_image(value=10.0)
        b = _make_image(value=10.0 + 1e-12)
        d = compare_image_pair(a, b, pixel_tol=1e-9)
        assert d.pixel_n_above_tol == 0
        assert d.is_identical

    def test_pair_dict_image_and_mask(self):
        a_img, b_img = _make_image(value=5.0), _make_image(value=5.0)
        a_msk = sitk.Image(a_img.GetSize(), sitk.sitkUInt8); a_msk[:] = 1
        b_msk = sitk.Image(a_img.GetSize(), sitk.sitkUInt8); b_msk[:] = 1
        diffs = compare_image_pair_dict((a_img, a_msk), (b_img, b_msk))
        assert diffs["image"].is_identical
        assert diffs["mask"].is_identical


_HEARTCB = Path("/data/datasets/heart-toxicity-heartcb/nrrd")


@pytest.mark.skipif(
    not _HEARTCB.exists(),
    reason="HeartCB cohort absent",
)
class TestPreprocessStress:
    """Stress-test preprocess_pair against every HeartCB planCT/manual
    pair to surface silent geometry regressions on the qradiomics side
    BEFORE JLR runs its Stage 3b AB-parity sweep."""

    def test_full_cohort_geometry_aware(self):
        """preprocess_pair raises on image/mask size mismatch (5 HeartCB cases
        have drifted NRRDs). For the remaining cases the cropped pair
        must share geometry."""
        from qradiomics.atomic import load_image_and_mask, preprocess_pair

        ok = 0
        size_mismatch = 0
        unexpected_fail = []
        for pdir in sorted(_HEARTCB.glob("Heart*")):
            if not pdir.is_dir():
                continue
            pid = pdir.name
            img_p = pdir / f"{pid}_planCT_cropped-1mm.nrrd"
            msk_p = pdir / f"{pid}_planCT_manual_cropped-1mm-label.nrrd"
            if not (img_p.exists() and msk_p.exists()):
                continue
            try:
                img, msk = load_image_and_mask(
                    img_p, msk_p, require_compatible_geometry=False)
                ic, mc = preprocess_pair(img, msk, pad_mm=20.0,
                                          resample_mm=(1.0, 1.0, 1.0))
                assert ic.GetSize() == mc.GetSize()
                assert ic.GetSpacing() == mc.GetSpacing()
                ok += 1
            except ValueError as e:
                if "size" in str(e).lower() or "Image/mask" in str(e):
                    size_mismatch += 1
                else:
                    unexpected_fail.append((pid, str(e)[:80]))
            except Exception as e:
                unexpected_fail.append((pid, str(e)[:80]))

        assert not unexpected_fail, (
            f"preprocess_pair errored unexpectedly for: {unexpected_fail}"
        )
        # The known drifted cases on this cohort total 5.
        assert ok + size_mismatch >= 35


@pytest.mark.skipif(
    not (_HEARTCB / "HeartB10").exists(),
    reason="HeartB10 fixture absent",
)
def test_preprocess_pair_deterministic():
    """Calling preprocess_pair twice with the same inputs yields
    bit-identical output. Required for any future AB-parity sweep."""
    from qradiomics.atomic import load_image_and_mask, preprocess_pair

    pdir = _HEARTCB / "HeartB10"
    img, msk = load_image_and_mask(
        pdir / "HeartB10_planCT_cropped-1mm.nrrd",
        pdir / "HeartB10_planCT_manual_cropped-1mm-label.nrrd",
    )
    a1, b1 = preprocess_pair(img, msk, pad_mm=20.0, resample_mm=(1.0, 1.0, 1.0))
    a2, b2 = preprocess_pair(img, msk, pad_mm=20.0, resample_mm=(1.0, 1.0, 1.0))
    d_img = compare_image_pair(a1, a2)
    d_msk = compare_image_pair(b1, b2)
    assert d_img.is_identical
    assert d_msk.is_identical
