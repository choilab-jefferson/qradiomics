"""Tests for qradiomics.atomic.registration_plastimatch.

Skipped entirely when the `plastimatch` binary is not on PATH. Where it is
present (CI images / dev machines with `apt install plastimatch`), these
exercise the real subprocess -- not mocks -- since the whole point of this
module is the command-file DSL plastimatch actually accepts.
"""
import numpy as np
import pytest
import SimpleITK as sitk

from qradiomics.atomic.registration_plastimatch import (
    plastimatch_available,
    register_pair_plastimatch,
    warp_with_plastimatch_transform,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not plastimatch_available(), reason="plastimatch binary not on PATH"),
]


def _gaussian_blob(center, shape=(50, 50, 50), sigma=4.0):
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return (1000 * np.exp(-dist2 / (2 * sigma**2))).astype(np.float32)


def _blob_centroid(arr, threshold=100.0):
    idx = np.array(np.nonzero(arr > threshold))
    weights = arr[arr > threshold]
    return (idx * weights).sum(axis=1) / weights.sum()


@pytest.fixture
def blob_pair(tmp_path):
    """A fixed blob at (25,25,25) and a moving blob shifted +3,-2,0."""
    fixed_arr = _gaussian_blob((25, 25, 25))
    moving_arr = _gaussian_blob((28, 23, 25))

    fixed = sitk.GetImageFromArray(fixed_arr)
    fixed.SetSpacing((1.0, 1.0, 1.0))
    moving = sitk.GetImageFromArray(moving_arr)
    moving.SetSpacing((1.0, 1.0, 1.0))

    fixed_path = tmp_path / "fixed.nrrd"
    moving_path = tmp_path / "moving.nrrd"
    sitk.WriteImage(fixed, str(fixed_path))
    sitk.WriteImage(moving, str(moving_path))
    return fixed_path, moving_path, fixed_arr, moving_arr


# ~760mm from the world origin -- typical of a real DICOM patient-coordinate
# CT/CBCT origin, and the distance at which plastimatch's default
# world-origin rotation center produces a lever-arm error large enough to
# throw every optimizer sample outside the moving image buffer.
_FAR_ORIGIN = (-300.0, -300.0, -700.0)


@pytest.fixture
def far_origin_blob_pair(tmp_path):
    """Same content/offset as ``blob_pair``, but placed far from (0, 0, 0)."""
    fixed_arr = _gaussian_blob((30, 30, 30), shape=(60, 60, 60))
    moving_arr = _gaussian_blob((33, 27, 30), shape=(60, 60, 60))

    fixed = sitk.GetImageFromArray(fixed_arr)
    fixed.SetSpacing((1.0, 1.0, 1.0))
    fixed.SetOrigin(_FAR_ORIGIN)
    moving = sitk.GetImageFromArray(moving_arr)
    moving.SetSpacing((1.0, 1.0, 1.0))
    moving.SetOrigin(_FAR_ORIGIN)

    fixed_path = tmp_path / "fixed.nrrd"
    moving_path = tmp_path / "moving.nrrd"
    sitk.WriteImage(fixed, str(fixed_path))
    sitk.WriteImage(moving, str(moving_path))
    return fixed_path, moving_path, fixed, moving


def _point_registration_error(transform_path, fixed_image, moving_image):
    """TRE-style check: map the fixed blob's known location through the
    saved transform and measure the residual distance to the moving blob's
    known location -- exercises the *saved file*, not just in-process state,
    so it also catches the rotation-center serialization bug this backend
    works around (see registration_plastimatch module docstring).
    """
    tx = sitk.ReadTransform(str(transform_path))
    fixed_point = fixed_image.TransformContinuousIndexToPhysicalPoint((30.0, 30.0, 30.0))
    moving_point = np.array(
        moving_image.TransformContinuousIndexToPhysicalPoint((30.0, 27.0, 33.0))
    )
    mapped = np.array(tx.TransformPoint(fixed_point))
    return float(np.linalg.norm(mapped - moving_point))


class TestRegisterPairPlastimatch:
    def test_rigid_stage_aligns_translated_blob(self, tmp_path, blob_pair):
        fixed_path, moving_path, fixed_arr, _ = blob_pair
        output_image = tmp_path / "warped.nrrd"
        output_transform = tmp_path / "transform.txt"

        # mse converges reliably for identical-intensity synthetic images;
        # the default 'mi' metric is tuned for real CT/CBCT texture instead.
        result = register_pair_plastimatch(
            fixed_path,
            moving_path,
            output_image=output_image,
            output_transform=output_transform,
            stages=("rigid",),
            stage_overrides={"rigid": {"metric": "mse"}},
        )

        assert result.warped_image_path == output_image
        assert output_image.exists()
        assert output_transform.exists()

        warped_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(output_image)))
        got = _blob_centroid(warped_arr)
        want = _blob_centroid(fixed_arr)
        assert np.allclose(got, want, atol=0.5), f"centroid {got} not close to fixed {want}"

    def test_unknown_stage_raises(self, blob_pair, tmp_path):
        fixed_path, moving_path, _, _ = blob_pair
        with pytest.raises(ValueError, match="Unknown plastimatch stage"):
            register_pair_plastimatch(
                fixed_path,
                moving_path,
                output_image=tmp_path / "out.nrrd",
                output_transform=tmp_path / "xf.txt",
                stages=("not-a-real-stage",),
            )

    @pytest.mark.slow
    def test_full_deformable_pipeline_runs(self, tmp_path, blob_pair):
        """rigid -> affine -> bspline: the default stage progression that
        makes this backend worth having (SimpleITK backend is linear-only).
        """
        fixed_path, moving_path, _, _ = blob_pair
        output_image = tmp_path / "warped.nrrd"
        output_transform = tmp_path / "transform.txt"

        result = register_pair_plastimatch(
            fixed_path,
            moving_path,
            output_image=output_image,
            output_transform=output_transform,
        )
        assert result.stages == ("rigid", "affine", "bspline")
        assert output_image.exists()
        assert output_transform.exists()


class TestAutoCenterFarFromWorldOrigin:
    """Regression coverage for the world-origin rotation-center trap.

    plastimatch's rigid/affine stages rotate about (0, 0, 0) unless told
    otherwise; real DICOM CT/CBCT origins sit hundreds of mm to over a
    meter away, which turns a normal optimizer step into a huge lever-arm
    displacement and produces a *silent* failure (exit 0, identity
    transform). ``auto_center=True`` (the default) works around this by
    registering in a translated near-origin frame and correcting the
    result back -- these tests exercise exactly that path, not just the
    near-origin synthetic images the rest of this file uses.
    """

    def test_auto_center_recovers_accurate_transform(self, tmp_path, far_origin_blob_pair):
        fixed_path, moving_path, fixed_image, moving_image = far_origin_blob_pair
        output_image = tmp_path / "warped.nrrd"
        output_transform = tmp_path / "transform.txt"

        register_pair_plastimatch(
            fixed_path,
            moving_path,
            output_image=output_image,
            output_transform=output_transform,
            stages=("rigid",),
            stage_overrides={"rigid": {"metric": "mse"}},
        )

        # output_image must be back in the ORIGINAL (far-from-origin) frame.
        assert sitk.ReadImage(str(output_image)).GetOrigin() == fixed_image.GetOrigin()

        # The saved transform file, read independently and applied to a
        # known-correspondence point, must be accurate to sub-mm -- this is
        # the check that would have caught the FixedParameters-collapse bug.
        error_mm = _point_registration_error(output_transform, fixed_image, moving_image)
        assert error_mm < 0.1, f"saved transform residual error {error_mm:.3f}mm is too large"

    def test_auto_center_false_reproduces_silent_failure(self, tmp_path, far_origin_blob_pair):
        fixed_path, moving_path, _, _ = far_origin_blob_pair
        with pytest.raises(RuntimeError, match="silently fell back"):
            register_pair_plastimatch(
                fixed_path,
                moving_path,
                output_image=tmp_path / "warped.nrrd",
                output_transform=tmp_path / "transform.txt",
                stages=("rigid",),
                stage_overrides={"rigid": {"metric": "mse"}},
                auto_center=False,
            )

    @pytest.mark.slow
    def test_auto_center_survives_bspline_stage(self, tmp_path, far_origin_blob_pair):
        """The B-spline transform format has no ITK-readable rotation
        center, so the correction here is a different code path
        (``img_origin`` header patch) from the rigid/affine case above --
        cover it separately rather than trusting the rigid case implies it.
        """
        fixed_path, moving_path, fixed_image, _ = far_origin_blob_pair
        output_image = tmp_path / "warped.nrrd"
        output_transform = tmp_path / "transform.txt"

        result = register_pair_plastimatch(
            fixed_path, moving_path, output_image=output_image, output_transform=output_transform
        )
        assert result.stages == ("rigid", "affine", "bspline")
        assert sitk.ReadImage(str(output_image)).GetOrigin() == fixed_image.GetOrigin()

        header = output_transform.read_text()
        origin_line = next(line for line in header.splitlines() if line.startswith("img_origin"))
        recovered_origin = tuple(float(v) for v in origin_line.split("=")[1].split())
        assert recovered_origin == pytest.approx(fixed_image.GetOrigin(), abs=1e-3)


class TestWarpWithPlastimatchTransform:
    def test_mask_stays_binary_after_warp(self, tmp_path, blob_pair):
        fixed_path, moving_path, _, moving_arr = blob_pair
        mask_arr = (moving_arr > 300).astype(np.uint8)
        mask = sitk.GetImageFromArray(mask_arr)
        mask.SetSpacing((1.0, 1.0, 1.0))
        mask_path = tmp_path / "mask.nrrd"
        sitk.WriteImage(mask, str(mask_path))

        output_image = tmp_path / "warped.nrrd"
        output_transform = tmp_path / "transform.txt"
        register_pair_plastimatch(
            fixed_path,
            moving_path,
            output_image=output_image,
            output_transform=output_transform,
            stages=("rigid",),
            stage_overrides={"rigid": {"metric": "mse"}},
        )

        output_mask = tmp_path / "warped_mask.nrrd"
        warp_with_plastimatch_transform(
            mask_path,
            fixed_path,
            output_transform,
            output_mask,
            interpolation="nn",
        )

        warped_mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(output_mask)))
        assert set(np.unique(warped_mask_arr).tolist()) <= {0, 1}
        # Should still be roughly the same size as the original label.
        assert abs(int(warped_mask_arr.sum()) - int(mask_arr.sum())) < 0.2 * mask_arr.sum()

    def test_bad_interpolation_raises(self, tmp_path, blob_pair):
        fixed_path, moving_path, _, _ = blob_pair
        with pytest.raises(ValueError, match="interpolation"):
            warp_with_plastimatch_transform(
                moving_path,
                fixed_path,
                tmp_path / "nonexistent.txt",
                tmp_path / "out.nrrd",
                interpolation="bogus",
            )
