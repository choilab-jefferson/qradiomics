"""Tests for qradiomics.atomic.registration_elastix.

Skipped entirely when `itk-elastix` is not installed (it is an optional
extra, `pip install qradiomics[elastix]`). Unlike the plastimatch tests,
these have NOT been exercised against a live itk-elastix install in this
repo's dev environment -- they follow the documented itk-elastix API but
should be treated as unverified until run somewhere the extra is present.
"""
import numpy as np
import pytest
import SimpleITK as sitk

from qradiomics.atomic.registration_elastix import (
    elastix_available,
    register_pair_elastix,
    warp_with_elastix_transform,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not elastix_available(), reason="itk-elastix not installed"),
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


class TestRegisterPairElastix:
    def test_rigid_stage_aligns_translated_blob(self, tmp_path, blob_pair):
        fixed_path, moving_path, fixed_arr, _ = blob_pair
        output_image = tmp_path / "warped.mha"
        output_transform = tmp_path / "transform.txt"

        result = register_pair_elastix(
            fixed_path,
            moving_path,
            output_image=output_image,
            output_transform=output_transform,
            stages=("rigid",),
        )

        assert result.warped_image_path == output_image
        assert output_image.exists()
        assert len(result.transform_paths) == 1
        assert result.transform_paths[0].exists()

        warped_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(output_image)))
        got = _blob_centroid(warped_arr)
        want = _blob_centroid(fixed_arr)
        assert np.allclose(got, want, atol=1.0), f"centroid {got} not close to fixed {want}"

    def test_unknown_stage_raises(self, blob_pair, tmp_path):
        fixed_path, moving_path, _, _ = blob_pair
        with pytest.raises(ValueError, match="Unknown elastix stage"):
            register_pair_elastix(
                fixed_path,
                moving_path,
                output_image=tmp_path / "out.mha",
                output_transform=tmp_path / "xf.txt",
                stages=("not-a-real-stage",),
            )


class TestWarpWithElastixTransform:
    def test_mask_stays_binary_after_warp(self, tmp_path, blob_pair):
        fixed_path, moving_path, _, moving_arr = blob_pair
        mask_arr = (moving_arr > 300).astype(np.uint8)
        mask = sitk.GetImageFromArray(mask_arr)
        mask.SetSpacing((1.0, 1.0, 1.0))
        mask_path = tmp_path / "mask.mha"
        sitk.WriteImage(mask, str(mask_path))

        output_image = tmp_path / "warped.mha"
        output_transform = tmp_path / "transform.txt"
        result = register_pair_elastix(
            fixed_path,
            moving_path,
            output_image=output_image,
            output_transform=output_transform,
            stages=("rigid",),
        )

        output_mask = tmp_path / "warped_mask.mha"
        warp_with_elastix_transform(
            mask_path,
            result.transform_parameter_object,
            output_mask,
            is_mask=True,
        )

        warped_mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(output_mask)))
        assert set(np.unique(warped_mask_arr).tolist()) <= {0, 1}
