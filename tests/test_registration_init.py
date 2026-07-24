"""Regression test for the affine transform initializer in atomic.registration.

The affine branch used to call ``affine.SetCenter(centred.GetParameters()[3:6])``,
which for an Euler3DTransform is the *translation*, not the rotation centre — so
the affine started from an identity matrix with an arbitrary centre and the
initializer's centre-of-geometry alignment was discarded.
"""
import numpy as np
import SimpleITK as sitk

from qradiomics.atomic.registration import _init_transform


def _blob(origin):
    arr = np.zeros((10, 10, 10), np.float32)
    arr[3:7, 3:7, 3:7] = 1.0
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(origin)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def test_affine_seeded_from_centered_initializer():
    fixed = _blob((0.0, 0.0, 0.0))
    moving = _blob((5.0, 2.0, 1.0))
    euler = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    aff = _init_transform(fixed, moving, "affine")

    # Rotation centre must be the geometry centre (GetFixedParameters), and the
    # centre-of-geometry alignment (the Euler translation) must be carried over.
    assert tuple(aff.GetCenter()) == tuple(euler.GetFixedParameters()[:3])
    assert tuple(aff.GetTranslation()) == tuple(euler.GetParameters()[3:6])
    # The old bug put the translation into the centre; guard against regressing.
    assert tuple(aff.GetCenter()) != tuple(euler.GetParameters()[3:6])
