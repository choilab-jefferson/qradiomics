"""Image-to-image registration at the atomic layer.

Two primitives:

* :func:`register_pair` — estimate a rigid (or affine) transform that
  aligns ``moving`` onto ``fixed`` using SimpleITK's
  ``ImageRegistrationMethod``. Returns the transform and the resampled
  moving image. Defaults are chosen for CT/CBCT alignment in a single
  patient's coordinate system — multi-resolution Mattes mutual
  information, regular gradient descent, three-level pyramid.
* :func:`resample_to_fixed` — apply a known transform to bring a moving
  image (and optionally its label mask) onto the fixed image grid.
  Useful when the transform was computed elsewhere (e.g. a clinical TPS
  export) and you only need to warp the data.

These functions wrap SimpleITK and keep no hidden state. Callers who
need deformable / B-spline registration should compose their own
``ImageRegistrationMethod`` pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import SimpleITK as sitk

__all__ = ["register_pair", "resample_to_fixed"]


def register_pair(
    fixed: sitk.Image,
    moving: sitk.Image,
    *,
    transform: str = "rigid",
    metric: str = "mattes",
    sampling_percentage: float = 0.20,
    learning_rate: float = 1.0,
    number_of_iterations: int = 200,
    shrink_factors: Tuple[int, ...] = (4, 2, 1),
    smoothing_sigmas: Tuple[float, ...] = (2.0, 1.0, 0.0),
    interpolator: int = sitk.sitkLinear,
    default_pixel_value: float = 0.0,
    seed: int = 12345,
) -> Tuple[sitk.Transform, sitk.Image]:
    """Register ``moving`` onto ``fixed`` and return ``(transform, warped_moving)``.

    Args:
        fixed: Reference image (e.g. planning CT).
        moving: Image to align (e.g. on-treatment CBCT).
        transform: ``"rigid"`` (default — 3D Euler) or ``"affine"``.
        metric: ``"mattes"`` (Mattes mutual information, default) or
            ``"correlation"`` (cross-correlation, faster, intramodal).
        sampling_percentage: Fraction of voxels sampled for the metric.
            Lower = faster but noisier; default 20% matches the
            SimpleITK examples for CT/CBCT.
        learning_rate: Step size for gradient descent.
        number_of_iterations: Iteration cap per resolution level.
        shrink_factors: Pyramid downsampling per level (length must equal
            ``smoothing_sigmas``).
        smoothing_sigmas: Gaussian σ (in voxels) per level.
        interpolator: SimpleITK interpolator enum for the final resample.
        default_pixel_value: Background value in the warped output.
        seed: RNG seed for the metric sampling — keep fixed for
            reproducibility.

    Returns:
        ``(transform, warped_moving)``. The transform maps **fixed →
        moving** (SimpleITK convention), which is what
        ``sitk.Resample`` expects.
    """
    if len(shrink_factors) != len(smoothing_sigmas):
        raise ValueError(
            f"shrink_factors and smoothing_sigmas must have equal length, "
            f"got {len(shrink_factors)} vs {len(smoothing_sigmas)}"
        )

    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    initial_transform = _init_transform(fixed_f, moving_f, transform)

    method = sitk.ImageRegistrationMethod()
    if metric == "mattes":
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric == "correlation":
        method.SetMetricAsCorrelation()
    else:
        raise ValueError(f"Unknown metric {metric!r}; expected 'mattes' or 'correlation'.")
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(float(sampling_percentage), seed=seed)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate),
        minStep=1e-4,
        numberOfIterations=int(number_of_iterations),
        gradientMagnitudeTolerance=1e-8,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(list(shrink_factors))
    method.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = method.Execute(fixed_f, moving_f)
    warped = sitk.Resample(
        moving,
        fixed,
        final_transform,
        interpolator,
        default_pixel_value,
        moving.GetPixelID(),
    )
    return final_transform, warped


def resample_to_fixed(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform: Optional[Union[sitk.Transform, str]] = None,
    *,
    interpolator: int = sitk.sitkLinear,
    default_pixel_value: float = 0.0,
) -> sitk.Image:
    """Resample ``moving`` onto the ``fixed`` grid.

    If ``transform`` is None, an identity transform is used (i.e. the
    images are assumed to already share a coordinate system). Passing a
    ``sitk.Transform`` warps via that transform — typically the output
    of :func:`register_pair` or a transform loaded from disk.
    """
    if transform is None:
        tx: sitk.Transform = sitk.Transform(fixed.GetDimension(), sitk.sitkIdentity)
    elif isinstance(transform, str):
        tx = sitk.ReadTransform(transform)
    else:
        tx = transform
    return sitk.Resample(
        moving, fixed, tx, interpolator, default_pixel_value, moving.GetPixelID()
    )


def _init_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    kind: str,
) -> sitk.Transform:
    centred = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    if kind == "rigid":
        return centred
    if kind == "affine":
        affine = sitk.AffineTransform(fixed.GetDimension())
        affine.SetCenter(centred.GetParameters()[3:6])
        return affine
    raise ValueError(f"Unknown transform kind {kind!r}; expected 'rigid' or 'affine'.")
