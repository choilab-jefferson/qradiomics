"""Deformable/rigid image registration via itk-elastix.

elastix (https://elastix.lumc.nl) is one of the most widely used and
validated open-source deformable image registration toolkits in medical
imaging research. This module wraps the maintained ``itk-elastix`` Python
bindings (the modern successor to the old, now-unmaintained SimpleElastix
fork) instead of shelling out to a standalone ``elastix`` binary -- no
system package is required beyond ``pip install qradiomics[elastix]``.

Like :mod:`qradiomics.atomic.registration_plastimatch`, this backend reads
fixed/moving from file paths and writes the warped image to a file path (to
keep the same calling convention across external-tool backends), but keeps
the registration result in memory for the optional mask-warp step --
elastix's ``TransformParameterObject`` can be reused directly in-process
without round-tripping through the on-disk ``TransformParameters.N.txt``
files it also writes for inspection/reuse.

``itk`` / ``itk-elastix`` is intentionally NOT imported at module load time
-- only inside the functions below -- so importing this module (or
``qradiomics.atomic``) never requires the extra to be installed.

Note: unlike the plastimatch backend, this module has not been exercised
against a live registration run in this environment (itk-elastix is not
installed here). It follows itk-elastix's documented public API
(``itk.ParameterObject``, ``itk.elastix_registration_method``,
``itk.transformix_filter``); treat it as unverified until smoke-tested
somewhere the extra is actually installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union

__all__ = [
    "ElastixResult",
    "elastix_available",
    "register_pair_elastix",
    "warp_with_elastix_transform",
]

_DEFAULT_STAGES: Tuple[str, ...] = ("rigid", "affine", "bspline")
_VALID_STAGES = {"translation", "rigid", "affine", "bspline"}


def elastix_available() -> bool:
    """True if ``itk-elastix`` (not just plain ``itk``) is importable."""
    try:
        import itk
    except ImportError:
        return False
    return hasattr(itk, "elastix_registration_method")


def _require_elastix():
    try:
        import itk
    except ImportError as exc:
        raise RuntimeError(
            "itk-elastix is not installed. Run `pip install "
            "qradiomics[elastix]` (or `pip install itk-elastix` directly) "
            "to use backend='elastix'."
        ) from exc
    if not hasattr(itk, "elastix_registration_method"):
        raise RuntimeError(
            "The installed `itk` package does not include elastix "
            "support. Install `itk-elastix` (not plain `itk`) to use "
            "backend='elastix'."
        )
    return itk


@dataclass
class ElastixResult:
    """Outcome of :func:`register_pair_elastix`."""

    warped_image_path: Path
    transform_paths: Tuple[Path, ...]
    transform_parameter_object: object
    stages: Tuple[str, ...]


def register_pair_elastix(
    fixed_path: Union[str, Path],
    moving_path: Union[str, Path],
    *,
    output_image: Union[str, Path],
    output_transform: Union[str, Path],
    stages: Sequence[str] = _DEFAULT_STAGES,
) -> ElastixResult:
    """Register ``moving`` onto ``fixed`` with elastix's multi-stage DIR engine.

    Args:
        fixed_path: Reference image (e.g. planning CT).
        moving_path: Image to align (e.g. on-treatment CBCT).
        output_image: Where the warped moving image is written.
        output_transform: Base path for the per-stage elastix transform
            parameter files. One file is written per stage, named
            ``"{stem}.{i}{suffix}"`` (e.g. ``transform.0.txt``,
            ``transform.1.txt``, ...) following elastix's own
            ``TransformParameters.N.txt`` convention.
        stages: Ordered elastix preset names, each one of
            ``"translation"``, ``"rigid"``, ``"affine"``, ``"bspline"``.
            Default ``("rigid", "affine", "bspline")`` mirrors the
            plastimatch backend so the two are drop-in comparable for the
            same command.

    Returns:
        :class:`ElastixResult`. ``transform_parameter_object`` is the live
        in-memory elastix transform -- pass it straight to
        :func:`warp_with_elastix_transform` to carry a mask through the
        same registration without re-reading from disk.

    Raises:
        ValueError: An unknown stage name was passed.
        RuntimeError: ``itk-elastix`` is not installed.
    """
    for stage in stages:
        if stage not in _VALID_STAGES:
            raise ValueError(
                f"Unknown elastix stage {stage!r}; expected one of " f"{sorted(_VALID_STAGES)}"
            )
    itk = _require_elastix()

    fixed_path = Path(fixed_path)
    moving_path = Path(moving_path)
    output_image = Path(output_image)
    output_transform = Path(output_transform)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_transform.parent.mkdir(parents=True, exist_ok=True)

    fixed_image = itk.imread(str(fixed_path), itk.F)
    moving_image = itk.imread(str(moving_path), itk.F)

    parameter_object = itk.ParameterObject.New()  # type: ignore[attr-defined]
    for stage in stages:
        parameter_object.AddParameterMap(parameter_object.GetDefaultParameterMap(stage))

    result_image, transform_parameters = itk.elastix_registration_method(  # type: ignore[attr-defined]
        fixed_image, moving_image, parameter_object=parameter_object
    )
    itk.imwrite(result_image, str(output_image))

    suffix = output_transform.suffix or ".txt"
    transform_paths = []
    for i in range(transform_parameters.GetNumberOfParameterMaps()):
        path = output_transform.with_name(f"{output_transform.stem}.{i}{suffix}")
        transform_parameters.WriteParameterFile(transform_parameters.GetParameterMap(i), str(path))
        transform_paths.append(path)

    return ElastixResult(
        warped_image_path=output_image,
        transform_paths=tuple(transform_paths),
        transform_parameter_object=transform_parameters,
        stages=tuple(stages),
    )


def warp_with_elastix_transform(
    input_path: Union[str, Path],
    transform_parameter_object,
    output_path: Union[str, Path],
    *,
    is_mask: bool = False,
) -> Path:
    """Apply an elastix transform (from :func:`register_pair_elastix`) to
    another volume, typically a label mask defined in the moving frame.

    Pass ``transform_parameter_object`` straight from
    ``ElastixResult.transform_parameter_object`` of the same registration --
    it already carries the fixed image's output geometry. For masks, set
    ``is_mask=True`` to force nearest-neighbour interpolation so labels stay
    integer-valued instead of blurring at the boundary.
    """
    itk = _require_elastix()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pixel_type = itk.UC if is_mask else itk.F
    moving = itk.imread(str(input_path), pixel_type)

    params = transform_parameter_object
    if is_mask:
        # Copy so we don't mutate the transform object the caller may reuse
        # for warping the (linearly interpolated) image.
        params = itk.ParameterObject.New()  # type: ignore[attr-defined]
        for i in range(transform_parameter_object.GetNumberOfParameterMaps()):
            params.AddParameterMap(transform_parameter_object.GetParameterMap(i))
        for i in range(params.GetNumberOfParameterMaps()):
            params.SetParameter(i, "ResampleInterpolator", "FinalNearestNeighborInterpolator")
            params.SetParameter(i, "FinalBSplineInterpolationOrder", "0")

    result = itk.transformix_filter(moving, params)  # type: ignore[attr-defined]
    itk.imwrite(result, str(output_path))
    return output_path
