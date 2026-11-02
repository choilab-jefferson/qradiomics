"""qr register — register a moving image (+ optional mask) onto a fixed image.

Used for Scenario C: a mask defined on `fixed` cannot be used on `moving`
directly because they live in different coordinate frames. This command
estimates a transform that maps moving->fixed, writes the resampled moving
image (+ optional mask) into the fixed frame, and saves the transform for
inspection or re-use.

Three backends are available via ``--backend``:

* ``sitk`` (default) — in-process SimpleITK rigid/affine registration
  (:func:`qradiomics.atomic.register_pair`). No external dependency, fast,
  linear transforms only.
* ``plastimatch`` — shells out to the ``plastimatch`` binary. Supports
  deformable (B-spline) registration via ``--stages rigid,affine,bspline``
  (the default for this backend). Requires ``plastimatch`` on PATH.
* ``elastix`` — uses ``itk-elastix`` in-process. Also supports deformable
  registration via the same ``--stages`` convention. Requires
  ``pip install qradiomics[elastix]``.
"""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

import click

_PLASTIMATCH_DEFAULT_STAGES = "rigid,affine,bspline"
_ELASTIX_DEFAULT_STAGES = "rigid,affine,bspline"


@contextmanager
def _transform_output_path(output_transform, output_image):
    """Where to write the (plastimatch/elastix-mandatory) transform file.

    If the user asked for it via --output-transform, write it there and
    keep it. Otherwise use a scratch temp path that is cleaned up on exit --
    matching the sitk backend, which leaves nothing behind when
    --output-transform is omitted, instead of silently littering the output
    directory with a file nobody asked for.
    """
    if output_transform:
        yield Path(output_transform)
    else:
        with tempfile.TemporaryDirectory(prefix="qr_register_xform_") as tmp_dir:
            yield Path(tmp_dir) / (Path(output_image).stem + "_xform.txt")


@click.command()
@click.option(
    "--fixed",
    "fixed_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Reference NRRD (mask is authored on this frame).",
)
@click.option(
    "--moving",
    "moving_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="NRRD to register into the fixed frame.",
)
@click.option(
    "--mask",
    "mask_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="(optional) NRRD label in the MOVING frame to resample "
    "into the fixed frame using the same transform.",
)
@click.option(
    "--output-image",
    required=True,
    type=click.Path(),
    help="Output path for the resampled moving image.",
)
@click.option(
    "--output-mask",
    default=None,
    type=click.Path(),
    help="(optional) Output path for the resampled mask (requires --mask).",
)
@click.option(
    "--output-transform",
    default=None,
    type=click.Path(),
    help="(optional) Save the estimated transform. Format depends "
    "on --backend: sitk writes a .tfm; plastimatch writes its "
    "native xform (linear or B-spline coefficients); elastix "
    "writes one TransformParameters.<i>.txt per stage.",
)
@click.option(
    "--backend",
    type=click.Choice(["sitk", "plastimatch", "elastix"]),
    default="sitk",
    show_default=True,
    help="Registration engine. 'sitk' = built-in rigid/affine, no "
    "external dependency. 'plastimatch'/'elastix' = external "
    "tools, support deformable (B-spline) registration via "
    "--stages.",
)
@click.option(
    "--stages",
    default=None,
    help="plastimatch/elastix only: comma-separated stage list, "
    "e.g. 'rigid,affine,bspline' (default) or 'rigid' to stop "
    "before the deformable stage.",
)
@click.option(
    "--transform-type",
    "transform_kind",
    type=click.Choice(["rigid", "affine"]),
    default="rigid",
    show_default=True,
    help="sitk backend only: linear transform model.",
)
@click.option(
    "--max-iterations", default=200, type=int, show_default=True, help="sitk backend only."
)
@click.option(
    "--sampling-fraction",
    default=0.20,
    type=float,
    show_default=True,
    help="sitk backend only: fraction of voxels sampled for the metric.",
)
def register(
    fixed_path,
    moving_path,
    mask_path,
    output_image,
    output_mask,
    output_transform,
    backend,
    stages,
    transform_kind,
    max_iterations,
    sampling_fraction,
):
    """Register MOVING -> FIXED.

    \b
    Example (Scenario C, planCT -> CBCT week4, default sitk rigid backend):
        qr register \\
          --fixed CBCT_week4.nrrd \\
          --moving planCT.nrrd \\
          --mask  planCT_Heart-label.nrrd \\
          --output-image planCT_in_w4.nrrd \\
          --output-mask  Heart_in_w4-label.nrrd

    \b
    Example (deformable registration via plastimatch):
        qr register \\
          --fixed CBCT_week4.nrrd --moving planCT.nrrd \\
          --output-image planCT_in_w4.nrrd \\
          --backend plastimatch --stages rigid,affine,bspline
    """
    click.echo(f"[register] backend={backend} fixed={fixed_path} moving={moving_path}")

    if mask_path and not output_mask:
        raise click.UsageError("--mask requires --output-mask")

    if backend == "sitk":
        _register_sitk(
            fixed_path,
            moving_path,
            mask_path,
            output_image,
            output_mask,
            output_transform,
            transform_kind,
            max_iterations,
            sampling_fraction,
        )
    elif backend == "plastimatch":
        stage_list = tuple((stages or _PLASTIMATCH_DEFAULT_STAGES).split(","))
        _register_plastimatch(
            fixed_path,
            moving_path,
            mask_path,
            output_image,
            output_mask,
            output_transform,
            stage_list,
        )
    else:
        stage_list = tuple((stages or _ELASTIX_DEFAULT_STAGES).split(","))
        _register_elastix(
            fixed_path,
            moving_path,
            mask_path,
            output_image,
            output_mask,
            output_transform,
            stage_list,
        )


def _register_sitk(
    fixed_path,
    moving_path,
    mask_path,
    output_image,
    output_mask,
    output_transform,
    transform_kind,
    max_iterations,
    sampling_fraction,
):
    import SimpleITK as sitk

    from qradiomics.atomic import register_pair, resample_to_fixed

    fixed = sitk.ReadImage(str(fixed_path))
    moving = sitk.ReadImage(str(moving_path))
    transform, warped = register_pair(
        fixed,
        moving,
        transform=transform_kind,
        number_of_iterations=max_iterations,
        sampling_percentage=sampling_fraction,
    )
    click.echo(f"[register] sitk registration complete (transform={transform_kind})")

    Path(output_image).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(warped, str(output_image))
    click.echo(f"[register] wrote resampled image -> {output_image}")

    if mask_path:
        moving_mask = sitk.ReadImage(str(mask_path))
        warped_mask = resample_to_fixed(
            fixed, moving_mask, transform, interpolator=sitk.sitkNearestNeighbor
        )
        Path(output_mask).parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(warped_mask, str(output_mask))
        click.echo(f"[register] wrote resampled mask -> {output_mask}")

    if output_transform:
        Path(output_transform).parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(transform, str(output_transform))
        click.echo(f"[register] wrote transform -> {output_transform}")


def _register_plastimatch(
    fixed_path, moving_path, mask_path, output_image, output_mask, output_transform, stage_list
):
    from qradiomics.atomic import register_pair_plastimatch, warp_with_plastimatch_transform

    with _transform_output_path(output_transform, output_image) as xf_path:
        result = register_pair_plastimatch(
            fixed_path,
            moving_path,
            output_image=output_image,
            output_transform=xf_path,
            stages=stage_list,
        )
        click.echo(f"[register] plastimatch stages={result.stages} -> {output_image}")

        if mask_path:
            warp_with_plastimatch_transform(
                mask_path, fixed_path, result.transform_path, output_mask, interpolation="nn"
            )
            click.echo(f"[register] wrote resampled mask -> {output_mask}")

        if output_transform:
            click.echo(f"[register] wrote transform -> {output_transform}")


def _register_elastix(
    fixed_path, moving_path, mask_path, output_image, output_mask, output_transform, stage_list
):
    from qradiomics.atomic import register_pair_elastix, warp_with_elastix_transform

    with _transform_output_path(output_transform, output_image) as xf_base:
        result = register_pair_elastix(
            fixed_path,
            moving_path,
            output_image=output_image,
            output_transform=xf_base,
            stages=stage_list,
        )
        click.echo(f"[register] elastix stages={result.stages} -> {output_image}")

        if mask_path:
            warp_with_elastix_transform(
                mask_path, result.transform_parameter_object, output_mask, is_mask=True
            )
            click.echo(f"[register] wrote resampled mask -> {output_mask}")

        if output_transform:
            click.echo(
                f"[register] wrote transform(s) -> "
                f"{', '.join(str(p) for p in result.transform_paths)}"
            )
