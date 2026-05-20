"""qr register — rigid registration between two NRRD volumes.

Used for Scenario C: a mask defined on `fixed` cannot be used on
`moving` directly because they live in different coordinate frames. This
command estimates a rigid transform that maps moving→fixed, writes the
resampled moving image (+ optional mask) into the fixed frame, and
saves the transform for inspection or re-use.
"""
from __future__ import annotations

from pathlib import Path

import click

from qradiomics.atomic import register_pair, resample_to_fixed


@click.command()
@click.option("--fixed", "fixed_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Reference NRRD (mask is authored on this frame).")
@click.option("--moving", "moving_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="NRRD to register into the fixed frame.")
@click.option("--mask", "mask_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="(optional) NRRD label in the MOVING frame to resample "
                   "into the fixed frame using the same transform.")
@click.option("--output-image", required=True, type=click.Path(),
              help="Output path for the resampled moving image.")
@click.option("--output-mask", default=None, type=click.Path(),
              help="(optional) Output path for the resampled mask (requires --mask).")
@click.option("--output-transform", default=None, type=click.Path(),
              help="(optional) Save the SimpleITK transform (.tfm).")
@click.option("--max-iterations", default=200, type=int)
@click.option("--sampling-fraction", default=0.25, type=float)
def register(fixed_path, moving_path, mask_path, output_image, output_mask,
             output_transform, max_iterations, sampling_fraction):
    """Rigid register MOVING → FIXED via Mattes MI + LBFGSB.

    \b
    Example (Scenario C, planCT → CBCT week4):
        qr register \\
          --fixed CBCT_week4.nrrd \\
          --moving planCT.nrrd \\
          --mask  planCT_Heart-label.nrrd \\
          --output-image planCT_in_w4.nrrd \\
          --output-mask  Heart_in_w4-label.nrrd
    """
    import SimpleITK as sitk

    click.echo(f"[register] fixed={fixed_path} moving={moving_path}")
    fixed = sitk.ReadImage(str(fixed_path))
    moving = sitk.ReadImage(str(moving_path))
    result = register_pair(fixed, moving,
                           max_iterations=max_iterations,
                           sampling_fraction=sampling_fraction)
    click.echo(f"[register] converged={result.converged} "
               f"iterations={result.iterations} metric={result.final_metric:.4f}")

    warped = resample_to_fixed(moving, fixed, result.transform, is_mask=False)
    Path(output_image).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(warped, str(output_image))
    click.echo(f"[register] wrote resampled image → {output_image}")

    if mask_path:
        if not output_mask:
            raise click.UsageError("--mask requires --output-mask")
        moving_mask = sitk.ReadImage(str(mask_path))
        warped_mask = resample_to_fixed(moving_mask, fixed, result.transform,
                                         is_mask=True)
        Path(output_mask).parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(warped_mask, str(output_mask))
        click.echo(f"[register] wrote resampled mask → {output_mask}")

    if output_transform:
        Path(output_transform).parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(result.transform, str(output_transform))
        click.echo(f"[register] wrote transform → {output_transform}")
