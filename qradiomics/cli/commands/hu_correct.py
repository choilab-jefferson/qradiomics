"""qr hu-correct — histogram-match a CBCT (or other CT-like image) to a
reference CT to align HU distributions before radiomics extraction.
"""
from __future__ import annotations

from pathlib import Path

import click

from qradiomics.atomic import histogram_match_hu


@click.command(name="hu-correct")
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="NRRD to correct (typically a CBCT).")
@click.option("--reference", "-r", "reference_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Reference CT NRRD whose histogram serves as the target "
                   "(typically the plan CT for the same patient).")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output NRRD with corrected HU values.")
@click.option("--levels", default=1024, type=int,
              help="Histogram resolution (default 1024).")
@click.option("--match-points", default=7, type=int,
              help="Number of CDF match points (default 7).")
@click.option("--threshold-mean/--no-threshold-mean", default=True,
              help="Threshold-at-mean-intensity (recommended for CBCT).")
def hu_correct(input_path, reference_path, output, levels, match_points,
               threshold_mean):
    """Apply SimpleITK histogram matching from --input to --reference.

    \b
    Example:
        qr hu-correct -i CBCT_w4.nrrd -r planCT.nrrd -o CBCT_w4_corrected.nrrd
    """
    import SimpleITK as sitk

    click.echo(f"[hu-correct] input={input_path} ref={reference_path}")
    cbct = sitk.ReadImage(str(input_path))
    ref = sitk.ReadImage(str(reference_path))
    matched = histogram_match_hu(cbct, ref, num_histogram_levels=levels,
                                 num_match_points=match_points,
                                 threshold_at_mean_intensity=threshold_mean)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(matched, str(output))
    click.echo(f"[hu-correct] wrote → {output} "
               f"(size={matched.GetSize()}, spacing={tuple(round(s,3) for s in matched.GetSpacing())})")
