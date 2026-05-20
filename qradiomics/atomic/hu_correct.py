"""CBCT HU correction via histogram matching.

CBCT volumes have inconsistent Hounsfield-unit calibration relative to
the planning CT they accompany (different reconstruction kernels,
scatter correction settings, beam hardening). For radiomics features
that depend on absolute HU values to compare across timepoints, we
typically re-map a CBCT's intensity distribution to match the planning
CT before extraction.

This module wraps SimpleITK's ``HistogramMatchingImageFilter``. The
defaults (200 histogram levels, 50 match points, mean intensity
threshold on) match the long-standing CBCT-radiomics literature
defaults — see e.g. *Radiother. Oncol.* 2019 145:43–51.

Callers who need a more sophisticated correction (e.g. piecewise linear
calibration, per-region patches) should compose their own pipeline.
"""

from __future__ import annotations

import SimpleITK as sitk

__all__ = ["histogram_match_hu"]


def histogram_match_hu(
    source: sitk.Image,
    reference: sitk.Image,
    *,
    number_of_histogram_levels: int = 200,
    number_of_match_points: int = 50,
    threshold_at_mean_intensity: bool = True,
) -> sitk.Image:
    """Map ``source``'s intensity distribution onto ``reference``'s.

    Args:
        source: Image whose intensities should be re-calibrated
            (typically a CBCT scan).
        reference: Image whose intensity distribution is the target
            (typically the matched planning CT).
        number_of_histogram_levels: Number of bins used when building
            the two cumulative distributions.
        number_of_match_points: Number of equispaced control points used
            to fit the piecewise transform — higher = finer match,
            but with diminishing returns past ~50.
        threshold_at_mean_intensity: When True (default), background
            voxels below the mean intensity are excluded from the
            histogram. Required to keep air/table voxels from dominating
            the histogram of a thoracic CBCT.

    Returns:
        Histogram-matched copy of ``source``.
    """
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(int(number_of_histogram_levels))
    matcher.SetNumberOfMatchPoints(int(number_of_match_points))
    matcher.SetThresholdAtMeanIntensity(bool(threshold_at_mean_intensity))
    # The filter requires inputs in the same scalar type.
    src = sitk.Cast(source, sitk.sitkFloat32)
    ref = sitk.Cast(reference, sitk.sitkFloat32)
    matched = matcher.Execute(src, ref)
    if matched.GetPixelID() != source.GetPixelID():
        matched = sitk.Cast(matched, source.GetPixelID())
    return matched
