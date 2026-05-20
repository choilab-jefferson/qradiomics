"""qradiomics.delta — longitudinal-feature differencing & trend fitting.

A delta-radiomics feature is the per-feature difference between two
timepoints for the same (patient, ROI). A trend-radiomics feature is
the per-feature slope across all timepoints for the same (patient,
ROI). Both operate at the CSV / DataFrame layer — they consume the
output of :func:`qradiomics.atomic.extract_features` aggregated into a
longitudinal table.
"""

from .compute import (
    DeltaPair,
    compute_delta,
    compute_trend,
)

__all__ = [
    "DeltaPair",
    "compute_delta",
    "compute_trend",
]
