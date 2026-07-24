"""Post-extraction analytics: multi-site harmonization, confounder
residualization, and ICC-based feature-robustness filtering.

These operate on the tidy feature tables produced by ``qr extract`` /
``qr results merge`` and are exposed on the CLI as ``qr results sanitize``.
"""
from qradiomics.analytics.harmonization import (
    combat_harmonize,
    residualize_linear,
)
from qradiomics.analytics.robustness import feature_icc, icc_filter

__all__ = [
    "combat_harmonize",
    "residualize_linear",
    "feature_icc",
    "icc_filter",
]
