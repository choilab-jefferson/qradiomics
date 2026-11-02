"""Post-extraction analytics: multi-site harmonization, confounder
residualization, and ICC-based feature-robustness filtering.

These operate on the tidy feature tables produced by ``qr extract`` /
``qr results merge`` and are exposed on the CLI as ``qr results sanitize``.
"""
from qradiomics.analytics.fusion import fuse_feature_tables
from qradiomics.analytics.harmonization import (
    combat_harmonize,
    residualize_linear,
)
from qradiomics.analytics.robustness import (
    feature_icc,
    icc_filter,
    volume_correlation_screen,
)

__all__ = [
    "fuse_feature_tables",
    "combat_harmonize",
    "residualize_linear",
    "feature_icc",
    "icc_filter",
    "volume_correlation_screen",
]
