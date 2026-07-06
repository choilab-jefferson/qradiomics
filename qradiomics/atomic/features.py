"""PyRadiomics feature extraction at the atomic layer.

This wraps a single ``(image, mask)`` extraction in one well-typed function.
The wrapper exists so consumers do not import ``radiomics.featureextractor``
directly — all parameter handling, defaults, geometry tolerance, and
``diagnostics_*`` filtering live here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import SimpleITK as sitk

logger = logging.getLogger(__name__)

__all__ = ["extract_features"]

# Geometry tolerance default matches the value JLR used at the call site we
# are replacing — 1e-3 covers SimpleITK vs rt_utils direction mismatches on
# the same DICOM series without masking real misalignment.
_DEFAULT_GEOMETRY_TOLERANCE = 1e-3


def extract_features(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    params_file: Optional[Union[str, Path]] = None,
    label: int = 1,
    geometry_tolerance: float = _DEFAULT_GEOMETRY_TOLERANCE,
    include_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Run PyRadiomics on one (image, mask) pair and return a flat feature dict.

    Args:
        image: Input image as a SimpleITK Image.
        mask: Binary or label mask as a SimpleITK Image. Must be geometrically
            aligned with ``image`` (use :func:`load_image_and_mask` or the
            caller's own resampling).
        params_file: Optional PyRadiomics YAML parameter file. When omitted,
            the extractor explicitly enables *all* image types and *all*
            feature classes (via ``enableAllImageTypes`` /
            ``enableAllFeatures``) — i.e. wavelet/LoG/square/etc., not just
            the ``original_`` set. Pass a params file for a curated subset.
        label: Label value to extract within ``mask``.
        geometry_tolerance: ``geometryTolerance`` setting passed to PyRadiomics.
        include_diagnostics: When False (default), strip keys starting with
            ``diagnostics_`` from the returned dict.

    Returns:
        ``{feature_name: value}`` with all values coerced to native Python
        scalars where possible (numpy scalars are unwrapped via ``.item()``).
    """
    # Import lazily so qradiomics.atomic is importable without PyRadiomics
    # installed — useful for the doc/test layer.
    from radiomics import featureextractor  # noqa: PLC0415

    if params_file is not None:
        extractor = featureextractor.RadiomicsFeatureExtractor(str(params_file))
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllImageTypes()
        extractor.enableAllFeatures()

    extractor.settings["geometryTolerance"] = geometry_tolerance

    raw = extractor.execute(image, mask, label=label)

    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if not include_diagnostics and key.startswith("diagnostics_"):
            continue
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, TypeError):
                pass
        out[key] = value
    return out
