"""qradiomics.atomic — single-image, single-mask radiomics primitives.

This package owns the atomic operations of the radiomics workflow. Consumers
(JeffLungRadiomics longitudinal pipelines, the qradiomics CLI, downstream
research scripts) call these functions instead of touching PyRadiomics or
SimpleITK directly. The contract is intentionally minimal:

* ``load_image_and_mask(image_path, mask_path)`` — read two NRRD/NIfTI/DICOM
  inputs into SimpleITK images and check that their geometry is compatible.
* ``extract_features(image, mask, params_file=None)`` — run a configured
  PyRadiomics extractor on one (image, mask) pair and return a flat dict of
  feature name → value with ``diagnostics_*`` keys stripped.

These two operations together cover the leaf of the workflow tree described
in [[wiki/architecture/ATOMIC_UNIT_AND_STAGES]]. Anything above this layer
(per-course iteration, per-cohort manifest, cross-timepoint aggregation)
belongs in the workflow / pipeline modules, not here.
"""

from .features import extract_features
from .hu_correct import histogram_match_hu
from .io import check_geometry, load_image_and_mask
from .preprocess import preprocess_pair
from .registration import register_pair, resample_to_fixed

__all__ = [
    "check_geometry",
    "extract_features",
    "histogram_match_hu",
    "load_image_and_mask",
    "preprocess_pair",
    "register_pair",
    "resample_to_fixed",
]
