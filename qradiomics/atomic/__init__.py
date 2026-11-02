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

from .denoise_subspace import (
    DEFAULT_RHO_GATE,
    DEFAULT_SHRINKAGE_LAMBDA,
    SubspaceProjector,
    apply_subspace_projection,
    compute_noise_scale_mad,
    fit_noise_whitening_and_snr_subspace,
    fit_transform_ctfm_blocks,
    gate_retest_channels,
    mono_pass,
    pca_scores,
    residualize_against_volume,
    select_stable_dims,
    whitener,
)
from .features import extract_features
from .hu_correct import histogram_match_hu
from .io import check_geometry, load_image_and_mask, save_image_and_mask
from .morphology import boundary_band, peritumoral_rim
from .preprocess import preprocess_pair
from .registration import register_pair, resample_to_fixed
from .registration_elastix import (
    elastix_available,
    register_pair_elastix,
    warp_with_elastix_transform,
)
from .registration_plastimatch import (
    plastimatch_available,
    register_pair_plastimatch,
    warp_with_plastimatch_transform,
)

__all__ = [
    "DEFAULT_RHO_GATE",
    "DEFAULT_SHRINKAGE_LAMBDA",
    "SubspaceProjector",
    "apply_subspace_projection",
    "boundary_band",
    "check_geometry",
    "compute_noise_scale_mad",
    "elastix_available",
    "extract_features",
    "fit_noise_whitening_and_snr_subspace",
    "fit_transform_ctfm_blocks",
    "gate_retest_channels",
    "histogram_match_hu",
    "load_image_and_mask",
    "mono_pass",
    "pca_scores",
    "peritumoral_rim",
    "plastimatch_available",
    "preprocess_pair",
    "register_pair",
    "register_pair_elastix",
    "register_pair_plastimatch",
    "resample_to_fixed",
    "residualize_against_volume",
    "save_image_and_mask",
    "select_stable_dims",
    "warp_with_elastix_transform",
    "warp_with_plastimatch_transform",
    "whitener",
]
