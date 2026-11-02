"""qradiomics.deep — 3D foundation-model pooling and multi-stage compartment decomposition.

Public primitives:
  pooling      — Area-integrated 3D soft-mask pooling, center-of-mass fallback,
                 and small-lesion adaptive zoom-in spacing calculation.
  compartment  — Multi-stage pyramid compartment-decomposed pooling across
                 whole lesion, solid core (HU >= -500), and boundary shell.
"""

from __future__ import annotations

from .compartment import (
    COMPARTMENTS,
    DEFAULT_PYRAMID_CHANNELS,
    DEFAULT_SOLID_HU_THRESHOLD,
    DEFAULT_STAGES,
    LEGACY_COMPARTMENTS,
    STATS,
    align_pyramid_stages,
    build_compartment_masks,
    extract_multistage_compartment_features,
    pool_compartment_stage,
    pool_pyramid_compartments,
)
from .pooling import (
    DEFAULT_CHANNELS,
    DEFAULT_FOCAL_PAD_MM,
    DEFAULT_FOCAL_RESAMPLE_MM,
    DEFAULT_MIN_FOCAL_SPACING_MM,
    DEFAULT_TARGET_ROI_VOXELS,
    aggregate_feature_map,
    centroid_sample_stage,
    compute_adaptive_focal_spacing,
    compute_adaptive_zoom_spacing,
    focal_crop_key,
    project_mask_centroid_to_feature_grid,
    resize_mask_to_feature_grid,
    soft_pool_stage,
)

__all__ = [
    "COMPARTMENTS",
    "DEFAULT_CHANNELS",
    "DEFAULT_FOCAL_PAD_MM",
    "DEFAULT_FOCAL_RESAMPLE_MM",
    "DEFAULT_MIN_FOCAL_SPACING_MM",
    "DEFAULT_PYRAMID_CHANNELS",
    "DEFAULT_SOLID_HU_THRESHOLD",
    "DEFAULT_STAGES",
    "DEFAULT_TARGET_ROI_VOXELS",
    "LEGACY_COMPARTMENTS",
    "STATS",
    "aggregate_feature_map",
    "align_pyramid_stages",
    "build_compartment_masks",
    "centroid_sample_stage",
    "compute_adaptive_focal_spacing",
    "compute_adaptive_zoom_spacing",
    "extract_multistage_compartment_features",
    "focal_crop_key",
    "pool_compartment_stage",
    "pool_pyramid_compartments",
    "project_mask_centroid_to_feature_grid",
    "resize_mask_to_feature_grid",
    "soft_pool_stage",
]
