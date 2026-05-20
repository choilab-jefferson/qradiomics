"""AB-parity verification harness for ADR-005 migration slices.

When JLR (or any external consumer) swaps a legacy atomic function for
its `qradiomics.atomic.*` counterpart, this module runs both
implementations against a manifest and produces the same artefacts
that the stage-3a verification produced:

    per_case_summary.csv   — case_id, status, n_features, n_diffs, max_abs, max_rel
    feature_diffs.csv      — case_id, feature, legacy, qradiomics, abs, rel

Tolerance defaults to 1e-9 absolute / 1e-9 relative (same as the
checked-in HeartB10 golden test).

Usage from JLR or any consumer:

    from qradiomics.verification import run_ab_sweep, FeatureExtractor

    def legacy_extract(image_path, mask_path):
        # …  return Dict[str, float]
        ...

    def qradiomics_extract(image_path, mask_path):
        from qradiomics.atomic import load_image_and_mask, extract_features
        img, msk = load_image_and_mask(image_path, mask_path)
        return extract_features(img, msk)

    result = run_ab_sweep(
        manifest_csv="cohort_manifest.csv",
        legacy=FeatureExtractor("legacy", legacy_extract),
        candidate=FeatureExtractor("qradiomics", qradiomics_extract),
        out_dir="verification_output/",
    )
    # → writes per_case_summary.csv + feature_diffs.csv
    # → returns SweepResult with totals

See `architecture/JLR_ATOMIC_MIGRATION.md` §6 for the canonical
verification record this harness produces.
"""
from qradiomics.verification.harness import (
    FeatureExtractor,
    SweepResult,
    run_ab_sweep,
    compare_feature_dicts,
)
from qradiomics.verification.image_parity import (
    ImageDiff,
    compare_image_pair,
    compare_image_pair_dict,
)

__all__ = [
    "FeatureExtractor",
    "ImageDiff",
    "SweepResult",
    "compare_feature_dicts",
    "compare_image_pair",
    "compare_image_pair_dict",
    "run_ab_sweep",
]
