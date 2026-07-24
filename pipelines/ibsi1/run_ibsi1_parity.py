#!/usr/bin/env python
"""IBSI-1 digital-phantom feature parity check for the qradiomics extractor.

Extracts features from the IBSI-1 chapter-2 digital phantom using the IBSI
config "A" settings (no interpolation / resegmentation / normalization, fixed
integer discretisation, full 3D), maps PyRadiomics feature names onto the IBSI
feature tags in the reference table, and reports per-feature agreement against
the published reference values + tolerances.

This is the engine behind both the standalone report
(reports/radiomics_qa/rigor/ibsi1_phantom_parity.md) and the regression test
(tests/test_ibsi1_phantom_parity.py).

PyRadiomics aggregation note
----------------------------
PyRadiomics computes GLCM/GLRLM as the mean over the 13 symmetric 3D direction
matrices -> this corresponds to the IBSI "3D, averaged" aggregation (tag suffix
``_3D_avg``). GLSZM / GLDZM / NGTDM / NGLDM are single 3D matrices -> IBSI
``_3D``. We therefore map texture features onto those IBSI variants only.

Known PyRadiomics / IBSI deviations (reported, not forced to pass):
  * firstorder_Kurtosis is the *non-excess* kurtosis; IBSI stat_kurt is excess
    kurtosis (PyRadiomics value - 3). We apply the -3 correction in the map.
  * GLCM ClusterShade / ClusterProminence / Imc1 differ from IBSI by known
    normalisation/sign conventions and are intentionally left unmapped.
  * GLDM (PyRadiomics) is NOT the IBSI NGLDM; alpha/coarseness conventions
    differ, so gldm_* features are not mapped to ngl_* tags.
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths (phantom lives outside the repo; override via env for portability).
# ---------------------------------------------------------------------------
MIRP_DATA = Path(
    os.environ.get("IBSI1_MIRP_DATA", str(Path.home() / "gitRepos/mirp/test/data"))
)
PHANTOM_IMAGE = MIRP_DATA / "ibsi_1_digital_phantom/nifti/image/phantom.nii.gz"
PHANTOM_MASK = MIRP_DATA / "ibsi_1_digital_phantom/nifti/mask/mask.nii.gz"
REFERENCE_CSV = MIRP_DATA / "ibsi_1_reference_values/ibsi_1_dig_phantom.csv"

PARAMS_FILE = Path(__file__).with_name("ibsi1_config_a.yaml")


def data_available() -> bool:
    return PHANTOM_IMAGE.is_file() and PHANTOM_MASK.is_file() and REFERENCE_CSV.is_file()


# ---------------------------------------------------------------------------
# IBSI tag  ->  (PyRadiomics feature name, transform).
# transform maps the raw PyRadiomics value onto the IBSI definition.
# ---------------------------------------------------------------------------
_ID = lambda v: v  # noqa: E731

# kurtosis: PyRadiomics reports non-excess; IBSI wants excess (== -3).
_EXCESS_KURT = lambda v: v - 3.0  # noqa: E731

FEATURE_MAP: Dict[str, Tuple[str, callable]] = {
    # ---- Morphology -------------------------------------------------------
    "morph_volume": ("original_shape_MeshVolume", _ID),
    "morph_vol_approx": ("original_shape_VoxelVolume", _ID),
    "morph_area_mesh": ("original_shape_SurfaceArea", _ID),
    "morph_av": ("original_shape_SurfaceVolumeRatio", _ID),
    "morph_sphericity": ("original_shape_Sphericity", _ID),
    "morph_diam": ("original_shape_Maximum3DDiameter", _ID),
    "morph_pca_maj_axis": ("original_shape_MajorAxisLength", _ID),
    "morph_pca_min_axis": ("original_shape_MinorAxisLength", _ID),
    "morph_pca_least_axis": ("original_shape_LeastAxisLength", _ID),
    "morph_pca_elongation": ("original_shape_Elongation", _ID),
    "morph_pca_flatness": ("original_shape_Flatness", _ID),
    # ---- First order / statistics ----------------------------------------
    "stat_mean": ("original_firstorder_Mean", _ID),
    "stat_var": ("original_firstorder_Variance", _ID),
    "stat_skew": ("original_firstorder_Skewness", _ID),
    "stat_kurt": ("original_firstorder_Kurtosis", _EXCESS_KURT),
    "stat_median": ("original_firstorder_Median", _ID),
    "stat_min": ("original_firstorder_Minimum", _ID),
    "stat_p10": ("original_firstorder_10Percentile", _ID),
    "stat_p90": ("original_firstorder_90Percentile", _ID),
    "stat_max": ("original_firstorder_Maximum", _ID),
    "stat_iqr": ("original_firstorder_InterquartileRange", _ID),
    "stat_range": ("original_firstorder_Range", _ID),
    "stat_mad": ("original_firstorder_MeanAbsoluteDeviation", _ID),
    "stat_rmad": ("original_firstorder_RobustMeanAbsoluteDeviation", _ID),
    "stat_energy": ("original_firstorder_Energy", _ID),
    "stat_rms": ("original_firstorder_RootMeanSquared", _ID),
    # ---- Intensity histogram (IBSI ih_*; phantom FBN=raw so same as stat) -
    "ih_mean": ("original_firstorder_Mean", _ID),
    "ih_var": ("original_firstorder_Variance", _ID),
    "ih_skew": ("original_firstorder_Skewness", _ID),
    "ih_kurt": ("original_firstorder_Kurtosis", _EXCESS_KURT),
    "ih_median": ("original_firstorder_Median", _ID),
    "ih_min": ("original_firstorder_Minimum", _ID),
    "ih_p10": ("original_firstorder_10Percentile", _ID),
    "ih_p90": ("original_firstorder_90Percentile", _ID),
    "ih_max": ("original_firstorder_Maximum", _ID),
    "ih_iqr": ("original_firstorder_InterquartileRange", _ID),
    "ih_range": ("original_firstorder_Range", _ID),
    "ih_mad": ("original_firstorder_MeanAbsoluteDeviation", _ID),
    "ih_rmad": ("original_firstorder_RobustMeanAbsoluteDeviation", _ID),
    "ih_entropy": ("original_firstorder_Entropy", _ID),
    "ih_uniformity": ("original_firstorder_Uniformity", _ID),
    # ---- GLCM (PyRadiomics = 3D averaged -> IBSI _3D_avg) -----------------
    "cm_joint_max_3D_avg": ("original_glcm_MaximumProbability", _ID),
    "cm_joint_avg_3D_avg": ("original_glcm_JointAverage", _ID),
    "cm_joint_var_3D_avg": ("original_glcm_SumSquares", _ID),
    "cm_joint_entr_3D_avg": ("original_glcm_JointEntropy", _ID),
    "cm_diff_avg_3D_avg": ("original_glcm_DifferenceAverage", _ID),
    "cm_diff_var_3D_avg": ("original_glcm_DifferenceVariance", _ID),
    "cm_diff_entr_3D_avg": ("original_glcm_DifferenceEntropy", _ID),
    "cm_sum_avg_3D_avg": ("original_glcm_SumAverage", _ID),
    "cm_sum_entr_3D_avg": ("original_glcm_SumEntropy", _ID),
    "cm_energy_3D_avg": ("original_glcm_JointEnergy", _ID),
    "cm_contrast_3D_avg": ("original_glcm_Contrast", _ID),
    "cm_dissimilarity_3D_avg": ("original_glcm_DifferenceAverage", _ID),
    "cm_inv_diff_3D_avg": ("original_glcm_Id", _ID),
    "cm_inv_diff_norm_3D_avg": ("original_glcm_Idn", _ID),
    "cm_inv_diff_mom_3D_avg": ("original_glcm_Idm", _ID),
    "cm_inv_diff_mom_norm_3D_avg": ("original_glcm_Idmn", _ID),
    "cm_inv_var_3D_avg": ("original_glcm_InverseVariance", _ID),
    "cm_corr_3D_avg": ("original_glcm_Correlation", _ID),
    "cm_auto_corr_3D_avg": ("original_glcm_Autocorrelation", _ID),
    "cm_clust_tend_3D_avg": ("original_glcm_ClusterTendency", _ID),
    "cm_info_corr2_3D_avg": ("original_glcm_Imc2", _ID),
    # ---- GLRLM (PyRadiomics = 3D averaged -> IBSI _3D_avg) ----------------
    "rlm_sre_3D_avg": ("original_glrlm_ShortRunEmphasis", _ID),
    "rlm_lre_3D_avg": ("original_glrlm_LongRunEmphasis", _ID),
    "rlm_lgre_3D_avg": ("original_glrlm_LowGrayLevelRunEmphasis", _ID),
    "rlm_hgre_3D_avg": ("original_glrlm_HighGrayLevelRunEmphasis", _ID),
    "rlm_srlge_3D_avg": ("original_glrlm_ShortRunLowGrayLevelEmphasis", _ID),
    "rlm_srhge_3D_avg": ("original_glrlm_ShortRunHighGrayLevelEmphasis", _ID),
    "rlm_lrlge_3D_avg": ("original_glrlm_LongRunLowGrayLevelEmphasis", _ID),
    "rlm_lrhge_3D_avg": ("original_glrlm_LongRunHighGrayLevelEmphasis", _ID),
    "rlm_glnu_3D_avg": ("original_glrlm_GrayLevelNonUniformity", _ID),
    "rlm_glnu_norm_3D_avg": ("original_glrlm_GrayLevelNonUniformityNormalized", _ID),
    "rlm_rlnu_3D_avg": ("original_glrlm_RunLengthNonUniformity", _ID),
    "rlm_rlnu_norm_3D_avg": ("original_glrlm_RunLengthNonUniformityNormalized", _ID),
    "rlm_r_perc_3D_avg": ("original_glrlm_RunPercentage", _ID),
    "rlm_gl_var_3D_avg": ("original_glrlm_GrayLevelVariance", _ID),
    "rlm_rl_var_3D_avg": ("original_glrlm_RunVariance", _ID),
    "rlm_rl_entr_3D_avg": ("original_glrlm_RunEntropy", _ID),
    # ---- GLSZM (single 3D matrix -> IBSI _3D) -----------------------------
    "szm_sze_3D": ("original_glszm_SmallAreaEmphasis", _ID),
    "szm_lze_3D": ("original_glszm_LargeAreaEmphasis", _ID),
    "szm_lgze_3D": ("original_glszm_LowGrayLevelZoneEmphasis", _ID),
    "szm_hgze_3D": ("original_glszm_HighGrayLevelZoneEmphasis", _ID),
    "szm_szlge_3D": ("original_glszm_SmallAreaLowGrayLevelEmphasis", _ID),
    "szm_szhge_3D": ("original_glszm_SmallAreaHighGrayLevelEmphasis", _ID),
    "szm_lzlge_3D": ("original_glszm_LargeAreaLowGrayLevelEmphasis", _ID),
    "szm_lzhge_3D": ("original_glszm_LargeAreaHighGrayLevelEmphasis", _ID),
    "szm_glnu_3D": ("original_glszm_GrayLevelNonUniformity", _ID),
    "szm_glnu_norm_3D": ("original_glszm_GrayLevelNonUniformityNormalized", _ID),
    "szm_zsnu_3D": ("original_glszm_SizeZoneNonUniformity", _ID),
    "szm_zsnu_norm_3D": ("original_glszm_SizeZoneNonUniformityNormalized", _ID),
    "szm_z_perc_3D": ("original_glszm_ZonePercentage", _ID),
    "szm_gl_var_3D": ("original_glszm_GrayLevelVariance", _ID),
    "szm_zs_var_3D": ("original_glszm_ZoneVariance", _ID),
    "szm_zs_entr_3D": ("original_glszm_ZoneEntropy", _ID),
    # ---- NGTDM (single 3D matrix -> IBSI _3D) -----------------------------
    "ngt_coarseness_3D": ("original_ngtdm_Coarseness", _ID),
    "ngt_contrast_3D": ("original_ngtdm_Contrast", _ID),
    "ngt_busyness_3D": ("original_ngtdm_Busyness", _ID),
    "ngt_complexity_3D": ("original_ngtdm_Complexity", _ID),
    "ngt_strength_3D": ("original_ngtdm_Strength", _ID),
}


def load_reference() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return {tag: (reference_value, tolerance)} from the IBSI CSV.

    Rows with empty value/tolerance (features IBSI leaves blank, e.g. ombb/mvee)
    are skipped.
    """
    ref: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    with REFERENCE_CSV.open() as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader)  # header
        for row in reader:
            if len(row) < 3:
                continue
            value_s, tol_s, tag = row[0].strip(), row[1].strip(), row[2].strip()
            if value_s == "" or tol_s == "":
                continue
            ref[tag] = (float(value_s), float(tol_s))
    return ref


def _ibsi_round(x: float) -> float:
    """Round to 3 significant figures, matching the IBSI test harness."""
    if x == 0.0:
        return 0.0
    return round(x, 3 - int(math.floor(math.log10(abs(x)))) - 1)


def run_extraction() -> Dict[str, float]:
    import SimpleITK as sitk

    from qradiomics.atomic.features import extract_features

    image = sitk.ReadImage(str(PHANTOM_IMAGE))
    mask = sitk.ReadImage(str(PHANTOM_MASK))
    return extract_features(image, mask, params_file=PARAMS_FILE, label=1)


def evaluate() -> dict:
    """Run extraction + comparison; return a structured result dict."""
    feats = run_extraction()
    ref = load_reference()

    rows = []  # (tag, ours, reference, tolerance, abs_diff, ok)
    for tag, (ref_val, tol) in sorted(ref.items()):
        if tag not in FEATURE_MAP:
            continue
        pr_name, transform = FEATURE_MAP[tag]
        if pr_name not in feats:
            continue
        ours_raw = transform(float(feats[pr_name]))
        ours = _ibsi_round(ours_raw)
        abs_diff = abs(ours - ref_val)
        ok = abs_diff <= tol
        rows.append((tag, ours, ref_val, tol, abs_diff, ok))

    n_mapped = len(rows)
    n_pass = sum(1 for r in rows if r[5])
    return {
        "n_reference_tags": len(ref),
        "n_mapped": n_mapped,
        "n_pass": n_pass,
        "pct_within_tolerance": (100.0 * n_pass / n_mapped) if n_mapped else 0.0,
        "rows": rows,
    }


def main() -> None:
    if not data_available():
        raise SystemExit(
            f"IBSI phantom data not found under {MIRP_DATA}. "
            "Set IBSI1_MIRP_DATA to override."
        )
    res = evaluate()
    print(f"reference tags : {res['n_reference_tags']}")
    print(f"mapped         : {res['n_mapped']}")
    print(
        f"within tol     : {res['n_pass']}/{res['n_mapped']} "
        f"({res['pct_within_tolerance']:.1f}%)"
    )
    fails = [r for r in res["rows"] if not r[5]]
    if fails:
        print("\nfailures (tag, ours, ref, tol, abs_diff):")
        for tag, ours, ref_val, tol, ad, _ in fails:
            print(f"  {tag:34s} {ours:>12.5g} {ref_val:>12.5g} {tol:>10.5g} {ad:>12.5g}")
    else:
        print("\nno failures within mapped subset")


if __name__ == "__main__":
    main()
