#!/usr/bin/env python
"""IBSI-1 digital-phantom feature parity check for the PySERA extractor.

Sibling of run_ibsi1_parity.py — same phantom, reference table, and rounding
rules, but runs pysera.process_batch instead of the PyRadiomics-backed
qradiomics.atomic.features.extract_features.

FEATURE_MAP_PYSERA below is VERIFIED against a real `pysera==2.1.5` run (not
an identity-mapping placeholder). Achieved 117/119 mapped tags within
tolerance (98.3%) — see reports/radiomics_qa/rigor/ibsi1_pysera_parity.md.

PySERA mapping notes
---------------------
* Discretisation: pysera defaults to `radiomics_BinSize=25` (fixed bin size).
  To reproduce IBSI config "A" (binWidth=1, one bin per integer grey level)
  we must pass `bin_size=1` explicitly — the default silently collapses the
  6-level phantom into a single bin and produces degenerate (NaN/constant)
  `ih_*` values.
* `apply_preprocessing=False` reproduces "no interpolation/resampling",
  matching pysera's own default.
* Unlike PyRadiomics, pysera's `stat_kurt`/`ih_kurt` are already *excess*
  kurtosis (matches the IBSI convention directly) — no `-3` correction
  needed.
* pysera's GLCM `clust_shade` / `clust_prom` / `info_corr1` (3D_avg) match
  the IBSI reference exactly. PyRadiomics' sibling map intentionally leaves
  these three unmapped (documented sign/normalisation mismatch there) — for
  pysera they are legitimately mappable and included here.
* pysera implements genuine NGLDM (`ngl_*`, IBSI dependence-count matrix),
  unlike PyRadiomics whose `gldm` is a different construct (see
  run_ibsi1_parity.py's PyRadiomics aggregation note). All 17 IBSI `ngl_*_3D`
  tags with a pysera column map cleanly and pass.
  (`ngl_dc_perc_3D` -> pysera column `ngl_dcperc_3D`, naming-only difference.)
* GLRLM run-entropy is named `glrlm_rl_entropy_3D_avg` in pysera vs IBSI tag
  `rlm_rl_entr_3D_avg` (naming-only difference).
* GLSZM: pysera uses `sdlge`/`sdhge`/`ldlge`/`ldhge` where the IBSI tags are
  `szlge`/`szhge`/`lzlge`/`lzhge` (naming-only difference, values match).
* Known deviation (2 failures, NOT config bugs): `cm_inv_diff_norm_3D_avg`
  and `cm_inv_diff_mom_norm_3D_avg` (IDN / IDMN). pysera's implementation
  normalises the (i-j) distance by `(Ng - 1)` in the denominator
  (`gray_level_cooccurrence_matrix_features_extractor.py`,
  `_calc_inv_diff_norm` / `_calc_inv_diff_mom_norm`), whereas the IBSI
  reference formula normalises by `Ng`. This is a genuine pysera
  implementation convention difference, reported honestly rather than
  patched around.
* Not mapped (pysera does not compute them at all under the categories used
  here): GLDZM (`dzm_*` — no `gldzm` category in this pysera version), IVH
  (`ivh_*`), local intensity peak (`loc_peak_*`), and the 2D/2.5D
  aggregations (we only map the `_3D`/`_3D_avg` forms, matching config A's
  "full 3D" intent, same as the PyRadiomics sibling).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ibsi1_parity import (  # noqa: E402
    MIRP_DATA,
    PHANTOM_IMAGE,
    PHANTOM_MASK,
    data_available,
    load_reference,
    _ibsi_round,
)

_ID = lambda v: v  # noqa: E731

# IBSI tag -> (pysera column name, transform). All identity transforms:
# pysera's kurtosis is already excess (unlike PyRadiomics), so no -3 needed.
FEATURE_MAP_PYSERA: Dict[str, Tuple[str, Callable]] = {
    # ---- Morphology --------------------------------------------------
    "morph_volume": ("morph_volume_mesh", _ID),
    "morph_vol_approx": ("morph_volume_count", _ID),
    "morph_area_mesh": ("morph_surface_area", _ID),
    "morph_av": ("morph_sv_ratio", _ID),
    "morph_sphericity": ("morph_sphericity", _ID),
    "morph_diam": ("morph_max_3d_diameter", _ID),
    "morph_pca_maj_axis": ("morph_major_axis_length", _ID),
    "morph_pca_min_axis": ("morph_minor_axis_length", _ID),
    "morph_pca_least_axis": ("morph_least_axis_length", _ID),
    "morph_pca_elongation": ("morph_elongation", _ID),
    "morph_pca_flatness": ("morph_flatness", _ID),
    # ---- First order / statistics (identity: same tag names) ---------
    "stat_mean": ("stat_mean", _ID),
    "stat_var": ("stat_var", _ID),
    "stat_skew": ("stat_skew", _ID),
    "stat_kurt": ("stat_kurt", _ID),
    "stat_median": ("stat_median", _ID),
    "stat_min": ("stat_min", _ID),
    "stat_p10": ("stat_p10", _ID),
    "stat_p90": ("stat_p90", _ID),
    "stat_max": ("stat_max", _ID),
    "stat_iqr": ("stat_iqr", _ID),
    "stat_range": ("stat_range", _ID),
    "stat_mad": ("stat_mad", _ID),
    "stat_rmad": ("stat_rmad", _ID),
    "stat_energy": ("stat_energy", _ID),
    "stat_rms": ("stat_rms", _ID),
    # ---- Intensity histogram (identity: same tag names) ---------------
    "ih_mean": ("ih_mean", _ID),
    "ih_var": ("ih_var", _ID),
    "ih_skew": ("ih_skew", _ID),
    "ih_kurt": ("ih_kurt", _ID),
    "ih_median": ("ih_median", _ID),
    "ih_min": ("ih_min", _ID),
    "ih_p10": ("ih_p10", _ID),
    "ih_p90": ("ih_p90", _ID),
    "ih_max": ("ih_max", _ID),
    "ih_iqr": ("ih_iqr", _ID),
    "ih_range": ("ih_range", _ID),
    "ih_mad": ("ih_mad", _ID),
    "ih_rmad": ("ih_rmad", _ID),
    "ih_entropy": ("ih_entropy", _ID),
    "ih_uniformity": ("ih_uniformity", _ID),
    # ---- GLCM (cm_* -> pysera glcm_*, 3D averaged) --------------------
    "cm_joint_max_3D_avg": ("glcm_joint_max_3D_avg", _ID),
    "cm_joint_avg_3D_avg": ("glcm_joint_avg_3D_avg", _ID),
    "cm_joint_var_3D_avg": ("glcm_joint_var_3D_avg", _ID),
    "cm_joint_entr_3D_avg": ("glcm_joint_entr_3D_avg", _ID),
    "cm_diff_avg_3D_avg": ("glcm_diff_avg_3D_avg", _ID),
    "cm_diff_var_3D_avg": ("glcm_diff_var_3D_avg", _ID),
    "cm_diff_entr_3D_avg": ("glcm_diff_entr_3D_avg", _ID),
    "cm_sum_avg_3D_avg": ("glcm_sum_avg_3D_avg", _ID),
    "cm_sum_entr_3D_avg": ("glcm_sum_entr_3D_avg", _ID),
    "cm_energy_3D_avg": ("glcm_energy_3D_avg", _ID),
    "cm_contrast_3D_avg": ("glcm_contrast_3D_avg", _ID),
    "cm_dissimilarity_3D_avg": ("glcm_dissimilarity_3D_avg", _ID),
    "cm_inv_diff_3D_avg": ("glcm_inv_diff_3D_avg", _ID),
    "cm_inv_diff_norm_3D_avg": ("glcm_inv_diff_norm_3D_avg", _ID),
    "cm_inv_diff_mom_3D_avg": ("glcm_inv_diff_mom_3D_avg", _ID),
    "cm_inv_diff_mom_norm_3D_avg": ("glcm_inv_diff_mom_norm_3D_avg", _ID),
    "cm_inv_var_3D_avg": ("glcm_inv_var_3D_avg", _ID),
    "cm_corr_3D_avg": ("glcm_corr_3D_avg", _ID),
    "cm_auto_corr_3D_avg": ("glcm_auto_corr_3D_avg", _ID),
    "cm_clust_tend_3D_avg": ("glcm_clust_tend_3D_avg", _ID),
    "cm_info_corr2_3D_avg": ("glcm_info_corr2_3D_avg", _ID),
    # Bonus vs. PyRadiomics sibling: pysera matches IBSI here (verified),
    # PyRadiomics' equivalents are intentionally left unmapped there.
    "cm_clust_shade_3D_avg": ("glcm_clust_shade_3D_avg", _ID),
    "cm_clust_prom_3D_avg": ("glcm_clust_prom_3D_avg", _ID),
    "cm_info_corr1_3D_avg": ("glcm_info_corr1_3D_avg", _ID),
    # ---- GLRLM (rlm_* -> pysera glrlm_*, 3D averaged) -----------------
    "rlm_sre_3D_avg": ("glrlm_sre_3D_avg", _ID),
    "rlm_lre_3D_avg": ("glrlm_lre_3D_avg", _ID),
    "rlm_lgre_3D_avg": ("glrlm_lgre_3D_avg", _ID),
    "rlm_hgre_3D_avg": ("glrlm_hgre_3D_avg", _ID),
    "rlm_srlge_3D_avg": ("glrlm_srlge_3D_avg", _ID),
    "rlm_srhge_3D_avg": ("glrlm_srhge_3D_avg", _ID),
    "rlm_lrlge_3D_avg": ("glrlm_lrlge_3D_avg", _ID),
    "rlm_lrhge_3D_avg": ("glrlm_lrhge_3D_avg", _ID),
    "rlm_glnu_3D_avg": ("glrlm_glnu_3D_avg", _ID),
    "rlm_glnu_norm_3D_avg": ("glrlm_glnu_norm_3D_avg", _ID),
    "rlm_rlnu_3D_avg": ("glrlm_rlnu_3D_avg", _ID),
    "rlm_rlnu_norm_3D_avg": ("glrlm_rlnu_norm_3D_avg", _ID),
    "rlm_r_perc_3D_avg": ("glrlm_r_perc_3D_avg", _ID),
    "rlm_gl_var_3D_avg": ("glrlm_gl_var_3D_avg", _ID),
    "rlm_rl_var_3D_avg": ("glrlm_rl_var_3D_avg", _ID),
    "rlm_rl_entr_3D_avg": ("glrlm_rl_entropy_3D_avg", _ID),  # naming-only diff
    # ---- GLSZM (szm_* -> pysera szm_*, mostly identity) ---------------
    "szm_sze_3D": ("szm_sze_3D", _ID),
    "szm_lze_3D": ("szm_lze_3D", _ID),
    "szm_lgze_3D": ("szm_lgze_3D", _ID),
    "szm_hgze_3D": ("szm_hgze_3D", _ID),
    "szm_szlge_3D": ("szm_sdlge_3D", _ID),  # naming-only diff
    "szm_szhge_3D": ("szm_sdhge_3D", _ID),  # naming-only diff
    "szm_lzlge_3D": ("szm_ldlge_3D", _ID),  # naming-only diff
    "szm_lzhge_3D": ("szm_ldhge_3D", _ID),  # naming-only diff
    "szm_glnu_3D": ("szm_glnu_3D", _ID),
    "szm_glnu_norm_3D": ("szm_glnu_norm_3D", _ID),
    "szm_zsnu_3D": ("szm_zsnu_3D", _ID),
    "szm_zsnu_norm_3D": ("szm_zsnu_norm_3D", _ID),
    "szm_z_perc_3D": ("szm_z_perc_3D", _ID),
    "szm_gl_var_3D": ("szm_gl_var_3D", _ID),
    "szm_zs_var_3D": ("szm_zs_var_3D", _ID),
    "szm_zs_entr_3D": ("szm_zs_entr_3D", _ID),
    # ---- NGTDM (ngt_* -> pysera ngtdm_*) -------------------------------
    "ngt_coarseness_3D": ("ngtdm_coarseness_3D", _ID),
    "ngt_contrast_3D": ("ngtdm_contrast_3D", _ID),
    "ngt_busyness_3D": ("ngtdm_busyness_3D", _ID),
    "ngt_complexity_3D": ("ngtdm_complexity_3D", _ID),
    "ngt_strength_3D": ("ngtdm_strength_3D", _ID),
    # ---- NGLDM (ngl_*) -- bonus vs. PyRadiomics sibling: pysera         --
    # implements genuine IBSI NGLDM (PyRadiomics' gldm is a different       -
    # construct and is not mapped there at all). All verified passing.
    "ngl_lde_3D": ("ngl_lde_3D", _ID),
    "ngl_hde_3D": ("ngl_hde_3D", _ID),
    "ngl_lgce_3D": ("ngl_lgce_3D", _ID),
    "ngl_hgce_3D": ("ngl_hgce_3D", _ID),
    "ngl_ldlge_3D": ("ngl_ldlge_3D", _ID),
    "ngl_ldhge_3D": ("ngl_ldhge_3D", _ID),
    "ngl_hdlge_3D": ("ngl_hdlge_3D", _ID),
    "ngl_hdhge_3D": ("ngl_hdhge_3D", _ID),
    "ngl_glnu_3D": ("ngl_glnu_3D", _ID),
    "ngl_glnu_norm_3D": ("ngl_glnu_norm_3D", _ID),
    "ngl_dcnu_3D": ("ngl_dcnu_3D", _ID),
    "ngl_dcnu_norm_3D": ("ngl_dcnu_norm_3D", _ID),
    "ngl_dc_perc_3D": ("ngl_dcperc_3D", _ID),  # naming-only diff
    "ngl_gl_var_3D": ("ngl_gl_var_3D", _ID),
    "ngl_dc_var_3D": ("ngl_dc_var_3D", _ID),
    "ngl_dc_entr_3D": ("ngl_dc_entr_3D", _ID),
    "ngl_dc_energy_3D": ("ngl_dc_energy_3D", _ID),
}


def run_extraction_pysera() -> Dict[str, float]:
    import pysera  # type: ignore

    # pysera writes an xlsx report to output_path as a side effect; use the
    # system tempdir (auto-cleaned) rather than a repo-local path.
    with tempfile.TemporaryDirectory(prefix="pysera_ibsi1_") as output_path:
        result = pysera.process_batch(
            image_input=str(PHANTOM_IMAGE),
            mask_input=str(PHANTOM_MASK),
            output_path=output_path,
            categories="diag,morph,stat,ih,glcm,glrlm,glszm,ngtdm,ngldm",
            dimensions="1st,2d,2_5d,3d",
            bin_size=1,  # match IBSI config A (binWidth=1); pysera default (25)
            # collapses the 6-level phantom into one bin.
            apply_preprocessing=False,  # no interpolation/resampling
            report="warning",
        )
        return result["features_extracted"].iloc[0].to_dict()


def evaluate_pysera() -> dict:
    """Run extraction + comparison; return a structured result dict."""
    feats = run_extraction_pysera()
    ref = load_reference()

    rows = []  # (tag, ours, reference, tolerance, abs_diff, ok)
    for tag, (ref_val, tol) in sorted(ref.items()):
        if tag not in FEATURE_MAP_PYSERA:
            continue
        pysera_name, transform = FEATURE_MAP_PYSERA[tag]
        if pysera_name not in feats:
            continue
        raw = feats[pysera_name]
        if raw is None or raw != raw:  # NaN guard
            continue
        ours_raw = transform(float(raw))
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
    res = evaluate_pysera()
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
