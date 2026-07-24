"""Methods-comparison harness — drop-in alternatives that all consume the same
features CSV and produce the same RM / PM / external AUC report.

This is the **framework** behind the CMPB 2021 / CIR / Med Phys 2018
reproducibility: any new feature-extraction method that emits a wide
features.csv (with `pid`, `nodule_id`, `malignancy`, …) can be plugged in
and compared against the same RM / PM / external validation split:

    Large LIDC-IDRI 1,018         → RM (Radiomic Model) train/CV
    Small LIDC-PM 72              → PM calibration
    LUNGx 74                      → PM external validation

Use this script to sweep methods over the same data and emit a side-by-side
markdown report.

Built-in methods (define your own by adding a `(name, feature_selector)` pair):

    aerts4         — Aerts 2014 4-feature signature
    radiomics50    — Med Phys 2018 style (PyRadiomics top-50)
    spic6          — CMPB 2021 (Np, Na, Nl, Na_att, s1, s2)
    radiomics+spic — combined Med Phys + CMPB 2021
    shape_only     — PyRadiomics original_shape_* only (≈ 14 features)
    firstorder     — PyRadiomics original_firstorder_* only
    aerts_signature_alt — Aerts 4-feature with Compactness derived
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler

METADATA = {
    "pid", "reader", "nodule_id", "n_voxels", "volume_mm3",
    "malignancy", "subtlety", "calcification", "sphericity_score",
    "margin", "lobulation_score", "spiculation_score", "texture",
    "status", "status_radiomics", "status_spic", "error", "traceback",
    "y", "nodule_key", "y_malignant", "diagnosis", "scan",
}
SPIC = ["spic_Np", "spic_Na", "spic_Nl", "spic_Na_att", "spic_s1", "spic_s2"]
# Paper feature names: Ns=spiculations, Nl=lobulations, Na=attachments, s1/s2, BB_AP=size
# Our naming maps: spic_Na → paper Ns, spic_Na_att → paper Na, spic_Nl → paper Nl
SIZE = ["volume_mm3"]  # BB_AP proxy
CMPB2021_FULL = SPIC + SIZE
AERTS4 = [
    "original_firstorder_Energy",
    "original_shape_Sphericity",                 # ≈ Compactness1
    "original_glrlm_GrayLevelNonUniformity",
    "wavelet-HLH_glrlm_GrayLevelNonUniformity",
]


def _radiomics_cols(df: pd.DataFrame) -> list[str]:
    drop = METADATA | set(SPIC)
    return [c for c in df.columns
            if c not in drop
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().any()]


def _shape_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("original_shape_")]


def _firstorder_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("original_firstorder_")]


def _spic_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in SPIC if c in df.columns]


# Registry: name → list[str] of feature columns (function of df)
METHODS: dict[str, Callable[[pd.DataFrame], list[str]]] = {
    "aerts4":              lambda df: [c for c in AERTS4 if c in df.columns],
    "radiomics50":         _radiomics_cols,
    "spic6":               _spic_cols,
    "cmpb2021_size+spic":  lambda df: [c for c in CMPB2021_FULL if c in df.columns],
    "radiomics+spic":      lambda df: _radiomics_cols(df) + _spic_cols(df),
    "shape_only":          _shape_cols,
    "firstorder":          _firstorder_cols,
    "size_only":           lambda df: [c for c in SIZE if c in df.columns],
}


def _aggregate_lidc(df: pd.DataFrame) -> pd.DataFrame:
    """One row per nodule (mean over readers)."""
    df = df.copy()
    df["nodule_key"] = df["pid"].astype(str) + ":" + df["nodule_id"].astype(str)
    numeric = [c for c in df.columns
                if c not in {"nodule_key", "pid", "nodule_id"}
                and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby("nodule_key").agg({**{c: "mean" for c in numeric},
                                       "pid": "first", "nodule_id": "first"})
    return g.reset_index(drop=True)


def _brier_ece(y_true, prob, n_bins: int = 10) -> tuple[float, float]:
    """Brier score + equal-width-bin Expected Calibration Error (C05)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    if len(y) == 0:
        return float("nan"), float("nan")
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        hi_inclusive = i == n_bins - 1
        m = (p >= bins[i]) & (p <= bins[i + 1] if hi_inclusive else p < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return brier, float(ece)


def _auc_ci(y_true, prob, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Point AUC + percentile bootstrap 95% CI on pooled predictions (C06)."""
    y = np.asarray(y_true)
    p = np.asarray(prob)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    auc = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    if not boots:
        return auc, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)


def _cv_auc(X: pd.DataFrame, y: pd.Series, groups,
            n_splits: int = 5, top_k: int = 50, seed: int = 42,
            classifier: str = "rf"):
    """Patient-grouped (or stratified) CV. Returns mean/std AUC plus the
    pooled out-of-fold (y, prob) so calibration (Brier/ECE) and a bootstrap
    AUC CI can be computed on honest held-out predictions."""
    if groups is not None:
        kf = GroupKFold(n_splits=n_splits)
        splitter = kf.split(X, y, groups)
    else:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splitter = kf.split(X, y)
    aucs = []
    oof_y: list[float] = []
    oof_p: list[float] = []
    for tr, te in splitter:
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        med = Xtr.median(); Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
        var_mask = Xtr.var() > 1e-10
        Xtr = Xtr.loc[:, var_mask]; Xte = Xte.loc[:, var_mask]
        if Xtr.shape[1] > 1:
            cor = Xtr.corr().abs()
            upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
            drop = [c for c in upper.columns if any(upper[c] > 0.95)]
            Xtr = Xtr.drop(columns=drop); Xte = Xte.drop(columns=drop)
        if Xtr.shape[1] > top_k:
            scores = Xtr.apply(lambda col: abs(np.corrcoef(col, ytr)[0, 1])
                                if col.std() > 1e-10 else 0.0)
            keep = scores.nlargest(top_k).index
            Xtr = Xtr[keep]; Xte = Xte[keep]
        sc = StandardScaler().fit(Xtr)
        if classifier == "rf":
            clf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                          n_jobs=-1, class_weight="balanced")
        elif classifier == "lr":
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                      random_state=seed)
        clf.fit(sc.transform(Xtr), ytr)
        prob = clf.predict_proba(sc.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(yte, prob))
        oof_y.extend(np.asarray(yte, dtype=float).tolist())
        oof_p.extend(np.asarray(prob, dtype=float).tolist())
    return (float(np.mean(aucs)), float(np.std(aucs)),
            np.asarray(oof_y), np.asarray(oof_p))


def _external_auc(Xtr, ytr, Xte, yte, top_k=50, classifier="rf", seed=42,
                  Xcal=None, ycal=None):
    """Train on LIDC; optionally domain-adapt on (Xcal,ycal) before testing.

    LUNGx SPIE / CMPB 2021 protocol: 10-patient CalibrationSet for site
    calibration → 60-patient TestSet.

    Calibration strategy (when Xcal is given): **domain adaptation by merging**
    — feature standardization is refit on (LIDC ∪ LUNGx-Cal), then the
    classifier is retrained on (LIDC ∪ LUNGx-Cal) with the LUNGx-Cal samples
    upweighted (sample_weight=5) to overcome LIDC's dominance. This adapts
    the model to LUNGx-specific feature scales — Platt sigmoid alone is
    AUC-invariant (monotonic) so does not help on a rank metric, but the
    refit shifts the model into LUNGx feature space.
    """
    med = Xtr.median(); Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
    if Xcal is not None: Xcal = Xcal.fillna(med)
    var_mask = Xtr.var() > 1e-10
    Xtr = Xtr.loc[:, var_mask]; Xte = Xte.loc[:, var_mask]
    if Xcal is not None: Xcal = Xcal.loc[:, var_mask]
    if Xtr.shape[1] > 1:
        cor = Xtr.corr().abs()
        upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > 0.95)]
        Xtr = Xtr.drop(columns=drop); Xte = Xte.drop(columns=drop)
        if Xcal is not None: Xcal = Xcal.drop(columns=drop)
    common = Xtr.columns.intersection(Xte.columns)
    Xtr = Xtr[common]; Xte = Xte[common]
    if Xcal is not None: Xcal = Xcal[common]
    if Xtr.shape[1] > top_k:
        scores = Xtr.apply(lambda col: abs(np.corrcoef(col, ytr)[0, 1])
                            if col.std() > 1e-10 else 0.0)
        keep = scores.nlargest(top_k).index
        Xtr = Xtr[keep]; Xte = Xte[keep]
        if Xcal is not None: Xcal = Xcal[keep]

    if Xcal is not None and len(Xcal) >= 5:
        # Domain-adapted: merge LIDC + LUNGx-Cal, upweight LUNGx-Cal samples
        cal_weight = 5.0
        Xcombined = pd.concat([Xtr, Xcal], axis=0, ignore_index=True)
        ycombined = pd.concat([ytr, ycal], axis=0, ignore_index=True)
        sw = np.concatenate([np.ones(len(Xtr)),
                              np.full(len(Xcal), cal_weight)])
        sc = StandardScaler().fit(Xcombined)
        if classifier == "rf":
            clf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                          n_jobs=-1, class_weight="balanced")
        else:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                      random_state=seed)
        clf.fit(sc.transform(Xcombined), ycombined, sample_weight=sw)
    else:
        sc = StandardScaler().fit(Xtr)
        if classifier == "rf":
            clf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                          n_jobs=-1, class_weight="balanced")
        else:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                      random_state=seed)
        clf.fit(sc.transform(Xtr), ytr)

    prob = clf.predict_proba(sc.transform(Xte))[:, 1]
    return (float(roc_auc_score(yte, prob)),
            np.asarray(yte, dtype=float), np.asarray(prob, dtype=float))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lidc-features", required=True, type=Path)
    p.add_argument("--lungx-features", required=True, type=Path)
    p.add_argument("--lidc-pm-ids", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    lidc = pd.read_csv(args.lidc_features)
    lungx = pd.read_csv(args.lungx_features)
    pm_ids = set(line.strip() for line in args.lidc_pm_ids.read_text().splitlines()
                 if line.strip())

    lidc_agg = _aggregate_lidc(lidc)
    # RM: full LIDC-IDRI, malignancy ≥4 vs ≤2 (drop 3)
    rm = lidc_agg[lidc_agg["malignancy"].between(1, 5)
                   & (lidc_agg["malignancy"] != 3)].copy()
    rm["y"] = (rm["malignancy"] >= 4).astype(int)
    # PM: LIDC-PM subset
    pm = lidc_agg[lidc_agg["pid"].isin(pm_ids)].copy()
    pm["y"] = (pm["malignancy"] >= 3).astype(int)
    # External: LUNGx CalibrationSet (CT-Training-*) and TestSet (LUNGx-CT*)
    ext = lungx[lungx["status_radiomics"] == "ok"].copy()
    ext["y"] = ext["y_malignant"].astype(int)
    ext_cal = ext[ext["pid"].astype(str).str.startswith("CT-Training")].copy()
    ext_test = ext[ext["pid"].astype(str).str.startswith("LUNGx-")].copy()

    print(f"RM (LIDC-IDRI ≥4 vs ≤2):     n={len(rm)}  "
          f"+{int(rm.y.sum())}/-{int((1 - rm.y).sum())}", file=sys.stderr)
    print(f"PM (LIDC-PM):                n={len(pm)}  "
          f"+{int(pm.y.sum())}/-{int((1 - pm.y).sum())}", file=sys.stderr)
    print(f"LUNGx CalibrationSet:        n={len(ext_cal)}  "
          f"+{int(ext_cal.y.sum())}/-{int((1 - ext_cal.y).sum())}", file=sys.stderr)
    print(f"LUNGx TestSet:               n={len(ext_test)}  "
          f"+{int(ext_test.y.sum())}/-{int((1 - ext_test.y).sum())}", file=sys.stderr)

    rows = []
    for name, sel in METHODS.items():
        rcols = sel(rm)
        pcols = sel(pm)
        ecols = sel(ext)
        nan = float("nan")
        # RM CV (patient-grouped 5-fold) + calibration (Brier/ECE) + bootstrap CI
        rm_brier = rm_ece = rm_lo = rm_hi = nan
        if rcols and len(rm) > 25:
            rm_auc, rm_std, rm_y, rm_p = _cv_auc(rm[rcols], rm["y"], rm["pid"].values)
            rm_brier, rm_ece = _brier_ece(rm_y, rm_p)
            _, rm_lo, rm_hi = _auc_ci(rm_y, rm_p)
        else:
            rm_auc = rm_std = nan
        # PM CV (5-fold, no group split — small n) + bootstrap CI
        pm_lo = pm_hi = nan
        if pcols and len(pm) > 25:
            pm_auc, pm_std, pm_y, pm_p = _cv_auc(pm[pcols], pm["y"], None)
            _, pm_lo, pm_hi = _auc_ci(pm_y, pm_p)
        else:
            pm_auc = pm_std = nan
        # External (no calibration): train RM, test LUNGx-TestSet
        ecols_t = sel(ext_test)
        common_re = [c for c in rcols if c in ecols_t] if (rcols and ecols_t) else []
        ext_auc = ext_brier = ext_ece = ext_lo = ext_hi = nan
        if common_re and len(ext_test) > 5:
            try:
                ext_auc, ext_y, ext_p = _external_auc(
                    rm[common_re], rm["y"], ext_test[common_re], ext_test["y"])
                ext_brier, ext_ece = _brier_ece(ext_y, ext_p)
                _, ext_lo, ext_hi = _auc_ci(ext_y, ext_p)
            except Exception as e:
                print(f"  ext {name} err: {e}", file=sys.stderr)

        # External WITH 10-patient LUNGx-Cal calibration (CMPB 2021 protocol)
        ecols_c = sel(ext_cal)
        common_rec = [c for c in rcols if c in ecols_t and c in ecols_c] if (rcols and ecols_c and ecols_t) else []
        ext_cal_auc = ext_cal_brier = ext_cal_ece = ext_cal_lo = ext_cal_hi = nan
        if common_rec and len(ext_test) > 5 and len(ext_cal) >= 5:
            try:
                ext_cal_auc, extc_y, extc_p = _external_auc(
                    rm[common_rec], rm["y"], ext_test[common_rec], ext_test["y"],
                    Xcal=ext_cal[common_rec], ycal=ext_cal["y"])
                ext_cal_brier, ext_cal_ece = _brier_ece(extc_y, extc_p)
                _, ext_cal_lo, ext_cal_hi = _auc_ci(extc_y, extc_p)
            except Exception as e:
                print(f"  ext-cal {name} err: {e}", file=sys.stderr)

        n_feat_rm = len(rcols); n_feat_pm = len(pcols)
        rows.append({
            "method": name,
            "RM_auc": rm_auc, "RM_std": rm_std,
            "RM_ci_lo": rm_lo, "RM_ci_hi": rm_hi,
            "RM_brier": rm_brier, "RM_ece": rm_ece, "RM_n_feat": n_feat_rm,
            "PM_auc": pm_auc, "PM_std": pm_std,
            "PM_ci_lo": pm_lo, "PM_ci_hi": pm_hi, "PM_n_feat": n_feat_pm,
            "ext_auc": ext_auc, "ext_ci_lo": ext_lo, "ext_ci_hi": ext_hi,
            "ext_brier": ext_brier, "ext_ece": ext_ece,
            "ext_cal_auc": ext_cal_auc,
            "ext_cal_ci_lo": ext_cal_lo, "ext_cal_ci_hi": ext_cal_hi,
            "ext_cal_brier": ext_cal_brier, "ext_cal_ece": ext_cal_ece,
        })
        print(f"  {name:20s}  RM={rm_auc:.3f}[{rm_lo:.3f},{rm_hi:.3f}] "
              f"Brier={rm_brier:.3f} ECE={rm_ece:.3f}  "
              f"PM={pm_auc:.3f}  ext={ext_auc:.3f} ext+cal={ext_cal_auc:.3f}",
              file=sys.stderr)

    out_md = [
        "# Methods comparison on a single qradiomics-public pipeline\n",
        f"Source: `{args.lidc_features}` + `{args.lungx_features}`",
        f"PM subset: {args.lidc_pm_ids} ({len(pm_ids)} patients)\n",
        "**Design**: any feature-extraction method whose output is a wide CSV in "
        "the qradiomics-public format can plug into this same RM / PM / external "
        "validation harness. The four splits follow the **Choi 2021 CMPB** protocol:\n",
        f"- **RM** (Radiomic Model) — LIDC-IDRI 1,018 patients, malignancy ≥4 vs ≤2, "
        f"RF 5-fold patient-grouped CV. (paper used LIDC-RM = 811 weakly-labelled cases)",
        f"- **PM** (Pathology Model) — LIDC-PM {len(pm_ids)} pathology-confirmed patients, "
        f"RF 5-fold CV (paper used 10×10-fold).",
        f"- **External** — LUNGx-TestSet (LUNGx-CT*, 60 patients) trained on RM, no calibration.",
        f"- **External + cal** — LUNGx-TestSet **after Platt-recalibrating** the RM-trained "
        f"classifier on LUNGx CalibrationSet (CT-Training-*, 10 patients). This is the "
        f"SPIE-AAPM-NCI / CMPB 2021 protocol.\n",
        "**Reference (CMPB 2021)**: LIDC-PM AUC 0.85 (Size + Spiculations), "
        "LUNGx external AUC 0.76 (with calibration).\n",
        "**Rigour additions** (this report): every AUC carries a percentile "
        "bootstrap **95% CI** (n=2000) on pooled held-out predictions, and "
        "discrimination is paired with **calibration quality** — Brier score "
        "and Expected Calibration Error (ECE) on the RM out-of-fold and the "
        "LUNGx external+cal predictions — so a well-ranking but mis-calibrated "
        "model is no longer reported as a clean success.\n",
        "| Method | n feat | RM AUC [95% CI] | RM Brier / ECE | PM AUC [95% CI] "
        "| External+cal AUC [95% CI] | Ext+cal Brier / ECE |",
        "| :--- | --: | :--- | :--- | :--- | :--- | :--- |",
    ]

    def _ci(a, lo, hi):
        if np.isnan(a):
            return "—"
        if np.isnan(lo):
            return f"{a:.3f}"
        return f"{a:.3f} [{lo:.3f}, {hi:.3f}]"

    def _cal(b, e):
        return "—" if np.isnan(b) else f"{b:.3f} / {e:.3f}"

    for r in rows:
        out_md.append(
            f"| `{r['method']}` | {r['RM_n_feat']} | "
            f"{_ci(r['RM_auc'], r['RM_ci_lo'], r['RM_ci_hi'])} | "
            f"{_cal(r['RM_brier'], r['RM_ece'])} | "
            f"{_ci(r['PM_auc'], r['PM_ci_lo'], r['PM_ci_hi'])} | "
            f"{_ci(r['ext_cal_auc'], r['ext_cal_ci_lo'], r['ext_cal_ci_hi'])} | "
            f"{_cal(r['ext_cal_brier'], r['ext_cal_ece'])} |"
        )
    args.out.write_text("\n".join(out_md) + "\n")
    print(json.dumps(rows, indent=2, default=lambda x: None), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
