"""CIR (Clinically-Interpretable Radiomics, MICCAI 2022) reproduction.

The CIR paper (Choi/Nadeem et al. — github.com/nadeemlab/CIR) reports
malignancy AUC on:

    LIDC-PM (N=72, pathology-confirmed) :  AUC = 0.813  (mesh + encoder)
    LUNGx   (N=73)                      :  AUC = 0.743

Our reproduction uses qradiomics-public interpretable features (radiomics
1409 + spiculation Np/Na/Nl/Na_att/s1/s2) — NOT the CIR mesh-encoder NN.

Proxy criteria (since LIDC pathology subset list is gated):
  LIDC-PM proxy  = LIDC nodules with malignancy ∈ {1, 5}  (strong consensus)
  LUNGx          = the full SPIE-AAPM-NCI cohort

The script trains:
    (a) Internal CV on LIDC-PM proxy (5-fold)
    (b) External: train on full LIDC-malignancy-binary → test on LUNGx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

METADATA = {
    "pid", "reader", "nodule_id", "n_voxels", "volume_mm3",
    "malignancy", "subtlety", "calcification", "sphericity_score",
    "margin", "lobulation_score", "spiculation_score", "texture",
    "status", "status_radiomics", "status_spic", "error", "traceback",
    "diagnosis", "y_malignant", "scan", "y", "nodule_key",
}
SPIC = ["spic_Np", "spic_Na", "spic_Nl", "spic_Na_att", "spic_s1", "spic_s2"]


def _aggregate_lidc(df: pd.DataFrame) -> pd.DataFrame:
    """Average reader values per (pid, nodule_id) → one row per nodule."""
    df = df.copy()
    df["nodule_key"] = df["pid"].astype(str) + ":" + df["nodule_id"].astype(str)
    numeric = [c for c in df.columns
                if c not in {"nodule_key", "pid", "nodule_id"}
                and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby("nodule_key").agg({**{c: "mean" for c in numeric},
                                       "pid": "first", "nodule_id": "first"})
    return g.reset_index(drop=True)


def _select_features(df: pd.DataFrame, mode: str) -> list[str]:
    if mode == "spic":
        return [c for c in SPIC if c in df.columns]
    if mode == "radiomics":
        return [c for c in df.columns
                if c not in METADATA | set(SPIC)
                and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    # combined
    return _select_features(df, "radiomics") + _select_features(df, "spic")


def _cv_auc(X, y, n_splits=5, top_k=50, seed=42):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in kf.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]; ytr, yte = y.iloc[tr], y.iloc[te]
        med = Xtr.median(); Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
        var_mask = Xtr.var() > 1e-10
        Xtr = Xtr.loc[:, var_mask]; Xte = Xte.loc[:, var_mask]
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
        clf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                      n_jobs=-1, class_weight="balanced")
        clf.fit(sc.transform(Xtr), ytr)
        prob = clf.predict_proba(sc.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(yte, prob))
    return float(np.mean(aucs)), float(np.std(aucs))


def _external_auc(Xtr, ytr, Xte, yte, top_k=50, seed=42):
    """Train on (Xtr, ytr), test on (Xte, yte)."""
    med = Xtr.median(); Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
    var_mask = Xtr.var() > 1e-10
    Xtr = Xtr.loc[:, var_mask]; Xte = Xte.loc[:, var_mask]
    cor = Xtr.corr().abs()
    upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    Xtr = Xtr.drop(columns=drop); Xte = Xte.drop(columns=drop)
    common = Xtr.columns.intersection(Xte.columns)
    Xtr = Xtr[common]; Xte = Xte[common]
    if Xtr.shape[1] > top_k:
        scores = Xtr.apply(lambda col: abs(np.corrcoef(col, ytr)[0, 1])
                            if col.std() > 1e-10 else 0.0)
        keep = scores.nlargest(top_k).index
        Xtr = Xtr[keep]; Xte = Xte[keep]
    sc = StandardScaler().fit(Xtr)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                  n_jobs=-1, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lidc-features", required=True, type=Path)
    p.add_argument("--lungx-features", required=True, type=Path)
    p.add_argument("--lidc-pm-ids", type=Path, default=None,
                   help="Text file with LIDC-PM patient IDs (one per line). If "
                        "given, restricts the LIDC-PM internal CV to these IDs "
                        "exactly (matching the CIR 2022 paper's pathology subset).")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    lidc = pd.read_csv(args.lidc_features)
    lungx = pd.read_csv(args.lungx_features)

    if args.lidc_pm_ids and args.lidc_pm_ids.exists():
        pm_ids = set(line.strip() for line in args.lidc_pm_ids.read_text().splitlines()
                     if line.strip())
        print(f"LIDC-PM (CIR paper) subset: {len(pm_ids)} patient IDs",
              file=sys.stderr)
    else:
        pm_ids = None

    print(f"LIDC: {len(lidc)} rows, LUNGx: {len(lungx)} rows", file=sys.stderr)

    # LIDC → one row per nodule, malignancy binary 1 vs 5 (proxy for LIDC-PM)
    lidc_agg = _aggregate_lidc(lidc)
    if pm_ids is not None:
        # Use the exact CIR paper subset
        lidc_pm = lidc_agg[lidc_agg["pid"].isin(pm_ids)].copy()
        # Binary malignancy: ≥3 (moderate-high) vs <3 (low) for pathology-confirmed
        lidc_pm["y"] = (lidc_pm["malignancy"] >= 3).astype(int)
    else:
        lidc_pm = lidc_agg[lidc_agg["malignancy"].isin([1, 5])].copy()
        lidc_pm["y"] = (lidc_pm["malignancy"] >= 4).astype(int)
    lidc_all = lidc_agg[lidc_agg["malignancy"].between(1, 5)
                          & (lidc_agg["malignancy"] != 3)].copy()
    lidc_all["y"] = (lidc_all["malignancy"] >= 4).astype(int)

    # LUNGx already binary
    lungx = lungx[lungx["status_radiomics"] == "ok"].copy()
    lungx["y"] = lungx["y_malignant"].astype(int)

    print(f"LIDC-PM proxy ({{1, 5}}): n={len(lidc_pm)} "
          f"(+{int(lidc_pm['y'].sum())} / -{int((1 - lidc_pm['y']).sum())})",
          file=sys.stderr)
    print(f"LIDC binary    ({{1,2}} vs {{4,5}}): n={len(lidc_all)} "
          f"(+{int(lidc_all['y'].sum())} / -{int((1 - lidc_all['y']).sum())})",
          file=sys.stderr)
    print(f"LUNGx:                                  n={len(lungx)} "
          f"(+{int(lungx['y'].sum())} / -{int((1 - lungx['y']).sum())})",
          file=sys.stderr)

    lines = ["# CIR reproducibility — interpretable feature classifier",
             f"\nSources:",
             f"  - LIDC features: `{args.lidc_features}`",
             f"  - LUNGx features: `{args.lungx_features}`",
             f"\nReference: CIR (Choi/Nadeem 2022, MICCAI) — mesh-encoder model",
             f"reports LIDC-PM(N=72) AUC = 0.813, LUNGx AUC = 0.743.",
             f"Our reproduction uses interpretable radiomics + spiculation",
             f"features (no NN encoder) — expected to undershoot the CIR mesh+encoder baseline."]

    out = {}

    # === (a) LIDC-PM proxy — internal CV ===
    for label, mode in (("radiomics 1409 → top-50", "radiomics"),
                         ("spiculation only (6 feats)", "spic"),
                         ("radiomics + spiculation",  "combined")):
        cols = _select_features(lidc_pm, mode)
        if not cols or len(lidc_pm) < 25: continue
        auc, std = _cv_auc(lidc_pm[cols], lidc_pm["y"])
        out[f"lidc_pm_{mode}"] = {"auc": auc, "std": std,
                                   "n": len(lidc_pm), "n_feat": len(cols)}
        lines.append(f"\n## LIDC-PM proxy — {label}")
        lines.append(f"- 5-fold CV AUC: **{auc:.3f} ± {std:.3f}**")
        lines.append(f"- n = {len(lidc_pm)} nodules, {len(cols)} candidate features")

    # === (b) External: LIDC train → LUNGx test ===
    for label, mode in (("radiomics", "radiomics"),
                         ("spiculation", "spic"),
                         ("radiomics + spiculation", "combined")):
        lidc_cols = _select_features(lidc_all, mode)
        lungx_cols = _select_features(lungx, mode)
        common = [c for c in lidc_cols if c in lungx_cols]
        if not common or len(lidc_all) < 25 or len(lungx) < 10: continue
        try:
            auc = _external_auc(lidc_all[common], lidc_all["y"],
                                 lungx[common], lungx["y"])
            out[f"external_lidc→lungx_{mode}"] = {"auc": float(auc),
                                                    "n_train": len(lidc_all),
                                                    "n_test": len(lungx),
                                                    "n_feat": len(common)}
            lines.append(f"\n## External — LIDC binary → LUNGx ({label})")
            lines.append(f"- Train: LIDC malignancy 1-2 vs 4-5 (n={len(lidc_all)})")
            lines.append(f"- Test: LUNGx (n={len(lungx)})")
            lines.append(f"- AUC: **{auc:.3f}**  vs CIR LUNGx 0.743 (mesh+encoder)")
        except Exception as e:
            lines.append(f"\n## External — LIDC → LUNGx ({label}): error {e}")

    args.out.write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
