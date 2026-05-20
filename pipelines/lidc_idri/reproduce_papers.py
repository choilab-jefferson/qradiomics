"""LIDC reproducibility — train classifiers, compute AUC, compare to published numbers.

Reproduces three Choi papers from a single features CSV produced by
`extract_features.py`:

    1. CMPB 2014 (AHSN nodule detection) — LIDC nodule vs non-nodule.
       Currently NOT implemented end-to-end here because the present
       features.csv only contains TRUE nodules. (AHSN candidate detection
       needs whole-lung candidate generation — TBD; see paper Table 4 ROC.)

    2. Med Phys 2018 (radiomics for early lung cancer detection) — LIDC
       malignancy ≥4 vs ≤2 binary classification using the full PyRadiomics
       feature set (1409 features) with RF + leakage-safe top-K.
       Reference AUC (paper): 0.83 — 0.95 depending on cohort / feature set.

    3. CMPB 2021 (spiculation quantification) — LIDC malignancy classification
       using only the 5 interpretable spiculation features Np/Na/Nl/s1/s2.
       Reference AUC (paper): 0.80 — 0.85 depending on cohort.

Usage:
    python reproduce_papers.py --features features.csv --out report.md
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


METADATA_COLS = {
    "pid", "reader", "nodule_id", "n_voxels", "volume_mm3",
    "malignancy", "subtlety", "calcification", "sphericity_score",
    "margin", "lobulation_score", "spiculation_score", "texture",
    "status", "status_radiomics", "status_spic", "error", "traceback",
    "y", "nodule_key", "y_malignant", "diagnosis", "scan",
}

SPICULATION_COLS = ["spic_Np", "spic_Na", "spic_Nl", "spic_Na_att",
                    "spic_s1", "spic_s2"]


def _binary_malignancy(df: pd.DataFrame, drop_ambiguous: bool = True) -> pd.DataFrame:
    """Map LIDC 1-5 malignancy to binary (≥4 = malignant, ≤2 = benign)."""
    df = df.copy()
    df = df[df["malignancy"].between(1, 5)]
    if drop_ambiguous:
        df = df[df["malignancy"] != 3]
    df["y"] = (df["malignancy"] >= 4).astype(int)
    return df


def _select_radiomics_columns(df: pd.DataFrame) -> list[str]:
    """All numeric pyradiomics feature columns (exclude metadata + spiculation)."""
    drop = set(METADATA_COLS) | set(SPICULATION_COLS)
    cols = [c for c in df.columns
            if c not in drop
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().any()]
    return cols


def _cv_auc(X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
            top_k: int = 50, seed: int = 42) -> tuple[float, float]:
    """5-fold leakage-safe CV: corr<0.95 drop + univariate top-K + RF AUC."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for fold, (tr, te) in enumerate(kf.split(X, y)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        # Imputation
        med = Xtr.median()
        Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)

        # Drop near-constant columns (zero variance training)
        var_mask = Xtr.var() > 1e-10
        Xtr = Xtr.loc[:, var_mask]; Xte = Xte.loc[:, var_mask]

        # Drop |corr| > 0.95 (keep first)
        cor = Xtr.corr().abs()
        upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > 0.95)]
        Xtr = Xtr.drop(columns=drop); Xte = Xte.drop(columns=drop)

        # Univariate selection — |Pearson r| with y, top-K
        if Xtr.shape[1] > top_k:
            scores = Xtr.apply(lambda col: abs(np.corrcoef(col, ytr)[0, 1])
                                if col.std() > 1e-10 else 0.0)
            keep = scores.nlargest(top_k).index
            Xtr = Xtr[keep]; Xte = Xte[keep]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)

        clf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                     n_jobs=-1, class_weight="balanced")
        clf.fit(Xtr_s, ytr)
        prob = clf.predict_proba(Xte_s)[:, 1]
        aucs.append(roc_auc_score(yte, prob))
    return float(np.mean(aucs)), float(np.std(aucs))


def reproduce(features_csv: Path, out_md: Path) -> dict:
    df = pd.read_csv(features_csv)
    print(f"loaded {len(df)} feature rows", file=sys.stderr)

    # Aggregate per (pid, reader, nodule_id) — already at that grain
    # For malignancy classification, average the 4 reader scores per nodule.
    # (More principled than treating each reader separately; matches paper.)
    if "reader" in df.columns:
        df["nodule_key"] = df["pid"] + ":" + df["nodule_id"].astype(str)
        # Mean-of-readers per nodule
        radio_cols = [c for c in df.columns if c not in METADATA_COLS | {"nodule_key"}]
        agg = (df.groupby("nodule_key")
                 .agg({**{c: "mean" for c in radio_cols if pd.api.types.is_numeric_dtype(df[c])},
                       "malignancy": "mean",
                       "pid": "first", "nodule_id": "first",
                       "n_voxels": "mean"})
                 .reset_index())
        df = agg

    df = _binary_malignancy(df, drop_ambiguous=True)
    print(f"  → {len(df)} nodules after malignancy filter "
          f"(mal+: {int(df['y'].sum())}, mal-: {int((1 - df['y']).sum())})",
          file=sys.stderr)

    lines = [f"# LIDC reproducibility — paper-by-paper comparison\n"]
    lines += [f"Source: `{features_csv}` ({len(df)} unique nodules; "
              f"{int(df['y'].sum())} malignancy ≥4, {int((1 - df['y']).sum())} ≤2)\n"]

    results = {}

    # ── Paper 2: Med Phys 2018 — full radiomics ──────────────────────────
    radio_cols = _select_radiomics_columns(df)
    print(f"\n[Med Phys 2018] {len(radio_cols)} radiomics features available",
          file=sys.stderr)
    if radio_cols:
        X = df[radio_cols]; y = df["y"]
        if len(df) >= 25:                            # need enough for CV
            auc, std = _cv_auc(X, y, top_k=50)
            results["medphys2018_radiomics"] = {"auc": auc, "std": std,
                                                 "n": len(df),
                                                 "n_features": len(radio_cols)}
            lines += [f"\n## Med Phys 2018 — full radiomics (1409 features → top-50)",
                      f"- CV AUC (5-fold): **{auc:.3f} ± {std:.3f}**",
                      f"- n = {len(df)} nodules, {len(radio_cols)} candidate radiomics features",
                      f"- Reference: AUC 0.83 — 0.95 (paper Tables 2-4)"]
        else:
            lines += [f"\n## Med Phys 2018 — skipped (n={len(df)} < 25)"]

    # ── Paper 3: CMPB 2021 — spiculation only ────────────────────────────
    spic_present = [c for c in SPICULATION_COLS if c in df.columns]
    print(f"\n[CMPB 2021] spiculation cols present: {spic_present}",
          file=sys.stderr)
    if spic_present:
        X = df[spic_present]; y = df["y"]
        if len(df) >= 25:
            auc, std = _cv_auc(X, y, top_k=len(spic_present))
            results["cmpb2021_spiculation"] = {"auc": auc, "std": std,
                                                "n": len(df),
                                                "features": spic_present}
            lines += [f"\n## CMPB 2021 — interpretable spiculation features only",
                      f"- CV AUC (5-fold): **{auc:.3f} ± {std:.3f}**",
                      f"- n = {len(df)} nodules, features: {', '.join(spic_present)}",
                      f"- Reference: AUC 0.80 — 0.85 (paper Table 5)"]

    # ── Paper 1: CMPB 2014 — AHSN nodule detection ───────────────────────
    lines += [f"\n## CMPB 2014 — AHSN nodule detection",
              f"- **Skipped here**: this CSV holds only TRUE annotated nodules.",
              f"  Reproduction requires whole-lung candidate generation",
              f"  (`qradiomics.shape.detect_candidates`) emitting positives + negatives.",
              f"  Build out as a separate pipeline (`pipelines/lidc_idri/ahsn_detection.py`)."]

    out_md.write_text("\n".join(lines) + "\n")
    print(f"\n→ {out_md}", file=sys.stderr)
    print(json.dumps(results, indent=2), file=sys.stderr)
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, type=Path,
                   help="Features CSV from extract_features.py")
    p.add_argument("--out", required=True, type=Path,
                   help="Output markdown report")
    args = p.parse_args()
    reproduce(args.features, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
