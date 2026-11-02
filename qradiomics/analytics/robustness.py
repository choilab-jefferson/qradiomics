"""ICC-based feature-robustness filtering.

Multi-centre radiomics studies routinely drop features that are not
reproducible across the harmonization step (or across manual vs auto
contours, or test/retest). The standard gate is the intraclass
correlation coefficient (ICC); features with ICC below a threshold
(commonly 0.75 for "good" / 0.90 for "excellent") are discarded.

This module computes ICC(3,1) — two-way mixed-effects, consistency,
single rater — between two measurements of the same feature on the
same patients (e.g. pre- vs post-ComBat, or manual vs auto contour)
using ``pingouin`` (already a project dependency).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

try:
    import pingouin as pg
except ImportError:  # pragma: no cover
    pg = None

__all__ = ["feature_icc", "icc_filter", "volume_correlation_screen"]

# pingouin 0.6 renamed intraclass_corr()'s "Type" column values from the
# ICC1/ICC2/ICC3(/...k) short codes to ICC(1,1)/ICC(A,1)/ICC(C,1)(/...k) --
# same six ICC forms, new labels. Map old -> new so callers can keep using
# the short codes regardless of which pingouin is installed; looked up as a
# fallback only, so a pinned old pingouin (which already returns the short
# codes) still works unchanged.
_ICC_TYPE_ALIASES = {
    "ICC1": "ICC(1,1)",
    "ICC2": "ICC(A,1)",
    "ICC3": "ICC(C,1)",
    "ICC1k": "ICC(1,k)",
    "ICC2k": "ICC(A,k)",
    "ICC3k": "ICC(C,k)",
}


def feature_icc(
    values_a: pd.Series, values_b: pd.Series, patient_ids: pd.Series, *, icc_type: str = "ICC3"
) -> float:
    """ICC between two measurement series for one feature.

    Returns NaN when the ICC cannot be computed (degenerate variance).
    """
    if pg is None:
        raise ImportError("pingouin is required for ICC filtering " "(pip install pingouin)")
    long = pd.concat(
        [
            pd.DataFrame({"PID": patient_ids.to_numpy(), "Val": values_a.to_numpy(), "Rater": "A"}),
            pd.DataFrame({"PID": patient_ids.to_numpy(), "Val": values_b.to_numpy(), "Rater": "B"}),
        ],
        ignore_index=True,
    )
    try:
        res = pg.intraclass_corr(
            data=long, targets="PID", raters="Rater", ratings="Val", nan_policy="omit"
        )
        by_type = res.set_index("Type")
        if icc_type in by_type.index:
            return float(by_type.loc[icc_type, "ICC"])
        return float(by_type.loc[_ICC_TYPE_ALIASES[icc_type], "ICC"])
    except Exception:
        return float("nan")


def icc_filter(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    feature_cols: Sequence[str],
    patient_id_col: str,
    *,
    threshold: float = 0.75,
    icc_type: str = "ICC3",
) -> tuple[list[str], pd.DataFrame]:
    """Keep features whose ICC(A, B) ≥ threshold.

    ``df_a`` and ``df_b`` are two measurements of the same patients
    (same ``patient_id_col`` ordering not required — joined on PID).

    Returns ``(kept_feature_names, icc_table)`` where ``icc_table``
    has one row per feature with its ICC value.
    """
    a = df_a.set_index(patient_id_col)
    b = df_b.set_index(patient_id_col)
    common_pids = a.index.intersection(b.index)
    a = a.loc[common_pids]
    b = b.loc[common_pids]
    pids = pd.Series(common_pids)

    rows = []
    kept = []
    for f in feature_cols:
        if f not in a.columns or f not in b.columns:
            continue
        icc = feature_icc(
            a[f].reset_index(drop=True), b[f].reset_index(drop=True), pids, icc_type=icc_type
        )
        rows.append({"feature": f, "icc": icc})
        if pd.notna(icc) and icc >= threshold:
            kept.append(f)
    return kept, pd.DataFrame(rows)


def volume_correlation_screen(
    df: pd.DataFrame,
    volume_col: str,
    feature_cols: Sequence[str] | None = None,
    *,
    max_abs_r: float = 0.5,
    log_volume: bool = True,
) -> pd.DataFrame:
    """Gate-0 size-confounding screen: correlate each feature against (log) ROI volume.

    Flags features whose absolute Pearson correlation with ``log(volume)`` (or
    raw volume when ``log_volume=False``) exceeds ``max_abs_r``.

    Returns a DataFrame with columns ``["feature", "pearson_r", "spearman_r",
    "abs_pearson_r", "passes_gate"]``.
    """
    if volume_col not in df.columns:
        raise KeyError(f"volume_col {volume_col!r} not in DataFrame columns")
    vol_raw = pd.to_numeric(df[volume_col], errors="coerce").to_numpy(dtype=float)
    if log_volume:
        if np.any(vol_raw <= 0):
            raise ValueError("volume_col must be strictly positive when log_volume=True")
        vol = np.log(vol_raw)
    else:
        vol = vol_raw

    if feature_cols is None:
        feature_cols = [
            c for c in df.columns if c != volume_col and pd.api.types.is_numeric_dtype(df[c])
        ]

    rows = []
    for col in feature_cols:
        if col not in df.columns or col == volume_col:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(vol) & np.isfinite(vals)
        if valid.sum() < 3 or np.std(vals[valid]) == 0.0 or np.std(vol[valid]) == 0.0:
            pr, sr = float("nan"), float("nan")
            abs_r = float("nan")
            passes = False
        else:
            pr = float(sp_stats.pearsonr(vol[valid], vals[valid]).statistic)
            sr = float(sp_stats.spearmanr(vol[valid], vals[valid]).statistic)
            abs_r = abs(pr)
            passes = bool(abs_r <= max_abs_r)
        rows.append(
            {
                "feature": col,
                "pearson_r": pr,
                "spearman_r": sr,
                "abs_pearson_r": abs_r,
                "passes_gate": passes,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["feature", "pearson_r", "spearman_r", "abs_pearson_r", "passes_gate"],
    )
