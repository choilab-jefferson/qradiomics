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

import pandas as pd

try:
    import pingouin as pg
except ImportError:  # pragma: no cover
    pg = None

__all__ = ["feature_icc", "icc_filter"]


def feature_icc(values_a: pd.Series, values_b: pd.Series,
                patient_ids: pd.Series, *, icc_type: str = "ICC3") -> float:
    """ICC between two measurement series for one feature.

    Returns NaN when the ICC cannot be computed (degenerate variance).
    """
    if pg is None:
        raise ImportError("pingouin is required for ICC filtering "
                          "(pip install pingouin)")
    long = pd.concat([
        pd.DataFrame({"PID": patient_ids.to_numpy(), "Val": values_a.to_numpy(),
                      "Rater": "A"}),
        pd.DataFrame({"PID": patient_ids.to_numpy(), "Val": values_b.to_numpy(),
                      "Rater": "B"}),
    ], ignore_index=True)
    try:
        res = pg.intraclass_corr(data=long, targets="PID", raters="Rater",
                                 ratings="Val", nan_policy="omit")
        return float(res.set_index("Type").loc[icc_type, "ICC"])
    except Exception:
        return float("nan")


def icc_filter(df_a: pd.DataFrame, df_b: pd.DataFrame,
               feature_cols: Sequence[str], patient_id_col: str,
               *, threshold: float = 0.75,
               icc_type: str = "ICC3") -> tuple[list[str], pd.DataFrame]:
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
        icc = feature_icc(a[f].reset_index(drop=True),
                          b[f].reset_index(drop=True), pids, icc_type=icc_type)
        rows.append({"feature": f, "icc": icc})
        if pd.notna(icc) and icc >= threshold:
            kept.append(f)
    return kept, pd.DataFrame(rows)
