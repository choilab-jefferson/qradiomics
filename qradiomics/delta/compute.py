"""Delta and trend feature computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["DeltaPair", "compute_delta", "compute_trend"]


@dataclass(frozen=True)
class DeltaPair:
    """Identifier for a delta pair: take ``late`` − ``early`` per group."""

    early: str
    late: str


# Columns that identify a (patient, ROI) group rather than a feature.
_DEFAULT_KEY_COLS = ("patient_id", "roi_name")
# Column whose value selects the timepoint within a group.
_DEFAULT_TIMEPOINT_COL = "timepoint"


def compute_delta(
    features: pd.DataFrame,
    pair: DeltaPair,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    key_cols: Sequence[str] = _DEFAULT_KEY_COLS,
    timepoint_col: str = _DEFAULT_TIMEPOINT_COL,
) -> pd.DataFrame:
    """Return ``feature(late) − feature(early)`` per group.

    Args:
        features: Long-form DataFrame with one row per
            ``(*key_cols, timepoint_col, *feature_cols)``.
        pair: The two timepoints to difference.
        feature_cols: Columns to difference. Defaults to all numeric
            columns that are not in ``key_cols`` or ``timepoint_col``.
        key_cols: Grouping columns (default ``(patient_id, roi_name)``).
        timepoint_col: Column selecting the timepoint within a group.

    Returns:
        DataFrame with ``key_cols`` plus one ``delta_<feature>`` column
        per requested feature. Groups missing either timepoint are
        dropped.
    """
    if timepoint_col not in features.columns:
        raise ValueError(f"timepoint column {timepoint_col!r} not found in features.")
    for col in key_cols:
        if col not in features.columns:
            raise ValueError(f"key column {col!r} not found in features.")

    if feature_cols is None:
        feature_cols = _infer_feature_cols(features, key_cols, timepoint_col)
    else:
        feature_cols = list(feature_cols)

    early = features[features[timepoint_col] == pair.early]
    late = features[features[timepoint_col] == pair.late]
    if early.empty or late.empty:
        return pd.DataFrame(columns=list(key_cols) + [f"delta_{c}" for c in feature_cols])

    early_grp = early.set_index(list(key_cols))[list(feature_cols)]
    late_grp = late.set_index(list(key_cols))[list(feature_cols)]
    common = early_grp.index.intersection(late_grp.index)
    if common.empty:
        return pd.DataFrame(columns=list(key_cols) + [f"delta_{c}" for c in feature_cols])

    delta = late_grp.loc[common] - early_grp.loc[common]
    delta = delta.rename(columns={c: f"delta_{c}" for c in feature_cols})
    return delta.reset_index()


def compute_trend(
    features: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    key_cols: Sequence[str] = _DEFAULT_KEY_COLS,
    timepoint_col: str = _DEFAULT_TIMEPOINT_COL,
    time_index_col: Optional[str] = None,
) -> pd.DataFrame:
    """Per-group least-squares slope of each feature against time.

    The time axis defaults to ``range(n)`` over the group's timepoints
    sorted **alphabetically** — pass ``time_index_col`` (e.g.
    ``relative_day`` or ``fraction_number``) to use an explicit time
    coordinate. Mixing numeric tokens of different widths into
    ``timepoint`` labels (``"fx5"`` vs ``"fx10"``) will reorder them
    lexically; use zero-padded labels or supply ``time_index_col``.

    Args:
        features: Long-form DataFrame as in :func:`compute_delta`.
        feature_cols: Columns to fit. Defaults to all numeric columns
            that are not in ``key_cols``, ``timepoint_col``, or
            ``time_index_col``.
        key_cols: Grouping columns.
        timepoint_col: Column used to sort the timeseries when
            ``time_index_col`` is None.
        time_index_col: Optional explicit time axis.

    Returns:
        DataFrame with ``key_cols`` plus one ``slope_<feature>`` column
        per requested feature. Groups with fewer than 2 timepoints are
        dropped.
    """
    exclude: List[str] = list(key_cols) + [timepoint_col]
    if time_index_col is not None:
        exclude.append(time_index_col)
    if feature_cols is None:
        feature_cols = _infer_feature_cols(features, key_cols, timepoint_col, time_index_col)
    else:
        feature_cols = list(feature_cols)

    rows: List[dict] = []
    sorted_features = features.sort_values(
        list(key_cols) + ([time_index_col] if time_index_col else [timepoint_col])
    )
    for group_key, group in sorted_features.groupby(list(key_cols), sort=False):
        if len(group) < 2:
            continue
        if time_index_col:
            t = group[time_index_col].to_numpy(dtype=float)
        else:
            t = np.arange(len(group), dtype=float)
        row: dict = {col: val for col, val in zip(key_cols, group_key if isinstance(group_key, tuple) else (group_key,))}
        for col in feature_cols:
            y = group[col].to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(t)
            if mask.sum() < 2:
                row[f"slope_{col}"] = float("nan")
                continue
            ti = t[mask]
            yi = y[mask]
            slope, _ = np.polyfit(ti, yi, 1)
            row[f"slope_{col}"] = float(slope)
        rows.append(row)
    return pd.DataFrame(rows)


def _infer_feature_cols(
    features: pd.DataFrame,
    key_cols: Iterable[str],
    timepoint_col: str,
    time_index_col: Optional[str] = None,
) -> List[str]:
    skip = set(key_cols) | {timepoint_col}
    if time_index_col:
        skip.add(time_index_col)
    return [
        c
        for c in features.columns
        if c not in skip and pd.api.types.is_numeric_dtype(features[c])
    ]
