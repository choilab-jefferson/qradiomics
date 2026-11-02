"""Multi-engine and multi-timepoint feature fusion.

Merges several feature tables (internal engine outputs such as the
``pyradiomics_``/``pysera_`` prefixed CSVs from ``qr extract --engine a,b``,
plus external feature or parameter CSVs) into one tidy table keyed by
``patient_id`` and, optionally, a timepoint column.

Two timepoint layouts are supported:

* long (default): rows keyed by ``(patient_id, timepoint)``.
* wide: one row per patient; timepoint-bearing tables are pivoted so each
  feature column becomes ``<timepoint>__<column>``.

Exposed on the CLI as ``qr results fuse``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd

TableLike = Union[str, Path, pd.DataFrame]
#: A bare table, or ``(label, table)``; the label prefixes non-key columns.
TableSpec = Union[TableLike, Tuple[str, TableLike]]


def _load(spec: TableSpec) -> Tuple[Optional[str], pd.DataFrame]:
    label: Optional[str] = None
    table = spec
    if isinstance(spec, tuple):
        label, table = spec
    if isinstance(table, pd.DataFrame):
        return label, table.copy()
    return label, pd.read_csv(table)


def _prepare(
    spec: TableSpec,
    id_col: str,
    timepoint_col: Optional[str],
    wide: bool,
) -> pd.DataFrame:
    label, df = _load(spec)
    if id_col not in df.columns:
        raise ValueError(f"table {label or ''} is missing id column '{id_col}'")
    has_tp = bool(timepoint_col) and timepoint_col in df.columns
    keys = [id_col] + ([timepoint_col] if has_tp else [])
    if label:
        df = df.rename(columns={c: f"{label}_{c}" for c in df.columns if c not in keys})
    if df.duplicated(subset=keys).any():
        raise ValueError(f"table {label or ''} has duplicate rows for keys {keys}")
    if has_tp and wide:
        df[timepoint_col] = df[timepoint_col].astype(str)
        df = df.pivot(index=id_col, columns=timepoint_col)
        df.columns = [f"{tp}__{col}" for col, tp in df.columns]
        df = df.reset_index()
    return df


def fuse_feature_tables(
    internal: Sequence[TableSpec],
    external: Sequence[TableSpec] = (),
    id_col: str = "patient_id",
    timepoint_col: Optional[str] = None,
    wide: bool = False,
    how: str = "outer",
) -> pd.DataFrame:
    """Fuse internal engine tables and external CSVs into one feature table.

    Args:
        internal: engine feature tables (DataFrame, CSV path, or
            ``(label, table)``). Columns are expected to be engine-prefixed
            already; pass a label to prefix them here.
        external: external feature/parameter tables, same accepted forms.
            Tables lacking ``timepoint_col`` join on ``id_col`` only.
        id_col: patient key column.
        timepoint_col: optional timepoint column. ``None`` ignores time.
        wide: pivot timepoint-bearing tables to ``<timepoint>__<column>``.
        how: pandas merge strategy (``outer``, ``inner``, ``left``).

    Raises:
        ValueError: no inputs, missing id column, duplicate keys, or
            column-name collisions between tables.
    """
    specs: List[TableSpec] = [*internal, *external]
    if not specs:
        raise ValueError("fuse_feature_tables needs at least one table")
    frames = [_prepare(s, id_col, timepoint_col, wide) for s in specs]

    fused = frames[0]
    for nxt in frames[1:]:
        keys = [id_col]
        if timepoint_col and timepoint_col in fused.columns and timepoint_col in nxt.columns:
            keys.append(timepoint_col)
        clash = (set(fused.columns) & set(nxt.columns)) - set(keys)
        if clash:
            raise ValueError(
                f"column collision on fuse: {sorted(clash)[:5]}; pass a label to prefix tables"
            )
        fused = fused.merge(nxt, on=keys, how=how)
    return fused
