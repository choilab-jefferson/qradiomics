"""Pair manifest: the contract between selection and the per-pair pipeline.

One row = one (image, label) *pair* of one case, e.g. ``CT`` x ``RS.manual``.
A selection step (outside qradiomics: PACS query, series picking, RTSTRUCT
choice) writes this table; everything downstream (convert / preprocess /
extract / Nextflow fan-out) only *reads* it.

Columns (all text; ``resample_mm`` parses as float):

========== ============================================================
cohort         cohort / study label (free text, id charset)
case_id        patient / case identifier (``[A-Za-z0-9_.-]+``)
pair_id        globally unique pair id (``[A-Za-z0-9_.-]+``); fan-out key
image_key      image role, e.g. ``CT`` / ``PT`` (ok rows: required)
label_key      label role, e.g. ``manual`` / ``auto`` (ok rows: required)
series_uid     SeriesInstanceUID of the chosen image series (ok: required)
series_dir     directory holding that series' DICOM files (ok: required)
rtstruct_path  RTSTRUCT .dcm file (ok: required)
rtstruct_sha   sha256 hex of the RTSTRUCT file (ok: required; feeds -resume)
roi_candidates ``|``-separated ROI names, tried in order; first match wins
resample_mm    isotropic resample spacing in mm; empty = no resample
status         ``ok`` | ``skipped`` | ``failed``
reason         why a row is not ``ok`` (required when status != ok)
========== ============================================================

The RTSTRUCT is rasterised on the CT series it was drawn on: for a pair whose
``image_key`` is not ``CT`` (e.g. PT) that is the ``series_dir`` of the same
case's ``CT`` row sharing the ``rtstruct_path`` (own ``series_dir`` if none), and
``qr preprocess`` resamples the label onto the image grid.

Only ``status == ok`` rows enter the pipeline; ``skipped`` / ``failed`` rows
stay in the manifest so the coverage table can report them. Format is chosen
by file suffix: ``.parquet`` (preferred) or ``.csv``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

PAIR_COLUMNS = (
    "cohort",
    "case_id",
    "pair_id",
    "image_key",
    "label_key",
    "series_uid",
    "series_dir",
    "rtstruct_path",
    "rtstruct_sha",
    "roi_candidates",
    "resample_mm",
    "status",
    "reason",
)
PAIR_STATUSES = ("ok", "skipped", "failed")
ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
#: columns every row needs regardless of status
_ALWAYS = ("cohort", "case_id", "pair_id", "status")
#: columns an ``ok`` row additionally needs
_OK_REQUIRED = (
    "image_key",
    "label_key",
    "series_uid",
    "series_dir",
    "rtstruct_path",
    "rtstruct_sha",
    "roi_candidates",
)


class PairManifestError(ValueError):
    """The pair manifest violates the contract (message lists every problem)."""


_PARQUET_HINT = (
    "Reading/writing .parquet manifests needs the 'pyarrow' package (>=14.0), "
    "which is a core qradiomics dependency but is missing from this environment. "
    "Reinstall with `pip install -e .` (or `pip install 'pyarrow>=14.0'`), "
    "or use a .csv manifest path instead."
)


def _require_parquet_engine() -> None:
    """Raise an actionable error if pandas has no parquet engine installed."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(_PARQUET_HINT) from exc


def read_manifest(path) -> pd.DataFrame:
    """Read any qr manifest (.csv or .parquet) as an all-text frame; NaN -> ''."""
    p = Path(path)
    if p.suffix == ".parquet":
        _require_parquet_engine()
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, dtype=str)
    return df.astype(object).where(df.notna(), "").astype(str)


def write_manifest(df: pd.DataFrame, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".parquet":
        _require_parquet_engine()
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


def is_pair_manifest(df: pd.DataFrame) -> bool:
    return "pair_id" in df.columns


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    return next((n for n in names if n in df.columns), None)


def select_rows(
    df: pd.DataFrame,
    only: Optional[str] = None,
    only_case: Optional[str] = None,
    cases: Sequence[str] = (),
) -> pd.DataFrame:
    """Filter a manifest to one pair (``only``: pair_id, else patient_id) and/or
    to cases (``only_case`` plus the repeatable ``cases``: case_id, else
    patient_id). Raises ValueError when a filter was asked for and matches
    nothing (never silently runs on zero rows)."""
    out = df
    if only is not None:
        key = _col(out, "pair_id", "patient_id")
        if key is None:
            raise ValueError("manifest has no pair_id / patient_id column for --only")
        out = out[out[key] == only]
        if out.empty:
            raise ValueError(f"--only {only!r}: no such {key} in the manifest")
    wanted = {c for c in (*cases, only_case) if c}
    if wanted:
        key = _col(out, "case_id", "patient_id")
        if key is None:
            raise ValueError("manifest has no case_id / patient_id column for --case")
        out = out[out[key].isin(wanted)]
        if out.empty:
            raise ValueError(f"No manifest rows for --case {', '.join(sorted(wanted))}")
    return out.reset_index(drop=True)


def split_roi_candidates(value: str) -> List[str]:
    """``'Heart|heart|HEART'`` -> ['Heart', 'heart', 'HEART'] (blank items dropped)."""
    return [c.strip() for c in str(value).split("|") if c.strip()]


def validate_pairs(df: pd.DataFrame, check_paths: bool = False) -> List[str]:
    """Return every contract violation as a message (empty list = valid)."""
    errs: List[str] = []
    missing = [c for c in PAIR_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing column(s): {missing}"]
    if df.empty:
        return ["manifest has no rows"]
    df = df.astype(object).where(df.notna(), "").astype(str)

    dup = df["pair_id"][df["pair_id"].duplicated()].unique().tolist()
    if dup:
        errs.append(f"duplicate pair_id(s): {dup[:5]}")

    for i, r in df.iterrows():
        where = f"row {i} (pair_id={r['pair_id']!r})"
        for c in _ALWAYS:
            if not r[c].strip():
                errs.append(f"{where}: {c} is empty")
        for c in ("cohort", "case_id", "pair_id"):
            if r[c].strip() and not ID_RE.fullmatch(r[c]):
                errs.append(f"{where}: {c} {r[c]!r} must match [A-Za-z0-9_.-]+")
        st = r["status"].strip()
        if st and st not in PAIR_STATUSES:
            errs.append(f"{where}: status {st!r} not in {list(PAIR_STATUSES)}")
        rs = r["resample_mm"].strip()
        if rs:
            try:
                if not float(rs) > 0:
                    raise ValueError
            except ValueError:
                errs.append(f"{where}: resample_mm {rs!r} must be a positive number or empty")
        if st != "ok":
            if st in PAIR_STATUSES and not r["reason"].strip():
                errs.append(f"{where}: status={st} requires a reason")
            continue
        for c in _OK_REQUIRED:
            if not r[c].strip():
                errs.append(f"{where}: status=ok requires {c}")
        if r["rtstruct_sha"].strip() and not SHA256_RE.fullmatch(r["rtstruct_sha"].strip()):
            errs.append(f"{where}: rtstruct_sha must be 64 hex chars")
        if r["roi_candidates"].strip() and not split_roi_candidates(r["roi_candidates"]):
            errs.append(f"{where}: roi_candidates has no names")
        if check_paths:
            if r["series_dir"].strip() and not Path(r["series_dir"]).is_dir():
                errs.append(f"{where}: series_dir does not exist")
            if r["rtstruct_path"].strip() and not Path(r["rtstruct_path"]).is_file():
                errs.append(f"{where}: rtstruct_path does not exist")
    return errs


def load_pairs(path, check_paths: bool = False) -> pd.DataFrame:
    """Read + validate a pair manifest; raise PairManifestError listing all problems."""
    df = read_manifest(path)
    errs = validate_pairs(df, check_paths=check_paths)
    if errs:
        raise PairManifestError(f"{path}: " + "; ".join(errs[:20]))
    return df


def ok_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that enter the pipeline (status == ok)."""
    return df[df["status"] == "ok"].reset_index(drop=True)
