"""The encoder contract.

Every branch that produces a patient-level vector -- IBSI radiomics, an ROI-cropped
foundation-model encoder, a pathology MIL aggregator -- emits an :class:`Encoding`.
Fusion consumes only :class:`Encoding`, never a raw array.

The point of the contract is :class:`Provenance`. An embedding without a recorded ROI,
discretization, and stability gate cannot be reproduced or safely fused, and that is the
specific failure mode this project exists to avoid. See ``wiki/architecture/Encoder-Contract.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "SCOPES",
    "Provenance",
    "Encoding",
    "roi_hash",
    "check_fusable",
    "gate_encodings_by_icc",
    "FusionError",
]

#: Which set of cells a measurement's numbers describe. The ladder descends patient -> organ
#: -> lesion -> cell -> gene and then TURNS: a germline variant is the smallest measurement
#: and the widest scope there is, present in every cell and shared with a population. Somatic
#: and germline are the same assay at opposite ends. See wiki concepts/Scope-Ladder.
SCOPES = (
    "population",
    "patient",
    "organ",
    "lesion",
    "cell",
    "gene-somatic",
    "gene-germline",
)


class FusionError(ValueError):
    """Encodings that cannot be fused: different patient, object, region role, or observation."""


def roi_hash(mask: np.ndarray, spacing_mm: Sequence[float]) -> str:
    """Content hash of a mask plus its voxel spacing.

    Spacing is part of the hash because the same voxel array at different spacing is a
    different physical ROI, and mixing the two is a silent geometry bug rather than a
    loud one.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())
    h.update(np.asarray(spacing_mm, dtype=np.float64).tobytes())
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class Provenance:
    """What was encoded, from what, and under which settings."""

    encoder: str
    encoder_version: str
    modality: str
    scope: str | None = None
    roi_source: str | None = None
    roi_role: str | None = None
    roi: str | None = None
    spacing_mm: tuple[float, ...] | None = None
    pad_mm: float = 0.0
    bin_width: float | None = None
    normalized: bool | None = None
    site: str | None = None
    stability: Mapping[str, float] | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope is not None and self.scope not in SCOPES:
            raise ValueError(f"scope {self.scope!r} not in SCOPES {SCOPES}")

    @property
    def params(self) -> Mapping[str, object]:
        """Encoder- and morphology-specific parameters recorded in ``extra``."""
        return self.extra


@dataclass(frozen=True, slots=True)
class Encoding:
    """One subject, one branch, one fixed-length vector."""

    subject_id: str
    vector: np.ndarray
    provenance: Provenance
    names: tuple[str, ...] | None = None
    object_id: str | None = None
    timepoint: str | None = None

    def __post_init__(self) -> None:
        if self.vector.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {self.vector.shape}")
        if self.names is not None and len(self.names) != self.vector.size:
            raise ValueError(
                f"names has {len(self.names)} entries but vector has {self.vector.size}"
            )

    @property
    def is_trainable(self) -> bool:
        """False until the branch has passed a stability gate."""
        return self.names is None or self.provenance.stability is not None

    def gated(self, keep: Sequence[str], icc: Mapping[str, float]) -> "Encoding":
        """Return a copy restricted to ``keep``, carrying the ICC table in provenance."""
        if self.names is None:
            raise TypeError("cannot gate an unnamed (opaque) encoding by feature name")
        index = {n: i for i, n in enumerate(self.names)}
        missing = [k for k in keep if k not in index]
        if missing:
            raise KeyError(f"{len(missing)} gated features absent from encoding: {missing[:5]}")
        take = [index[k] for k in keep]
        return replace(
            self,
            vector=self.vector[take],
            names=tuple(keep),
            provenance=replace(self.provenance, stability={k: icc[k] for k in keep}),
        )


def check_fusable(encodings: Sequence[Encoding]) -> None:
    """Raise unless all encodings describe the same subject, object, region, and observation."""
    if not encodings:
        return
    subjects = {e.subject_id for e in encodings}
    if len(subjects) > 1:
        raise FusionError(f"encodings span multiple subjects: {sorted(subjects)}")
    subject = next(iter(subjects))
    objects = {e.object_id for e in encodings if e.object_id is not None}
    if len(objects) > 1:
        raise FusionError(
            f"encodings for {subject} span multiple anatomical objects {sorted(objects)}; "
            f"the unit of integration is the object, not the patient"
        )
    scopes = {e.provenance.scope for e in encodings if e.provenance.scope is not None}
    if len(scopes) > 1:
        raise FusionError(
            f"encodings for {subject} span multiple scopes {sorted(scopes)}; combine across "
            f"scopes by ratio (e.g. ctDNA per unit tumor volume), never by concatenation"
        )
    times = {e.timepoint for e in encodings if e.timepoint is not None}
    if len(times) > 1:
        raise FusionError(
            f"encodings for {subject} span multiple timepoints {sorted(times)}; "
            f"observations of one object are a series, not a fusion"
        )
    roles = {e.provenance.roi_role for e in encodings if e.provenance.roi_role is not None}
    if len(roles) > 1:
        raise FusionError(
            f"encodings for {subject} span multiple region roles {sorted(roles)}; a GTV and a "
            f"PTV describe different things and are not fusable"
        )
    rois = {e.provenance.roi for e in encodings if e.provenance.roi is not None}
    if len(rois) > 1:
        offenders = {e.provenance.encoder: e.provenance.roi for e in encodings}
        raise FusionError(f"encodings for {subject} disagree on ROI: {offenders}")


_ICC_TYPE_ALIASES: dict[str, str] = {
    "ICC1": "ICC(1,1)",
    "ICC2": "ICC(A,1)",
    "ICC3": "ICC(C,1)",
    "ICC1k": "ICC(1,k)",
    "ICC2k": "ICC(A,k)",
    "ICC3k": "ICC(C,k)",
}


def _validate_patient_ids(patient_ids: Sequence[str]) -> list[str]:
    """Validate ``patient_ids`` before calling ``feature_icc`` / ``icc_filter``."""
    if patient_ids is None or isinstance(patient_ids, (str, bytes)):
        raise ValueError("patient_ids must be a sequence of patient identifiers")
    raw_list = list(patient_ids)
    if not raw_list:
        raise ValueError("patient_ids must not be empty")

    cleaned: list[str] = []
    for idx, pid in enumerate(raw_list):
        if pid is None:
            raise ValueError(f"patient_ids[{idx}] is None; missing IDs are not permitted")
        if isinstance(pid, (float, np.floating)) and np.isnan(pid):
            raise ValueError(f"patient_ids[{idx}] is NaN; missing IDs are not permitted")
        try:
            import pandas as pd

            if pd.isna(pid):
                raise ValueError(f"patient_ids[{idx}] is NaN; missing IDs are not permitted")
        except ImportError:
            pass
        s = str(pid).strip()
        if not s or s.lower() == "nan":
            raise ValueError(
                f"patient_ids[{idx}]={pid!r} is empty or NaN; missing IDs are not permitted"
            )
        cleaned.append(s)

    unique_pids = set(cleaned)
    if len(unique_pids) < 2:
        raise ValueError(
            f"ICC requires at least 2 unique patients, got {len(unique_pids)}: {sorted(unique_pids)}"
        )
    return cleaned


def gate_encodings_by_icc(
    encodings: Sequence[Encoding],
    feature_df: "pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]",
    patient_ids: Sequence[str],
    threshold: float = 0.75,
    icc_type: str = "ICC3",
) -> list[Encoding]:
    """Validate ``patient_ids``, compute per-feature ICC, and return gated encodings."""
    import pandas as pd
    from qradiomics.analytics.robustness import feature_icc

    cleaned_pids = _validate_patient_ids(patient_ids)
    if not encodings:
        return []

    first_names = encodings[0].names
    if first_names is None:
        raise TypeError("cannot gate unnamed (opaque) encodings by feature ICC")
    feature_cols = list(first_names)

    if isinstance(feature_df, (tuple, list)) and len(feature_df) == 2:
        df_a, df_b = feature_df[0].reset_index(drop=True), feature_df[1].reset_index(drop=True)
        if len(df_a) != len(cleaned_pids) or len(df_b) != len(cleaned_pids):
            raise ValueError(
                f"df_a ({len(df_a)}) and df_b ({len(df_b)}) must match "
                f"patient_ids length ({len(cleaned_pids)})"
            )
        pids_series = pd.Series(cleaned_pids)
    elif isinstance(feature_df, pd.DataFrame):
        if len(feature_df) == len(cleaned_pids):
            grouped: dict[str, list[int]] = {}
            for row_idx, pid in enumerate(cleaned_pids):
                grouped.setdefault(pid, []).append(row_idx)
            paired_pids = [pid for pid, idxs in grouped.items() if len(idxs) >= 2]
            if len(paired_pids) < 2:
                raise ValueError(
                    f"ICC requires at least 2 patients with repeated measurements "
                    f"(>= 2 rows per patient); found {len(paired_pids)}"
                )
            idx_a = [grouped[pid][0] for pid in paired_pids]
            idx_b = [grouped[pid][1] for pid in paired_pids]
            df_a = feature_df.iloc[idx_a].reset_index(drop=True)
            df_b = feature_df.iloc[idx_b].reset_index(drop=True)
            pids_series = pd.Series(paired_pids)
        elif len(feature_df) == 2 * len(cleaned_pids):
            n = len(cleaned_pids)
            df_a = feature_df.iloc[:n].reset_index(drop=True)
            df_b = feature_df.iloc[n:].reset_index(drop=True)
            pids_series = pd.Series(cleaned_pids)
        else:
            raise ValueError(
                f"feature_df length ({len(feature_df)}) does not align with "
                f"patient_ids length ({len(cleaned_pids)})"
            )
    else:
        raise TypeError("feature_df must be a pandas DataFrame or a (df_a, df_b) pair")

    icc_map: dict[str, float] = {}
    keep: list[str] = []
    alias = _ICC_TYPE_ALIASES.get(icc_type)
    for col in feature_cols:
        if col not in df_a.columns or col not in df_b.columns:
            raise KeyError(f"feature {col!r} missing from feature_df")
        s_a = df_a[col].reset_index(drop=True)
        s_b = df_b[col].reset_index(drop=True)
        val = float(feature_icc(s_a, s_b, pids_series, icc_type=icc_type))
        if np.isnan(val) and alias is not None:
            val = float(feature_icc(s_a, s_b, pids_series, icc_type=alias))
        icc_map[col] = val
        if np.isfinite(val) and val >= threshold:
            keep.append(col)

    return [enc.gated(keep, icc_map) for enc in encodings]
