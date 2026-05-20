"""YAML / JSON persistence for qradiomics cohort dataclasses.

These helpers are intentionally tiny — ``save_cohort`` walks the
dataclasses with ``dataclasses.asdict`` plus a small enum/date coercion,
and ``load_cohort`` reconstructs via ``__init__`` keyword arguments. The
goal is a round-tripable on-disk schema, not a versioned migration tool.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from .core import (
    Cohort,
    ImageSeries,
    Patient,
    ROI,
    RTStructureSet,
    Study,
    TreatmentCourse,
)
from .enums import Modality, StudyType

__all__ = ["save_cohort", "load_cohort"]


def save_cohort(cohort: Cohort, path: Union[str, Path]) -> Path:
    """Serialize a Cohort to YAML (``.yml``/``.yaml``) or JSON (``.json``).

    Returns the resolved output path.
    """
    p = Path(path)
    payload = _to_serializable(cohort)
    if p.suffix.lower() in (".yml", ".yaml"):
        with p.open("w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
    elif p.suffix.lower() == ".json":
        with p.open("w") as f:
            json.dump(payload, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported cohort file suffix: {p.suffix!r}")
    return p


def load_cohort(path: Union[str, Path]) -> Cohort:
    """Reconstruct a Cohort from a YAML or JSON file written by :func:`save_cohort`."""
    p = Path(path)
    if p.suffix.lower() in (".yml", ".yaml"):
        with p.open() as f:
            payload = yaml.safe_load(f)
    elif p.suffix.lower() == ".json":
        with p.open() as f:
            payload = json.load(f)
    else:
        raise ValueError(f"Unsupported cohort file suffix: {p.suffix!r}")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping at top level of {p}, got {type(payload).__name__}")
    return _build_cohort(payload)


def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        out: Dict[str, Any] = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            out[f.name] = _to_serializable(value)
        return out
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _build_cohort(d: Dict[str, Any]) -> Cohort:
    return Cohort(
        id=d["id"],
        name=d["name"],
        description=d.get("description", ""),
        patients=[_build_patient(p) for p in d.get("patients", [])],
        created=_maybe_datetime(d.get("created")),
        institution=d.get("institution"),
        metadata=d.get("metadata") or {},
    )


def _build_patient(d: Dict[str, Any]) -> Patient:
    return Patient(
        id=d["id"],
        treatment_courses=[_build_course(c) for c in d.get("treatment_courses", [])],
        studies=[_build_study(s) for s in d.get("studies", [])],
        age_at_enrollment=d.get("age_at_enrollment"),
        sex=d.get("sex"),
        diagnosis=d.get("diagnosis"),
        stage=d.get("stage"),
        histology=d.get("histology"),
        survival_days=d.get("survival_days"),
        event=d.get("event"),
    )


def _build_course(d: Dict[str, Any]) -> TreatmentCourse:
    return TreatmentCourse(
        id=d["id"],
        studies=[_build_study(s) for s in d.get("studies", [])],
        start_date=d.get("start_date"),
        end_date=d.get("end_date"),
        prescription_dose_cgy=d.get("prescription_dose_cgy"),
        dose_per_fraction_cgy=d.get("dose_per_fraction_cgy"),
        num_fractions=d.get("num_fractions"),
        technique=d.get("technique"),
        intent=d.get("intent"),
    )


def _build_study(d: Dict[str, Any]) -> Study:
    return Study(
        id=d["id"],
        study_type=d.get("study_type", StudyType.UNKNOWN.value),
        study_date=d.get("study_date"),
        relative_day=d.get("relative_day"),
        fraction_number=d.get("fraction_number"),
        image_series=[_build_image_series(s) for s in d.get("image_series", [])],
        rt_structures=[_build_rt(s) for s in d.get("rt_structures", [])],
    )


def _build_image_series(d: Dict[str, Any]) -> ImageSeries:
    return ImageSeries(
        id=d["id"],
        modality=d.get("modality", Modality.OTHER.value),
        file_paths=list(d.get("file_paths") or []),
        image_tag=d.get("image_tag"),
        series_description=d.get("series_description"),
        acquisition_date=d.get("acquisition_date"),
        frame_of_reference_uid=d.get("frame_of_reference_uid"),
        pixel_spacing_mm=d.get("pixel_spacing_mm"),
        slice_thickness_mm=d.get("slice_thickness_mm"),
    )


def _build_rt(d: Dict[str, Any]) -> RTStructureSet:
    return RTStructureSet(
        id=d["id"],
        file_path=d["file_path"],
        references_image_series_id=d.get("references_image_series_id"),
        rois=[_build_roi(r) for r in d.get("rois", [])],
    )


def _build_roi(d: Dict[str, Any]) -> ROI:
    return ROI(
        name=d["name"],
        label=d.get("label"),
        roi_type=d.get("roi_type"),
        color=d.get("color"),
        volume_cm3=d.get("volume_cm3"),
    )


def _maybe_datetime(v: Any) -> Any:
    if v is None or isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return v
    return v
