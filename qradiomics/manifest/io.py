"""Manifest CSV read/write + cohort flattening."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from ..data_model import Cohort, Patient, RTStructureSet, Study, TreatmentCourse

__all__ = ["MANIFEST_COLUMNS", "flatten_cohort", "read_manifest", "write_manifest"]


# Canonical column order. Workflow runners depend on these names — keep
# them stable. Extra columns are allowed but unrecognised names will be
# preserved verbatim through read/write but ignored by built-in
# consumers.
MANIFEST_COLUMNS = (
    "patient_id",
    "course_id",
    "study_id",
    "timepoint",
    "image_series_id",
    "image_path",
    "image_tag",
    "modality",
    "rt_id",
    "mask_path",
    "mask_tag",
    "mask_label",
    "mask_image_tag",  # which image the mask is meant for — supports
                       # re-using one RTSTRUCT against multiple CBCT
                       # timepoints when the user explicitly tags it.
)


def flatten_cohort(cohort: Cohort) -> List[Dict[str, Any]]:
    """Cross-product of (Patient × [Course] × Study × ImageSeries × ROI).

    Each row pairs an ImageSeries with every ROI from an RTStructureSet
    inside the same Study. If a Study has multiple RT structure sets,
    every ROI from every set is emitted. Studies without an
    RTStructureSet emit one row per ImageSeries with mask fields blank.
    """
    rows: List[Dict[str, Any]] = []
    for patient in cohort.patients:
        if patient.treatment_courses:
            for course in patient.treatment_courses:
                for study in course.studies:
                    rows.extend(_flatten_study(patient, course, study))
        for study in patient.studies:
            rows.extend(_flatten_study(patient, None, study))
    return rows


def _flatten_study(
    patient: Patient,
    course: Optional[TreatmentCourse],
    study: Study,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    timepoint = study.id
    if study.relative_day is not None:
        timepoint = f"day{study.relative_day:+d}"
    elif study.fraction_number is not None:
        timepoint = f"fx{study.fraction_number}"
    elif study.study_date is not None:
        timepoint = study.study_date.isoformat()

    if not study.image_series:
        return rows

    matched_rts = _index_rt_by_image(study.rt_structures)

    for img in study.image_series:
        rts: List[RTStructureSet] = matched_rts.get(img.id, []) + matched_rts.get("__any__", [])
        if not rts:
            rows.append(
                _row(patient, course, study, timepoint, img, rt=None, roi=None)
            )
            continue
        for rt in rts:
            if not rt.rois:
                rows.append(_row(patient, course, study, timepoint, img, rt=rt, roi=None))
            else:
                for roi in rt.rois:
                    rows.append(_row(patient, course, study, timepoint, img, rt=rt, roi=roi))
    return rows


def _index_rt_by_image(
    rt_structures: Iterable[RTStructureSet],
) -> Dict[str, List[RTStructureSet]]:
    """Group RT structures by the image they reference.

    Structures whose ``references_image_series_id`` is None are listed
    under the sentinel key ``"__any__"`` and apply to every image in
    the study.
    """
    out: Dict[str, List[RTStructureSet]] = {}
    for rt in rt_structures:
        key = rt.references_image_series_id or "__any__"
        out.setdefault(key, []).append(rt)
    return out


def _row(
    patient: Patient,
    course: Optional[TreatmentCourse],
    study: Study,
    timepoint: str,
    img,
    rt,
    roi,
) -> Dict[str, Any]:
    return {
        "patient_id": patient.id,
        "course_id": course.id if course else "",
        "study_id": study.id,
        "timepoint": timepoint,
        "image_series_id": img.id,
        "image_path": img.file_paths[0] if img.file_paths else "",
        "image_tag": img.image_tag or "",
        "modality": img.modality.value if hasattr(img.modality, "value") else str(img.modality),
        "rt_id": rt.id if rt else "",
        "mask_path": rt.file_path if rt else "",
        "mask_tag": roi.name if roi else "",
        "mask_label": roi.label if (roi and roi.label is not None) else "",
        "mask_image_tag": img.image_tag or "",
    }


def write_manifest(rows: List[Dict[str, Any]], path: Union[str, Path]) -> Path:
    """Write rows to a CSV file with the canonical column order plus any extras.

    Extra keys present in ``rows`` are written after the canonical
    columns, in the order they first appear.
    """
    p = Path(path)
    extra_cols: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in MANIFEST_COLUMNS and k not in extra_cols:
                extra_cols.append(k)
    fieldnames = list(MANIFEST_COLUMNS) + extra_cols
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in fieldnames})
    return p


def read_manifest(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a manifest CSV into a list of plain dict rows."""
    p = Path(path)
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
