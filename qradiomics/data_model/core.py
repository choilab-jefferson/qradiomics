"""Minimum cohort dataclasses for the qradiomics atomic data model.

Mirrors the JLR hierarchy
``Cohort → Patient → TreatmentCourse → Study → (ImageSeries | RTStructureSet)
  → ROI`` but carries only the fields the atomic layer reads. Each level is a
plain dataclass with native dicts/lists for collections, so YAML/JSON
serialization is direct.

Diagnostic cohorts (TCIA-style: lung1, ACRIN, …) omit the
``TreatmentCourse`` layer; in that case the ``Patient.studies`` shortcut
holds studies directly. Either path is legal — see
[[wiki/architecture/ATOMIC_UNIT_AND_STAGES]] §1.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

from .enums import Modality, StudyType

__all__ = [
    "ROI",
    "ImageSeries",
    "RTStructureSet",
    "Study",
    "TreatmentCourse",
    "Patient",
    "Cohort",
]


@dataclass
class ROI:
    """Single contour inside an RTStructureSet."""

    name: str
    label: Optional[int] = None  # Mask label value used in the rasterized mask
    roi_type: Optional[str] = None  # GTV / CTV / PTV / ITV / OAR / OTHER (free string)
    color: Optional[str] = None
    volume_cm3: Optional[float] = None


@dataclass
class ImageSeries:
    """One image volume — CT/CBCT/MR/PET. Paths point to either DICOM
    series directories, NRRD files, or NIfTI files."""

    id: str
    modality: Modality
    file_paths: List[str] = field(default_factory=list)
    # Free-text tag the atomic dispatcher uses to disambiguate timepoints
    # (e.g. "planning", "cbct-w1", "post-tx") without needing a richer enum.
    image_tag: Optional[str] = None
    series_description: Optional[str] = None
    acquisition_date: Optional[str] = None  # YYYYMMDD or ISO date
    frame_of_reference_uid: Optional[str] = None
    pixel_spacing_mm: Optional[List[float]] = None
    slice_thickness_mm: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.modality, str):
            self.modality = Modality(self.modality)
        self.file_paths = [str(p) for p in self.file_paths]


@dataclass
class RTStructureSet:
    """RTSTRUCT or seg.nrrd referencing one ImageSeries."""

    id: str
    file_path: str
    references_image_series_id: Optional[str] = None
    rois: List[ROI] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.file_path = str(self.file_path)


@dataclass
class Study:
    """One imaging session — Plan CT, CBCT-week1, PET-followup, etc."""

    id: str
    study_type: StudyType = StudyType.UNKNOWN
    study_date: Optional[date] = None
    relative_day: Optional[int] = None  # Days from treatment start
    fraction_number: Optional[int] = None
    image_series: List[ImageSeries] = field(default_factory=list)
    rt_structures: List[RTStructureSet] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.study_type, str):
            self.study_type = StudyType(self.study_type)
        if isinstance(self.study_date, str):
            self.study_date = _parse_date(self.study_date)


@dataclass
class TreatmentCourse:
    """One episode of radiotherapy care (curative, re-irradiation, …)."""

    id: str
    studies: List[Study] = field(default_factory=list)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    prescription_dose_cgy: Optional[float] = None  # cGy
    dose_per_fraction_cgy: Optional[float] = None
    num_fractions: Optional[int] = None
    technique: Optional[str] = None
    intent: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.start_date, str):
            self.start_date = _parse_date(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = _parse_date(self.end_date)


@dataclass
class Patient:
    """De-identified subject. Either ``treatment_courses`` (longitudinal RT
    cohort) or ``studies`` (diagnostic-only cohort) is populated; using
    both at once is supported but discouraged."""

    id: str
    treatment_courses: List[TreatmentCourse] = field(default_factory=list)
    studies: List[Study] = field(default_factory=list)
    age_at_enrollment: Optional[int] = None
    sex: Optional[str] = None  # 'M', 'F', 'U'
    diagnosis: Optional[str] = None
    stage: Optional[str] = None
    histology: Optional[str] = None
    survival_days: Optional[float] = None
    event: Optional[int] = None  # 1 = death, 0 = censored


@dataclass
class Cohort:
    """Top-level grouping for a research study."""

    id: str
    name: str
    description: str = ""
    patients: List[Patient] = field(default_factory=list)
    created: Optional[datetime] = None
    institution: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.created, str):
            self.created = datetime.fromisoformat(self.created)


def iter_studies(cohort: Cohort) -> List[tuple[Patient, Optional[TreatmentCourse], Study]]:
    """Flatten a cohort into ``(patient, course_or_None, study)`` triples."""
    rows: List[tuple[Patient, Optional[TreatmentCourse], Study]] = []
    for patient in cohort.patients:
        for course in patient.treatment_courses:
            for study in course.studies:
                rows.append((patient, course, study))
        for study in patient.studies:
            rows.append((patient, None, study))
    return rows


def _parse_date(value: Union[str, date]) -> date:
    if isinstance(value, date):
        return value
    s = str(value)
    if len(s) == 8 and s.isdigit():
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    return date.fromisoformat(s)
