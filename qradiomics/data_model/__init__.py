"""qradiomics.data_model — minimum cohort/patient/course/study dataclasses.

Mirrors the JLR hierarchy used at the atomic layer. See
:mod:`qradiomics.data_model.core` for the field-level contract and
[[wiki/architecture/ATOMIC_UNIT_AND_STAGES]] §1.1 for the conceptual
model. JLR's richer dataclasses (motion phase, processing status,
quality grade, …) stay on the JLR side until a concrete consumer needs
them.
"""

from .core import (
    Cohort,
    ImageSeries,
    Patient,
    ROI,
    RTStructureSet,
    Study,
    TreatmentCourse,
    iter_studies,
)
from .enums import Modality, StudyType
from .io import load_cohort, save_cohort

__all__ = [
    "Cohort",
    "ImageSeries",
    "Modality",
    "Patient",
    "ROI",
    "RTStructureSet",
    "Study",
    "StudyType",
    "TreatmentCourse",
    "iter_studies",
    "load_cohort",
    "save_cohort",
]
