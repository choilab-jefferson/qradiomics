"""Minimum enums for the qradiomics atomic data model.

Kept as plain ``str`` Enums so YAML/JSON serialization round-trips
naturally and external code can pass strings interchangeably with enum
members. Only the values actually consulted by the atomic layer and the
manifest are defined here; the JLR set is much larger (motion phase,
quality grade, processing status …) and stays on the JLR side until a
specific consumer needs it.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Modality", "StudyType"]


class Modality(str, Enum):
    """Imaging modality. Values match DICOM (0008,0060) where applicable."""

    CT = "CT"
    CBCT = "CBCT"  # not DICOM-standard; treated as a CT variant by qradiomics
    MR = "MR"
    PT = "PT"  # PET
    NM = "NM"
    US = "US"
    XA = "XA"
    DX = "DX"
    OTHER = "OTHER"


class StudyType(str, Enum):
    """Coarse study category used by the manifest and stage dispatch."""

    PLANNING_CT = "PLANNING_CT"
    ON_TREATMENT_CBCT = "ON_TREATMENT_CBCT"
    DIAGNOSTIC = "DIAGNOSTIC"
    FOLLOWUP = "FOLLOWUP"
    UNKNOWN = "UNKNOWN"
