"""PACS backend abstraction.

The ABC defines a small, backend-neutral surface for querying metadata and
moving DICOM objects in/out of any PACS — Orthanc REST, DICOMweb, or DIMSE.
All implementations return plain ``dict`` rows with simplified DICOM keyword
keys (``PatientID``, ``StudyInstanceUID``, ``Modality`` …) so calling code
does not need to know which backend produced them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class PACSError(Exception):
    """Base class for all PACS backend errors."""


class PACSConnectionError(PACSError):
    """Network/association failed."""


class PACSNotFoundError(PACSError):
    """Requested DICOM object was not found."""


class PACSUnsupportedError(PACSError):
    """Backend does not support the requested operation."""


# Keys backends should populate on each row when the server provides them.
STUDY_KEYS = (
    "PatientID",
    "PatientName",
    "StudyInstanceUID",
    "StudyDate",
    "StudyDescription",
    "AccessionNumber",
    "ModalitiesInStudy",
    "NumberOfStudyRelatedSeries",
    "NumberOfStudyRelatedInstances",
)
SERIES_KEYS = (
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesNumber",
    "SeriesDescription",
    "Modality",
    "SeriesDate",
    "BodyPartExamined",
    "NumberOfSeriesRelatedInstances",
)
INSTANCE_KEYS = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SOPClassUID",
    "InstanceNumber",
)


class PACSBackend(ABC):
    """Backend-neutral PACS access.

    Implementations override the abstract methods; default implementations
    of :meth:`fetch_study` and :meth:`store_directory` compose the lower-
    level operations.
    """

    #: Human-readable backend kind ("orthanc" | "dicomweb" | "dimse").
    name: str = "abstract"

    # ------------------------------------------------------------------ connectivity

    @abstractmethod
    def ping(self) -> bool:
        """Verify the PACS responds (HTTP /system, QIDO probe, or C-ECHO)."""

    # ------------------------------------------------------------------ query

    @abstractmethod
    def query_studies(self, **filters: Any) -> List[Dict[str, Any]]:
        """Return studies matching simplified DICOM keyword filters."""

    @abstractmethod
    def query_series(
        self, study_uid: Optional[str] = None, **filters: Any
    ) -> List[Dict[str, Any]]:
        """Return series; constrain to one study with ``study_uid``."""

    @abstractmethod
    def query_instances(
        self, study_uid: str, series_uid: str, **filters: Any
    ) -> List[Dict[str, Any]]:
        """Return instances in a series."""

    # ------------------------------------------------------------------ fetch

    @abstractmethod
    def fetch_instance(
        self,
        study_uid: str,
        series_uid: str,
        sop_instance_uid: str,
        output_path: Path,
    ) -> Path:
        """Download a single SOP instance to ``output_path``."""

    @abstractmethod
    def fetch_series(
        self, study_uid: str, series_uid: str, output_dir: Path
    ) -> List[Path]:
        """Download every instance of a series into ``output_dir``."""

    def fetch_study(
        self, study_uid: str, output_dir: Path
    ) -> Dict[str, List[Path]]:
        """Download a whole study, grouped by ``SeriesInstanceUID``.

        Default implementation calls :meth:`query_series` then
        :meth:`fetch_series`; backends with a bulk endpoint should override.
        """
        out: Dict[str, List[Path]] = {}
        for row in self.query_series(study_uid=study_uid):
            s_uid = row.get("SeriesInstanceUID")
            if not s_uid:
                continue
            series_dir = Path(output_dir) / s_uid
            out[s_uid] = self.fetch_series(study_uid, s_uid, series_dir)
        return out

    # ------------------------------------------------------------------ store

    @abstractmethod
    def store_dicom(self, dicom_path: Path) -> Dict[str, Any]:
        """Upload a single DICOM file (STOW / Orthanc /instances / C-STORE)."""

    def store_directory(
        self, directory: Path, recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """Upload every ``*.dcm`` under ``directory``."""
        pattern = "**/*.dcm" if recursive else "*.dcm"
        return [self.store_dicom(p) for p in sorted(Path(directory).glob(pattern))]

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Release any held resources (HTTP client, association)."""

    def __enter__(self) -> "PACSBackend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
