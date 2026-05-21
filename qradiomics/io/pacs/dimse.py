"""DIMSE backend — C-ECHO / C-FIND / C-GET / C-STORE via ``pynetdicom``.

``pynetdicom`` and ``pydicom`` are imported lazily so the rest of the
``qradiomics.io.pacs`` package stays importable on installs that don't
need DIMSE.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    PACSBackend,
    PACSConnectionError,
    PACSError,
    PACSNotFoundError,
    PACSUnsupportedError,
)

logger = logging.getLogger(__name__)


# Curated storage SOP classes negotiated for C-GET. pynetdicom caps a single
# association at 128 presentation contexts, so we can't request every
# AllStoragePresentationContexts entry — keep this list narrow but cover the
# modalities qradiomics actually reads.
_CGET_STORAGE_SOP_CLASSES: Tuple[str, ...] = (
    "1.2.840.10008.5.1.4.1.1.2",        # CT Image Storage
    "1.2.840.10008.5.1.4.1.1.2.1",      # Enhanced CT Image Storage
    "1.2.840.10008.5.1.4.1.1.4",        # MR Image Storage
    "1.2.840.10008.5.1.4.1.1.4.1",      # Enhanced MR Image Storage
    "1.2.840.10008.5.1.4.1.1.128",      # PET Image Storage
    "1.2.840.10008.5.1.4.1.1.130",      # Enhanced PET Image Storage
    "1.2.840.10008.5.1.4.1.1.481.1",    # RT Image Storage
    "1.2.840.10008.5.1.4.1.1.481.2",    # RT Dose Storage
    "1.2.840.10008.5.1.4.1.1.481.3",    # RT Structure Set Storage
    "1.2.840.10008.5.1.4.1.1.481.5",    # RT Plan Storage
    "1.2.840.10008.5.1.4.1.1.481.8",    # RT Ion Plan Storage
    "1.2.840.10008.5.1.4.1.1.481.9",    # RT Treatment Summary Record Storage
    "1.2.840.10008.5.1.4.1.1.66.4",     # Segmentation Storage
    "1.2.840.10008.5.1.4.1.1.7",        # Secondary Capture Image Storage
    "1.2.840.10008.5.1.4.1.1.88.33",    # Comprehensive SR
    "1.2.840.10008.5.1.4.1.1.88.11",    # Basic Text SR
)


_QUERY_KEYS_BY_LEVEL: Dict[str, Tuple[str, ...]] = {
    "PATIENT": ("PatientID", "PatientName", "PatientBirthDate", "PatientSex"),
    "STUDY": (
        "PatientID",
        "StudyInstanceUID",
        "StudyDate",
        "StudyDescription",
        "AccessionNumber",
        "ModalitiesInStudy",
    ),
    "SERIES": (
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SeriesNumber",
        "SeriesDescription",
        "Modality",
        "NumberOfSeriesRelatedInstances",
    ),
    "IMAGE": (
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "SOPClassUID",
        "InstanceNumber",
    ),
}


class DIMSEBackend(PACSBackend):
    """Talk to a classic DIMSE PACS over TCP via ``pynetdicom``.

    Configuration keys:

    ===========  =============================================================
    ``host``     PACS hostname / IP.
    ``port``     PACS port.
    ``aet``      PACS Application Entity Title (called AE).
    ``aec``      Our AE Title (calling AE, default ``QRADIOMICS``).
    ``timeout``  DIMSE/network/ACSE timeout in seconds (default 60).
    ===========  =============================================================

    ``store_dicom`` performs a C-STORE association. Bulk retrieval uses
    C-GET (no separate Move SCP listener required).
    """

    name = "dimse"

    def __init__(
        self,
        name: str = "dimse",
        host: str = "",
        port: int = 0,
        aet: str = "",
        aec: str = "QRADIOMICS",
        timeout: float = 60.0,
        **_: Any,
    ) -> None:
        try:
            import pynetdicom  # noqa: F401
            import pydicom  # noqa: F401
        except ImportError as e:
            raise PACSUnsupportedError(
                "DIMSE backend requires the 'pynetdicom' package "
                "(pip install pynetdicom)."
            ) from e
        if not (host and port and aet):
            raise ValueError(
                "DIMSEBackend: 'host', 'port', and 'aet' are all required"
            )
        self.name = name
        self.host = host
        self.port = int(port)
        self.aet = aet
        self.aec = aec
        self.timeout = float(timeout)

    # ------------------------------------------------------------------ helpers

    def _ae(self):
        from pynetdicom import AE

        ae = AE(ae_title=self.aec)
        ae.dimse_timeout = self.timeout
        ae.network_timeout = self.timeout
        ae.acse_timeout = self.timeout
        return ae

    def _associate(self, ae, evt_handlers: Optional[list] = None):
        kwargs: Dict[str, Any] = {"ae_title": self.aet}
        if evt_handlers is not None:
            kwargs["evt_handlers"] = evt_handlers
        assoc = ae.associate(self.host, self.port, **kwargs)
        if not assoc.is_established:
            raise PACSConnectionError(
                f"DIMSE association rejected by {self.host}:{self.port}"
            )
        return assoc

    @staticmethod
    def _build_query(level: str, filters: Dict[str, Any]):
        import pydicom

        ds = pydicom.Dataset()
        ds.QueryRetrieveLevel = level
        for key in _QUERY_KEYS_BY_LEVEL[level]:
            setattr(ds, key, filters.get(key, ""))
        for key, value in filters.items():
            if value is None:
                continue
            setattr(ds, key, str(value))
        return ds

    @staticmethod
    def _ds_to_dict(ds) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for elem in ds:
            if not elem.keyword:
                continue
            value = elem.value
            if hasattr(value, "original_string"):
                value = str(value)
            out[elem.keyword] = value
        return out

    # ------------------------------------------------------------------ C-FIND

    def _c_find(
        self, level: str, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        from pynetdicom.sop_class import (
            PatientRootQueryRetrieveInformationModelFind,
        )

        ae = self._ae()
        ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
        ds = self._build_query(level, filters)
        assoc = self._associate(ae)
        results: List[Dict[str, Any]] = []
        try:
            for status, identifier in assoc.send_c_find(
                ds,
                query_model=PatientRootQueryRetrieveInformationModelFind,
            ):
                if status is None:
                    raise PACSConnectionError("C-FIND aborted (no status)")
                code = getattr(status, "Status", None)
                if code is None:
                    logger.debug("C-FIND status dataset without Status: %r", status)
                    continue
                if code in (0xFF00, 0xFF01) and identifier is not None:
                    results.append(self._ds_to_dict(identifier))
                elif code == 0x0000:
                    if identifier is not None:
                        results.append(self._ds_to_dict(identifier))
                    break
                else:
                    logger.warning("C-FIND non-success status: 0x%04X", code)
        finally:
            assoc.release()
        return results

    # ------------------------------------------------------------------ API

    def ping(self) -> bool:
        try:
            from pynetdicom.sop_class import Verification

            ae = self._ae()
            ae.add_requested_context(Verification)
            try:
                assoc = self._associate(ae)
            except PACSConnectionError:
                return False
            try:
                status = assoc.send_c_echo()
                return bool(status) and status.Status == 0x0000
            finally:
                assoc.release()
        except PACSError:
            return False

    def query_studies(self, **filters: Any) -> List[Dict[str, Any]]:
        return self._c_find("STUDY", filters)

    def query_series(
        self, study_uid: Optional[str] = None, **filters: Any
    ) -> List[Dict[str, Any]]:
        if study_uid:
            filters["StudyInstanceUID"] = study_uid
        return self._c_find("SERIES", filters)

    def query_instances(
        self, study_uid: str, series_uid: str, **filters: Any
    ) -> List[Dict[str, Any]]:
        filters["StudyInstanceUID"] = study_uid
        filters["SeriesInstanceUID"] = series_uid
        return self._c_find("IMAGE", filters)

    # ------------------------------------------------------------------ C-GET

    def _c_get(self, query_ds, output_dir: Path) -> List[Path]:
        from pynetdicom import build_role, evt
        from pynetdicom.sop_class import (
            PatientRootQueryRetrieveInformationModelGet,
        )

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stored: List[Path] = []

        def on_store(event):
            ds = event.dataset
            ds.file_meta = event.file_meta
            sop = getattr(ds, "SOPInstanceUID", None) or f"inst_{len(stored):05d}"
            path = out_dir / f"{sop}.dcm"
            try:
                ds.save_as(str(path), write_like_original=False)
                stored.append(path)
                return 0x0000
            except Exception as exc:
                logger.error("Failed to save instance %s: %s", sop, exc)
                return 0x0110

        ae = self._ae()
        ae.add_requested_context(PatientRootQueryRetrieveInformationModelGet)
        for sop_uid in _CGET_STORAGE_SOP_CLASSES:
            ae.add_requested_context(sop_uid)
        # C-GET reuses the SCU's association to receive incoming C-STORE
        # sub-operations — the SCP role must be negotiated per storage class.
        roles = [build_role(sop_uid, scp_role=True) for sop_uid in _CGET_STORAGE_SOP_CLASSES]
        assoc = ae.associate(
            self.host,
            self.port,
            ae_title=self.aet,
            ext_neg=roles,
            evt_handlers=[(evt.EVT_C_STORE, on_store)],
        )
        if not assoc.is_established:
            raise PACSConnectionError(
                f"DIMSE association rejected by {self.host}:{self.port}"
            )
        try:
            for status, _ in assoc.send_c_get(
                query_ds,
                query_model=PatientRootQueryRetrieveInformationModelGet,
            ):
                if status is None:
                    break
        finally:
            assoc.release()
        return stored

    def fetch_series(
        self, study_uid: str, series_uid: str, output_dir: Path
    ) -> List[Path]:
        import pydicom

        ds = pydicom.Dataset()
        ds.QueryRetrieveLevel = "SERIES"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        return self._c_get(ds, output_dir)

    def fetch_instance(
        self,
        study_uid: str,
        series_uid: str,
        sop_instance_uid: str,
        output_path: Path,
    ) -> Path:
        import pydicom

        ds = pydicom.Dataset()
        ds.QueryRetrieveLevel = "IMAGE"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = sop_instance_uid
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        paths = self._c_get(ds, out_path.parent)
        target_name = f"{sop_instance_uid}.dcm"
        for path in paths:
            if path.name == target_name and path != out_path:
                path.rename(out_path)
                return out_path
            if path == out_path:
                return out_path
        if paths:
            paths[0].rename(out_path)
            return out_path
        raise PACSNotFoundError(f"Instance {sop_instance_uid} not retrieved")

    def fetch_study(
        self, study_uid: str, output_dir: Path
    ) -> Dict[str, List[Path]]:
        out: Dict[str, List[Path]] = {}
        for row in self.query_series(study_uid=study_uid):
            s_uid = row.get("SeriesInstanceUID")
            if not s_uid:
                continue
            out[s_uid] = self.fetch_series(
                study_uid, s_uid, Path(output_dir) / s_uid
            )
        return out

    # ------------------------------------------------------------------ C-STORE

    def store_dicom(self, dicom_path: Path) -> Dict[str, Any]:
        import pydicom

        ds = pydicom.dcmread(str(dicom_path))
        ae = self._ae()
        ae.add_requested_context(ds.SOPClassUID)
        assoc = self._associate(ae)
        try:
            status = assoc.send_c_store(ds)
            code = int(status.Status) if status else None
            return {
                "status": code,
                "success": code == 0x0000,
                "sop_instance_uid": getattr(ds, "SOPInstanceUID", None),
                "sop_class_uid": getattr(ds, "SOPClassUID", None),
            }
        finally:
            assoc.release()
