"""DICOMweb backend — QIDO-RS / WADO-RS / STOW-RS over HTTP(S)."""

from __future__ import annotations

from email.parser import BytesParser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .base import (
    PACSBackend,
    PACSConnectionError,
    PACSError,
    PACSNotFoundError,
)


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _simplify_dicom_json(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten ``{tag: {vr, Value: [...]}}`` to ``{Keyword: value}``."""
    try:
        from pydicom.datadict import keyword_for_tag
    except ImportError:
        keyword_for_tag = None  # type: ignore[assignment]

    out: Dict[str, Any] = {}
    for tag, payload in record.items():
        if not isinstance(payload, dict):
            continue
        keyword: Optional[str] = None
        if keyword_for_tag is not None:
            try:
                keyword = keyword_for_tag(int(tag, 16)) or None
            except (ValueError, TypeError):
                keyword = None
        key = keyword or tag
        value = payload.get("Value")
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "Alphabetic" in value[0]
        ):
            out[key] = value[0]["Alphabetic"]
        else:
            out[key] = _first(value)
    return out


def _parse_multipart_dicom(body: bytes, content_type: str) -> List[bytes]:
    """Extract every ``application/dicom`` part from a multipart/related body."""
    header = f"Content-Type: {content_type}\r\n\r\n".encode()
    msg = BytesParser().parsebytes(header + body)
    parts: List[bytes] = []
    for part in msg.walk():
        if (
            part.get_content_maintype() == "application"
            and part.get_content_subtype() == "dicom"
        ):
            payload = part.get_payload(decode=True)
            if isinstance(payload, (bytes, bytearray)):
                parts.append(bytes(payload))
    return parts


class DICOMwebBackend(PACSBackend):
    """Talk to any DICOMweb server (Orthanc, dcm4chee, Google HCAPI, …).

    Configuration keys (``base_url`` required, rest optional):

    ============  ============================================================
    ``base_url``  DICOMweb root, e.g. ``https://pacs.example.com/dicom-web``.
    ``auth_token``  Bearer token (sent as ``Authorization: Bearer …``).
    ``username``  HTTP basic-auth user (alternative to ``auth_token``).
    ``password``  HTTP basic-auth password.
    ``verify_ssl``  Verify TLS (default True).
    ``timeout``   Request timeout in seconds (default 60).
    ``qido_prefix``  Optional sub-path for QIDO-RS endpoints.
    ``wado_prefix``  Optional sub-path for WADO-RS endpoints.
    ``stow_prefix``  Optional sub-path for STOW-RS endpoints.
    ============  ============================================================
    """

    name = "dicomweb"

    def __init__(
        self,
        name: str = "dicomweb",
        base_url: str = "",
        auth_token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_ssl: bool = True,
        timeout: float = 60.0,
        qido_prefix: str = "",
        wado_prefix: str = "",
        stow_prefix: str = "",
        **_: Any,
    ) -> None:
        if not base_url:
            raise ValueError("DICOMwebBackend: 'base_url' is required")
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.qido_prefix = qido_prefix.strip("/")
        self.wado_prefix = wado_prefix.strip("/")
        self.stow_prefix = stow_prefix.strip("/")

        headers: Dict[str, str] = {"Accept": "application/dicom+json"}
        auth: Optional[tuple] = None
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        elif username is not None:
            auth = (username, password or "")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            auth=auth,
            timeout=httpx.Timeout(float(timeout)),
            verify=verify_ssl,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ helpers

    def _path(self, prefix: str, rest: str) -> str:
        rest = rest.lstrip("/")
        return f"/{prefix}/{rest}" if prefix else f"/{rest}"

    def _qido(self, rest: str, **params: Any) -> List[Dict[str, Any]]:
        path = self._path(self.qido_prefix, rest)
        clean = {k: v for k, v in params.items() if v is not None}
        try:
            r = self._client.get(path, params=clean)
        except httpx.ConnectError as e:
            raise PACSConnectionError(f"Cannot reach DICOMweb: {e}") from e
        if r.status_code == 204:
            return []
        if r.status_code == 404:
            raise PACSNotFoundError(f"Not found: {path}")
        if r.status_code >= 400:
            raise PACSError(f"DICOMweb HTTP {r.status_code}: {r.text[:200]}")
        if not r.text:
            return []
        return [_simplify_dicom_json(rec) for rec in r.json()]

    def _wado(
        self,
        rest: str,
        *,
        accept: str = 'multipart/related; type="application/dicom"',
    ) -> httpx.Response:
        path = self._path(self.wado_prefix, rest)
        try:
            r = self._client.get(path, headers={"Accept": accept})
        except httpx.ConnectError as e:
            raise PACSConnectionError(f"Cannot reach DICOMweb: {e}") from e
        if r.status_code == 404:
            raise PACSNotFoundError(f"Not found: {path}")
        if r.status_code >= 400:
            raise PACSError(f"DICOMweb HTTP {r.status_code}: {r.text[:200]}")
        return r

    # ------------------------------------------------------------------ API

    def ping(self) -> bool:
        try:
            self._qido("studies", limit="1")
            return True
        except PACSError:
            return False

    def query_studies(self, **filters: Any) -> List[Dict[str, Any]]:
        return self._qido("studies", **filters)

    def query_series(
        self, study_uid: Optional[str] = None, **filters: Any
    ) -> List[Dict[str, Any]]:
        if study_uid:
            return self._qido(f"studies/{study_uid}/series", **filters)
        return self._qido("series", **filters)

    def query_instances(
        self, study_uid: str, series_uid: str, **filters: Any
    ) -> List[Dict[str, Any]]:
        return self._qido(
            f"studies/{study_uid}/series/{series_uid}/instances", **filters
        )

    # ------------------------------------------------------------------ fetch

    @staticmethod
    def _split_response(response: httpx.Response) -> List[bytes]:
        ctype = response.headers.get("Content-Type", "")
        if ctype.startswith("multipart/"):
            parts = _parse_multipart_dicom(response.content, ctype)
            if not parts:
                raise PACSError("DICOMweb returned an empty multipart body")
            return parts
        return [response.content]

    def fetch_instance(
        self,
        study_uid: str,
        series_uid: str,
        sop_instance_uid: str,
        output_path: Path,
    ) -> Path:
        r = self._wado(
            f"studies/{study_uid}/series/{series_uid}/instances/{sop_instance_uid}"
        )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self._split_response(r)[0])
        return out

    def fetch_series(
        self, study_uid: str, series_uid: str, output_dir: Path
    ) -> List[Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        r = self._wado(f"studies/{study_uid}/series/{series_uid}")
        written: List[Path] = []
        for idx, blob in enumerate(self._split_response(r)):
            sop = self._extract_sop_uid(blob, fallback=f"inst_{idx:05d}")
            path = out_dir / f"{sop}.dcm"
            path.write_bytes(blob)
            written.append(path)
        return written

    def fetch_study(
        self, study_uid: str, output_dir: Path
    ) -> Dict[str, List[Path]]:
        r = self._wado(f"studies/{study_uid}")
        out: Dict[str, List[Path]] = {}
        for idx, blob in enumerate(self._split_response(r)):
            series_uid, sop = self._extract_series_and_sop(
                blob, fallback_idx=idx
            )
            series_dir = Path(output_dir) / series_uid
            series_dir.mkdir(parents=True, exist_ok=True)
            path = series_dir / f"{sop}.dcm"
            path.write_bytes(blob)
            out.setdefault(series_uid, []).append(path)
        return out

    @staticmethod
    def _extract_sop_uid(blob: bytes, *, fallback: str) -> str:
        try:
            from pydicom import dcmread

            ds = dcmread(BytesIO(blob), stop_before_pixels=True)
            return str(getattr(ds, "SOPInstanceUID", fallback))
        except Exception:
            return fallback

    @staticmethod
    def _extract_series_and_sop(blob: bytes, *, fallback_idx: int) -> tuple[str, str]:
        try:
            from pydicom import dcmread

            ds = dcmread(BytesIO(blob), stop_before_pixels=True)
            series = str(getattr(ds, "SeriesInstanceUID", f"series_{fallback_idx}"))
            sop = str(getattr(ds, "SOPInstanceUID", f"inst_{fallback_idx:05d}"))
            return series, sop
        except Exception:
            return f"series_{fallback_idx}", f"inst_{fallback_idx:05d}"

    # ------------------------------------------------------------------ store

    def store_dicom(self, dicom_path: Path) -> Dict[str, Any]:
        boundary = "qradiomicsSTOW"
        data = Path(dicom_path).read_bytes()
        body = (
            f"--{boundary}\r\n"
            "Content-Type: application/dicom\r\n"
            "\r\n"
        ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
        path = self._path(self.stow_prefix, "studies")
        try:
            r = self._client.post(
                path,
                content=body,
                headers={
                    "Content-Type": (
                        f'multipart/related; type="application/dicom"; boundary={boundary}'
                    ),
                    "Accept": "application/dicom+json",
                },
            )
        except httpx.ConnectError as e:
            raise PACSConnectionError(f"Cannot reach DICOMweb: {e}") from e
        if r.status_code >= 400:
            raise PACSError(f"STOW HTTP {r.status_code}: {r.text[:200]}")
        return r.json() if r.text else {"status": r.status_code}
