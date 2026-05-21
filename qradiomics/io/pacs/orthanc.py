"""Orthanc REST API backend (sync, httpx)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .base import (
    PACSBackend,
    PACSConnectionError,
    PACSError,
    PACSNotFoundError,
)


class OrthancBackend(PACSBackend):
    """Talk to an Orthanc PACS over its REST API.

    Configuration keys (all optional, env-falling-back where noted):

    ============  ===========================================================
    ``base_url``  Server root, default ``$ORTHANC_URL`` or
                  ``http://localhost:8042``.
    ``username``  HTTP basic-auth user, default ``$ORTHANC_USERNAME`` or
                  ``orthanc``.
    ``password``  HTTP basic-auth password, default ``$ORTHANC_PASSWORD``.
    ``timeout``   Request timeout in seconds (default 60).
    ``verify_ssl``  Verify TLS certificates (default True).
    ============  ===========================================================
    """

    name = "orthanc"

    def __init__(
        self,
        name: str = "orthanc",
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 60.0,
        verify_ssl: bool = True,
        **_: Any,
    ) -> None:
        self.name = name
        self.base_url = (
            base_url
            or os.environ.get("ORTHANC_URL")
            or "http://localhost:8042"
        )
        user = username if username is not None else os.environ.get("ORTHANC_USERNAME", "orthanc")
        pwd = password if password is not None else os.environ.get("ORTHANC_PASSWORD", "")
        auth = (user, pwd) if user else None
        self._client = httpx.Client(
            base_url=self.base_url,
            auth=auth,
            timeout=httpx.Timeout(float(timeout)),
            verify=verify_ssl,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ HTTP

    def _get(self, path: str) -> httpx.Response:
        try:
            r = self._client.get(path)
        except httpx.ConnectError as e:
            raise PACSConnectionError(f"Cannot reach Orthanc: {e}") from e
        if r.status_code == 404:
            raise PACSNotFoundError(f"Not found: {path}")
        if r.status_code >= 400:
            raise PACSError(f"Orthanc HTTP {r.status_code}: {r.text[:200]}")
        return r

    def _post(
        self,
        path: str,
        *,
        json_data: Any = None,
        content: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        try:
            r = self._client.post(
                path, json=json_data, content=content, headers=headers
            )
        except httpx.ConnectError as e:
            raise PACSConnectionError(f"Cannot reach Orthanc: {e}") from e
        if r.status_code >= 400:
            raise PACSError(f"Orthanc HTTP {r.status_code}: {r.text[:200]}")
        return r

    def _find(
        self, level: str, query: Dict[str, str], limit: int = 1000
    ) -> List[Dict[str, Any]]:
        body = {"Level": level, "Query": query, "Expand": True, "Limit": int(limit)}
        r = self._post("/tools/find", json_data=body)
        return r.json()

    # ------------------------------------------------------------------ flatteners

    @staticmethod
    def _flatten_study(study: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **study.get("PatientMainDicomTags", {}),
            **study.get("MainDicomTags", {}),
            "NumberOfStudyRelatedSeries": len(study.get("Series", [])) or None,
            "_OrthancStudyID": study.get("ID"),
        }

    @staticmethod
    def _flatten_series(series: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **series.get("MainDicomTags", {}),
            "NumberOfSeriesRelatedInstances": len(series.get("Instances", [])) or None,
            "_OrthancSeriesID": series.get("ID"),
            "_OrthancParentStudy": series.get("ParentStudy"),
        }

    @staticmethod
    def _flatten_instance(inst: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **inst.get("MainDicomTags", {}),
            "_OrthancInstanceID": inst.get("ID"),
            "_OrthancParentSeries": inst.get("ParentSeries"),
        }

    # ------------------------------------------------------------------ API

    def ping(self) -> bool:
        try:
            return self._get("/system").status_code == 200
        except PACSError:
            return False

    def query_studies(self, **filters: Any) -> List[Dict[str, Any]]:
        limit = int(filters.pop("limit", 1000))
        query = {k: str(v) for k, v in filters.items() if v is not None}
        return [self._flatten_study(s) for s in self._find("Study", query, limit)]

    def query_series(
        self, study_uid: Optional[str] = None, **filters: Any
    ) -> List[Dict[str, Any]]:
        limit = int(filters.pop("limit", 1000))
        query = {k: str(v) for k, v in filters.items() if v is not None}
        if study_uid:
            query["StudyInstanceUID"] = study_uid
        return [self._flatten_series(s) for s in self._find("Series", query, limit)]

    def query_instances(
        self, study_uid: str, series_uid: str, **filters: Any
    ) -> List[Dict[str, Any]]:
        limit = int(filters.pop("limit", 10000))
        query = {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,
        }
        query.update({k: str(v) for k, v in filters.items() if v is not None})
        return [self._flatten_instance(i) for i in self._find("Instance", query, limit)]

    # ------------------------------------------------------------------ fetch

    def _resolve_series_id(self, study_uid: str, series_uid: str) -> str:
        rows = self._find(
            "Series",
            {"StudyInstanceUID": study_uid, "SeriesInstanceUID": series_uid},
            limit=1,
        )
        if not rows:
            raise PACSNotFoundError(
                f"Series {series_uid} not found in study {study_uid}"
            )
        return rows[0]["ID"]

    def _resolve_instance_id(
        self, study_uid: str, series_uid: str, sop_uid: str
    ) -> str:
        rows = self._find(
            "Instance",
            {
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "SOPInstanceUID": sop_uid,
            },
            limit=1,
        )
        if not rows:
            raise PACSNotFoundError(f"Instance {sop_uid} not found")
        return rows[0]["ID"]

    def fetch_instance(
        self,
        study_uid: str,
        series_uid: str,
        sop_instance_uid: str,
        output_path: Path,
    ) -> Path:
        inst_id = self._resolve_instance_id(study_uid, series_uid, sop_instance_uid)
        r = self._get(f"/instances/{inst_id}/file")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
        return out

    def fetch_series(
        self, study_uid: str, series_uid: str, output_dir: Path
    ) -> List[Path]:
        series_id = self._resolve_series_id(study_uid, series_uid)
        meta = self._get(f"/series/{series_id}").json()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for inst_id in meta.get("Instances", []):
            r = self._get(f"/instances/{inst_id}/file")
            path = out_dir / f"{inst_id}.dcm"
            path.write_bytes(r.content)
            written.append(path)
        return written

    def fetch_study(
        self, study_uid: str, output_dir: Path
    ) -> Dict[str, List[Path]]:
        studies = self._find("Study", {"StudyInstanceUID": study_uid}, limit=1)
        if not studies:
            raise PACSNotFoundError(f"Study {study_uid} not found")
        study_meta = self._get(f"/studies/{studies[0]['ID']}").json()
        out: Dict[str, List[Path]] = {}
        for sid in study_meta.get("Series", []):
            series_meta = self._get(f"/series/{sid}").json()
            s_uid = (
                series_meta.get("MainDicomTags", {}).get("SeriesInstanceUID")
                or sid
            )
            series_dir = Path(output_dir) / s_uid
            series_dir.mkdir(parents=True, exist_ok=True)
            for inst_id in series_meta.get("Instances", []):
                r = self._get(f"/instances/{inst_id}/file")
                path = series_dir / f"{inst_id}.dcm"
                path.write_bytes(r.content)
                out.setdefault(s_uid, []).append(path)
        return out

    # ------------------------------------------------------------------ store

    def store_dicom(self, dicom_path: Path) -> Dict[str, Any]:
        data = Path(dicom_path).read_bytes()
        r = self._post(
            "/instances",
            content=data,
            headers={"Content-Type": "application/dicom"},
        )
        return r.json() if r.text else {"status": r.status_code}

    # ------------------------------------------------------------------ remote modalities

    def list_modalities(self) -> List[str]:
        """List remote modalities registered on this Orthanc."""
        return self._get("/modalities").json()

    def get_modality_config(self, modality: str) -> Dict[str, Any]:
        """Return host/port/AET config for a registered modality."""
        return self._get(f"/modalities/{modality}/configuration").json()

    def echo_modality(self, modality: str) -> bool:
        """C-ECHO a remote modality through Orthanc."""
        try:
            self._post(f"/modalities/{modality}/echo", json_data={})
            return True
        except PACSError:
            return False

    def query_remote(
        self,
        modality: str,
        level: str = "Study",
        **query: Any,
    ) -> Dict[str, Any]:
        """Issue a C-FIND against a remote modality via Orthanc.

        Returns Orthanc's query handle dict — pass it (or its ``ID``) to
        :meth:`retrieve_query` to trigger the C-MOVE step.
        """
        clean = {k: str(v) for k, v in query.items() if v is not None}
        body = {"Level": level, "Query": clean}
        r = self._post(f"/modalities/{modality}/query", json_data=body)
        return r.json()

    def query_answers(self, query_id: str) -> List[Dict[str, Any]]:
        """Return the answer rows for a previously issued remote query."""
        ids = self._get(f"/queries/{query_id}/answers").json()
        answers: List[Dict[str, Any]] = []
        for ans_id in ids:
            data = self._get(
                f"/queries/{query_id}/answers/{ans_id}/content?simplify"
            ).json()
            answers.append(data)
        return answers

    def retrieve_query(
        self,
        query_id: str,
        target_aet: Optional[str] = None,
        synchronous: bool = True,
    ) -> Dict[str, Any]:
        """Trigger a C-MOVE for every answer of ``query_id``.

        ``target_aet`` defaults to Orthanc's own ``DicomAet`` (so the
        remote PACS sends instances back here). Set ``synchronous=False``
        to return immediately with a job handle.
        """
        body: Dict[str, Any] = {"Synchronous": bool(synchronous)}
        if target_aet:
            body["TargetAet"] = target_aet
        r = self._post(f"/queries/{query_id}/retrieve", json_data=body)
        return r.json() if r.text else {"status": r.status_code}

    def retrieve_from(
        self,
        modality: str,
        level: str = "Study",
        target_aet: Optional[str] = None,
        synchronous: bool = True,
        **query: Any,
    ) -> Dict[str, Any]:
        """One-shot remote retrieve: query + retrieve.

        Convenience for ``query_remote`` followed by ``retrieve_query``.
        Returns ``{"query_id", "answers", "retrieve"}``.
        """
        handle = self.query_remote(modality, level=level, **query)
        qid = handle.get("ID") or handle.get("Path", "").rsplit("/", 1)[-1]
        if not qid:
            raise PACSError(f"Orthanc returned no query id: {handle}")
        answers = self.query_answers(qid)
        retrieve = self.retrieve_query(
            qid, target_aet=target_aet, synchronous=synchronous
        )
        return {"query_id": qid, "answers": answers, "retrieve": retrieve}

    # ------------------------------------------------------------------ changes feed

    def get_changes(
        self,
        since: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Read the Orthanc changes feed.

        Returns ``{"Changes": [...], "Done": bool, "Last": int}``. Used to
        detect new instances/series/studies pushed in by an external SCU
        (e.g. MIM Assistant).
        """
        r = self._get(f"/changes?since={int(since)}&limit={int(limit)}")
        return r.json()
