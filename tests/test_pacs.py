"""Tests for ``qradiomics.io.pacs`` — config loader + Orthanc & DICOMweb backends.

Network access is mocked via :class:`httpx.MockTransport`; the DIMSE backend
needs ``pynetdicom`` and a live association, so it isn't covered here.
"""
from __future__ import annotations

import json

import httpx
import pytest

from qradiomics.io import pacs
from qradiomics.io.pacs.config import _expand_env, load_profile, load_profiles
from qradiomics.io.pacs.dicomweb import (
    DICOMwebBackend,
    _simplify_dicom_json,
)
from qradiomics.io.pacs.orthanc import OrthancBackend


# ---------------------------------------------------------------------------- config


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    assert _expand_env("x=${FOO}") == "x=bar"
    assert _expand_env({"k": "${FOO}", "list": ["${FOO}"]}) == {
        "k": "bar",
        "list": ["bar"],
    }
    monkeypatch.delenv("MISSING", raising=False)
    assert _expand_env("${MISSING:-fallback}") == "fallback"
    assert _expand_env("${MISSING}") == ""


def test_load_profiles_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("ORTHANC_PWD", "secret123")
    cfg = tmp_path / "pacs.yaml"
    cfg.write_text(
        "default_profile: local\n"
        "profiles:\n"
        "  local:\n"
        "    backend: orthanc\n"
        "    base_url: http://localhost:8042\n"
        "    username: orthanc\n"
        "    password: ${ORTHANC_PWD}\n"
        "  remote:\n"
        "    backend: dicomweb\n"
        "    base_url: https://pacs.example.com/dicom-web\n"
    )
    profiles = load_profiles(cfg)
    assert set(profiles) == {"local", "remote"}
    assert profiles["local"].backend == "orthanc"
    assert profiles["local"].settings["password"] == "secret123"
    chosen = load_profile(None, cfg)
    assert chosen.name == "local"


def test_unknown_backend_raises(tmp_path):
    cfg = tmp_path / "pacs.yaml"
    cfg.write_text("profiles:\n  x:\n    backend: bogus\n    base_url: http://x\n")
    profile = load_profiles(cfg)["x"]
    with pytest.raises(pacs.PACSUnsupportedError):
        pacs.make_backend(profile)


def test_load_profile_missing_config(tmp_path, monkeypatch):
    monkeypatch.delenv("QR_PACS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_profile(None, None)


# ---------------------------------------------------------------------------- Orthanc


def _orthanc_backend(handler) -> OrthancBackend:
    backend = OrthancBackend(base_url="http://test")
    backend._client = httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    return backend


def test_orthanc_query_studies(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tools/find"
        body = json.loads(request.content)
        assert body["Level"] == "Study"
        assert body["Query"]["PatientID"] == "PAT001"
        return httpx.Response(
            200,
            json=[
                {
                    "ID": "orthanc-study-1",
                    "MainDicomTags": {
                        "StudyInstanceUID": "1.2.3",
                        "StudyDate": "20240101",
                    },
                    "PatientMainDicomTags": {"PatientID": "PAT001"},
                    "Series": ["s1", "s2"],
                }
            ],
        )

    backend = _orthanc_backend(handler)
    rows = backend.query_studies(PatientID="PAT001")
    assert rows[0]["PatientID"] == "PAT001"
    assert rows[0]["StudyInstanceUID"] == "1.2.3"
    assert rows[0]["NumberOfStudyRelatedSeries"] == 2
    assert rows[0]["_OrthancStudyID"] == "orthanc-study-1"


def test_orthanc_fetch_series(tmp_path):
    instance_bytes = b"DICM-payload"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tools/find":
            return httpx.Response(
                200,
                json=[
                    {
                        "ID": "series-1",
                        "MainDicomTags": {"SeriesInstanceUID": "S.UID"},
                        "Instances": ["inst-a"],
                    }
                ],
            )
        if request.url.path == "/series/series-1":
            return httpx.Response(
                200,
                json={
                    "ID": "series-1",
                    "Instances": ["inst-a", "inst-b"],
                    "MainDicomTags": {"SeriesInstanceUID": "S.UID"},
                },
            )
        if request.url.path.startswith("/instances/") and request.url.path.endswith(
            "/file"
        ):
            return httpx.Response(200, content=instance_bytes)
        return httpx.Response(404)

    backend = _orthanc_backend(handler)
    written = backend.fetch_series("STUDY.UID", "S.UID", tmp_path)
    assert {p.name for p in written} == {"inst-a.dcm", "inst-b.dcm"}
    assert (tmp_path / "inst-a.dcm").read_bytes() == instance_bytes


def test_orthanc_ping_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    backend = _orthanc_backend(handler)
    assert backend.ping() is False


def test_orthanc_store_dicom(tmp_path):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["ct"] = request.headers.get("Content-Type")
        captured["body"] = request.content
        return httpx.Response(200, json={"ID": "orthanc-inst-1", "Status": "Success"})

    backend = _orthanc_backend(handler)
    dcm = tmp_path / "x.dcm"
    dcm.write_bytes(b"DICM-PAYLOAD")
    res = backend.store_dicom(dcm)
    assert captured["path"] == "/instances"
    assert captured["ct"] == "application/dicom"
    assert captured["body"] == b"DICM-PAYLOAD"
    assert res["ID"] == "orthanc-inst-1"


# ---------------------------------------------------------------------------- DICOMweb


def _dicomweb_backend(handler) -> DICOMwebBackend:
    backend = DICOMwebBackend(base_url="http://test/dicom-web")
    backend._client = httpx.Client(
        base_url="http://test/dicom-web",
        transport=httpx.MockTransport(handler),
        headers={"Accept": "application/dicom+json"},
    )
    return backend


def test_dicomweb_simplify_dicom_json():
    row = {
        "00100020": {"vr": "LO", "Value": ["PAT001"]},
        "00100010": {"vr": "PN", "Value": [{"Alphabetic": "Doe^John"}]},
        "0020000D": {"vr": "UI", "Value": ["1.2.3"]},
    }
    out = _simplify_dicom_json(row)
    assert out["PatientID"] == "PAT001"
    assert out["PatientName"] == "Doe^John"
    assert out["StudyInstanceUID"] == "1.2.3"


def test_dicomweb_query_studies():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/studies")
        return httpx.Response(
            200,
            json=[
                {
                    "00100020": {"vr": "LO", "Value": ["PAT001"]},
                    "0020000D": {"vr": "UI", "Value": ["1.2.3"]},
                    "00080050": {"vr": "SH", "Value": ["ACC1"]},
                }
            ],
        )

    backend = _dicomweb_backend(handler)
    rows = backend.query_studies()
    assert rows[0]["PatientID"] == "PAT001"
    assert rows[0]["StudyInstanceUID"] == "1.2.3"
    assert rows[0]["AccessionNumber"] == "ACC1"


def test_dicomweb_query_series_under_study():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/studies/1.2.3/series")
        return httpx.Response(
            200,
            json=[
                {
                    "0020000E": {"vr": "UI", "Value": ["1.2.3.4"]},
                    "00080060": {"vr": "CS", "Value": ["CT"]},
                }
            ],
        )

    backend = _dicomweb_backend(handler)
    rows = backend.query_series(study_uid="1.2.3")
    assert rows[0]["SeriesInstanceUID"] == "1.2.3.4"
    assert rows[0]["Modality"] == "CT"


def test_dicomweb_fetch_instance_singlepart(tmp_path):
    payload = b"DICM-RAW-BYTES"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/instances/SOP.UID")
        return httpx.Response(
            200,
            content=payload,
            headers={"Content-Type": "application/dicom"},
        )

    backend = _dicomweb_backend(handler)
    out = backend.fetch_instance("STUDY", "SERIES", "SOP.UID", tmp_path / "out.dcm")
    assert out.read_bytes() == payload


def test_dicomweb_fetch_series_multipart(tmp_path):
    boundary = "TESTBOUNDARY"
    part_a = b"DICOM-A"
    part_b = b"DICOM-B"
    body = (
        f"--{boundary}\r\nContent-Type: application/dicom\r\n\r\n".encode()
        + part_a
        + f"\r\n--{boundary}\r\nContent-Type: application/dicom\r\n\r\n".encode()
        + part_b
        + f"\r\n--{boundary}--\r\n".encode()
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={
                "Content-Type": (
                    f'multipart/related; type="application/dicom"; boundary={boundary}'
                )
            },
        )

    backend = _dicomweb_backend(handler)
    out_dir = tmp_path / "dicom"
    paths = backend.fetch_series("STUDY", "SERIES", out_dir)
    blobs = sorted(p.read_bytes() for p in paths)
    assert blobs == sorted([part_a, part_b])


def test_dicomweb_ping_uses_qido():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/studies"):
            return httpx.Response(204)
        return httpx.Response(404)

    backend = _dicomweb_backend(handler)
    assert backend.ping() is True


# ---------------------------------------------------------------------------- factory


def test_make_backend_orthanc(tmp_path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        "profiles:\n"
        "  local:\n"
        "    backend: orthanc\n"
        "    base_url: http://orthanc:8042\n"
    )
    profile = load_profiles(cfg)["local"]
    backend = pacs.make_backend(profile)
    assert isinstance(backend, OrthancBackend)
    assert backend.base_url == "http://orthanc:8042"
    backend.close()


def test_make_backend_dicomweb_alias(tmp_path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        "profiles:\n"
        "  remote:\n"
        "    backend: wado\n"
        "    base_url: https://pacs/dicom-web\n"
    )
    profile = load_profiles(cfg)["remote"]
    backend = pacs.make_backend(profile)
    assert isinstance(backend, DICOMwebBackend)
    backend.close()


# ---------------------------------------------------------------------------- Orthanc remote-modality / changes


def test_orthanc_list_modalities():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/modalities"
        return httpx.Response(200, json=["MIM", "Eclipse", "STORESCP"])

    backend = _orthanc_backend(handler)
    assert backend.list_modalities() == ["MIM", "Eclipse", "STORESCP"]


def test_orthanc_get_modality_config():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/modalities/MIM/configuration"
        return httpx.Response(
            200,
            json={"AET": "AE_MIM", "Host": "10.0.0.1", "Port": 4008},
        )

    backend = _orthanc_backend(handler)
    cfg = backend.get_modality_config("MIM")
    assert cfg["AET"] == "AE_MIM"
    assert cfg["Port"] == 4008


def test_orthanc_retrieve_from_chains_query_answers_retrieve(monkeypatch):
    """``retrieve_from`` issues remote /query, lists /answers, and posts /retrieve."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(f"{request.method} {path}")
        if request.method == "POST" and path == "/modalities/MIM/query":
            body = json.loads(request.content)
            assert body["Level"] == "Study"
            assert body["Query"] == {"PatientID": "PAT001"}
            return httpx.Response(200, json={"ID": "Q1", "Path": "/queries/Q1"})
        if request.method == "GET" and path == "/queries/Q1/answers":
            return httpx.Response(200, json=["0", "1"])
        if request.method == "GET" and path.startswith("/queries/Q1/answers/"):
            return httpx.Response(
                200,
                json={
                    "PatientID": "PAT001",
                    "StudyInstanceUID": "1.2.3",
                },
            )
        if request.method == "POST" and path == "/queries/Q1/retrieve":
            body = json.loads(request.content)
            assert body["Synchronous"] is True
            assert body["TargetAet"] == "ORTHANC"
            return httpx.Response(
                200, json={"ID": "JOB1", "Path": "/jobs/JOB1"}
            )
        return httpx.Response(404)

    backend = _orthanc_backend(handler)
    result = backend.retrieve_from(
        "MIM",
        level="Study",
        target_aet="ORTHANC",
        PatientID="PAT001",
    )
    assert result["query_id"] == "Q1"
    assert len(result["answers"]) == 2
    assert result["retrieve"]["ID"] == "JOB1"
    # Confirm the chain executed in the expected order
    posts = [c for c in calls if c.startswith("POST")]
    assert posts == ["POST /modalities/MIM/query", "POST /queries/Q1/retrieve"]


def test_orthanc_get_changes_passes_since_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/changes"
        assert request.url.params.get("since") == "42"
        assert request.url.params.get("limit") == "10"
        return httpx.Response(
            200,
            json={
                "Changes": [
                    {
                        "ChangeType": "StableStudy",
                        "ResourceType": "Study",
                        "ID": "study-1",
                        "Date": "20260520T000000",
                    }
                ],
                "Done": True,
                "Last": 43,
            },
        )

    backend = _orthanc_backend(handler)
    page = backend.get_changes(since=42, limit=10)
    assert page["Last"] == 43
    assert page["Changes"][0]["ChangeType"] == "StableStudy"


def test_orthanc_echo_modality_success_and_failure():
    state = {"fail": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["fail"]:
            return httpx.Response(500, text="Cannot reach modality")
        return httpx.Response(200, json={})

    backend = _orthanc_backend(handler)
    assert backend.echo_modality("MIM") is True
    state["fail"] = True
    assert backend.echo_modality("MIM") is False
