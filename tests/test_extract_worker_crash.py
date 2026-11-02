"""qr extract must survive a worker process being killed mid-run.

A large volume can trip the OS OOM killer, which poisons the whole
ProcessPoolExecutor (every pending future then raises BrokenProcessPool).
The command must keep the patients that already finished, report the rest
as failed, and exit 0 — not crash with a traceback.
"""

import csv
import importlib
import logging
import sys

import pytest
from click.testing import CliRunner

# ProcessPoolExecutor uses 'fork' on Linux, so a monkeypatch in the parent
# is inherited by workers. On spawn-based platforms it would not be, so scope
# this test to Linux.
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="relies on fork-inherited monkeypatch"
)

extract_mod = importlib.import_module("qradiomics.cli.commands.extract")
extractor_mod = importlib.import_module("qradiomics.extractor")
extract = extract_mod.extract


def _noop_worker_init(*args, **kwargs):
    return None


def _fake_extract_one_worker(patient_id: str, image_path: str, mask_path: str):
    """Return a tiny feature dict, but hard-kill the worker for one patient."""
    import os

    if patient_id == "BOOM":
        os._exit(1)  # simulate the OOM killer taking down this worker
    return {"patient_id": patient_id, "f1": 1.0, "f2": 2.0}


def _write_manifest(path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "image_path", "mask_path"])
        w.writeheader()
        for pid in ("P1", "P2", "BOOM", "P3", "P4"):
            w.writerow({"patient_id": pid, "image_path": "x.nrrd", "mask_path": "m.nrrd"})


def test_extract_survives_killed_worker(tmp_path, monkeypatch, caplog):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    fork_context = multiprocessing.get_context("fork")
    monkeypatch.setattr(
        extractor_mod,
        "ProcessPoolExecutor",
        partial(ProcessPoolExecutor, mp_context=fork_context),
    )
    monkeypatch.setattr(extractor_mod, "_worker_init", _noop_worker_init)
    monkeypatch.setattr(extractor_mod, "_extract_one_worker", _fake_extract_one_worker)

    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    out = tmp_path / "features.csv"

    with caplog.at_level(logging.ERROR, logger="qradiomics.extractor"):
        result = CliRunner().invoke(extract, ["-m", str(manifest), "-o", str(out), "-j", "2"])

    # No traceback should escape; the command exits cleanly.
    assert result.exit_code == 0, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "out of memory" in caplog.text.lower() or "failed" in result.output.lower()

    # The patients that finished before/around the crash are preserved.
    with open(out) as f:
        rows = list(csv.DictReader(f))
    saved = {r["patient_id"] for r in rows}
    assert saved, "expected at least some patients to be saved"
    assert "BOOM" not in saved
