"""qr extract must survive a worker process being killed mid-run.

A large volume can trip the OS OOM killer, which poisons the whole
ProcessPoolExecutor (every pending future then raises BrokenProcessPool).
The command must keep the patients that already finished, report the rest
as failed, and exit 0 — not crash with a traceback.
"""

import csv
import sys

import pytest
from click.testing import CliRunner

# ProcessPoolExecutor uses 'fork' on Linux, so a monkeypatch in the parent
# is inherited by workers. On spawn-based platforms it would not be, so scope
# this test to Linux.
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="relies on fork-inherited monkeypatch"
)

import importlib

extract_mod = importlib.import_module("qradiomics.cli.commands.extract")
extract = extract_mod.extract


def _fake_process_one(args):
    """Return a tiny feature dict, but hard-kill the worker for one patient."""
    import os

    patient_id = args[0]
    if patient_id == "BOOM":
        os._exit(1)  # simulate the OOM killer taking down this worker
    return (patient_id, {"patient_id": patient_id, "f1": 1.0, "f2": 2.0}, None)


def _write_manifest(path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "image_path", "mask_path"])
        w.writeheader()
        for pid in ("P1", "P2", "BOOM", "P3", "P4"):
            w.writerow({"patient_id": pid, "image_path": "x.nrrd", "mask_path": "m.nrrd"})


def test_extract_survives_killed_worker(tmp_path, monkeypatch):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    # In Python 3.14+, the default start method on Linux changed from 'fork' to 'forkserver'.
    # Under 'forkserver', monkeypatches in the parent process are not inherited by workers.
    # We explicitly force ProcessPoolExecutor to use the 'fork' context for this test.
    fork_context = multiprocessing.get_context("fork")
    monkeypatch.setattr(
        extract_mod,
        "ProcessPoolExecutor",
        partial(ProcessPoolExecutor, mp_context=fork_context),
    )

    monkeypatch.setattr(extract_mod, "_process_one", _fake_process_one)

    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    out = tmp_path / "features.csv"

    result = CliRunner().invoke(
        extract, ["-m", str(manifest), "-o", str(out), "-j", "2"]
    )

    # No traceback should escape; the command exits cleanly.
    assert result.exit_code == 0, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "out of memory" in result.output.lower()

    # The patients that finished before/around the crash are preserved.
    with open(out) as f:
        rows = list(csv.DictReader(f))
    saved = {r["patient_id"] for r in rows}
    assert saved, "expected at least some patients to be saved"
    assert "BOOM" not in saved
