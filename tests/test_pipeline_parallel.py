"""Mechanics tests for pipelines/common/parallel.run_parallel_rows.

Exercises the shared ProcessPoolExecutor + CSV-write helper used by the 8
pipelines/lidc_idri/*.py extraction scripts, with trivial workers — no
LIDC/radiomics data involved.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.common.parallel import run_parallel_rows  # noqa: E402


# Worker functions must be importable (picklable) at module scope for
# ProcessPoolExecutor.

def _worker_list(item):
    pid, n = item
    if pid == "bad":
        raise ValueError("boom")
    if pid == "empty":
        return []
    return [{"pid": pid, "i": i, "val": pid + str(i)} for i in range(n)]


def _worker_dict(item):
    pid = item
    if pid == "skip":
        return None
    return {"pid": pid, "status": "ok"}


def test_list_worker_writes_csv_and_returns_rows(tmp_path):
    out = tmp_path / "out.csv"
    work = [("a", 2), ("b", 1), ("empty", 0)]
    rows = run_parallel_rows(work, _worker_list, jobs=2, out_csv=out)

    assert len(rows) == 3
    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        written = list(reader)
    assert len(written) == 3
    pids = {r["pid"] for r in written}
    assert pids == {"a", "b"}


def test_dict_worker_wrapped_as_single_row(tmp_path):
    out = tmp_path / "out.csv"
    work = ["x", "y", "skip"]
    rows = run_parallel_rows(work, _worker_dict, jobs=2, out_csv=out,
                              key_fn=lambda w: w)

    assert len(rows) == 2
    assert {r["pid"] for r in rows} == {"x", "y"}


def test_exception_is_counted_as_fail_and_skipped(tmp_path, capsys):
    out = tmp_path / "out.csv"
    work = [("a", 1), ("bad", 1)]
    rows = run_parallel_rows(
        work, _worker_list, jobs=2, out_csv=out,
        is_ok=lambda rows: bool(rows),
    )
    captured = capsys.readouterr()
    assert len(rows) == 1
    assert "✘ bad:" in captured.err
    assert "(1 ok / 1 fail)" in captured.err


def test_skip_write_if_empty(tmp_path):
    out = tmp_path / "out.csv"
    work = [("empty", 0)]
    rows = run_parallel_rows(work, _worker_list, jobs=1, out_csv=out,
                              skip_write_if_empty=True)
    assert rows == []
    assert not out.exists()


def test_format_success_and_key_fn_are_used(tmp_path, capsys):
    out = tmp_path / "out.csv"
    work = [("a", 2)]
    run_parallel_rows(
        work, _worker_list, jobs=1, out_csv=out,
        key_fn=lambda w: f"{w[0]}#{w[1]}",
        format_success=lambda key, rows: f"custom {len(rows)}",
    )
    captured = capsys.readouterr()
    assert "✓ a#2: custom 2" in captured.err


def test_explicit_fieldnames_used_verbatim(tmp_path):
    out = tmp_path / "out.csv"
    work = [("a", 1)]
    run_parallel_rows(work, _worker_list, jobs=1, out_csv=out,
                       fieldnames=["pid", "val"])
    with open(out, newline="") as f:
        header = next(csv.reader(f))
    assert header == ["pid", "val"]
