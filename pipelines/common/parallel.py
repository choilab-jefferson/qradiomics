"""Shared parallel-extraction helper for pipelines/lidc_idri/*.py.

Eight scripts under pipelines/lidc_idri/ each hand-rolled the same skeleton:
``ProcessPoolExecutor`` + ``as_completed`` + a ``✓``/``✘`` per-item stderr
progress line + a ``csv.DictWriter(extrasaction="ignore")`` fed by a
union-of-keys fieldname pass (``keys = sorted({k for r in rows for k in
r.keys()})``). ``run_parallel_rows`` factors that skeleton out; only the
worker function (radiomics/AHSN/etc. logic) and the per-item progress
message stay per-script.

Design notes (these 8 scripts are PUBLIC_MANIFEST deliverables, so this
refactor must not change any script's CSV output or exit code — only how
the executor loop is spelled):

* Workers return their row(s) for one work item as ``dict``, ``list[dict]``,
  or a falsy value (no rows). A ``dict`` is wrapped as a 1-element list.
  This matches the 8 scripts as-is: most workers already emit
  ``list[dict]`` (one row per nodule/candidate for a patient);
  ``extract_cir.py``, ``validate_lcsr.py`` and ``mesh_to_voxel_compare.py``
  emit a single ``dict`` per (pid, nodule) work item.
* Row/id fields (``pid``, ``reader``, ``nodule_id``, ...) are the worker's
  job, not this helper's — every one of the 8 scripts already puts those
  into the returned row itself, so there is no separate id-injection step
  to reproduce here.
* The per-item stderr *success* message differs a lot across scripts (row
  counts, positive/negative counts, dice scores, truncated status
  strings, ...) — callers supply ``format_success(key, rows) -> str``,
  which becomes the text after ``"  ✓ {key}: "``. The *failure* message
  (raised exception) is identical across all 8 scripts
  (``f"  ✘ {key}: {exc}"``) and is therefore built into the helper.
  (Two of the 8 originals — ``mesh_to_voxel_compare.py`` for a degenerate
  "no dice" row — printed a bespoke ``"  ⚠ {key}: {row}"`` line *without*
  the ``✓`` prefix for a successful-but-degenerate result; here that case
  is instead rendered as ``"  ✓ {key}: ⚠ {row}"`` via ``format_success``,
  a deliberate cosmetic simplification of a stderr-only diagnostic line —
  CSV output and exit codes are unaffected.)
* ``is_ok(rows) -> bool`` is optional; when given, the helper tracks an
  "(N ok / M fail)" tally the way 5 of the 8 scripts do. When omitted, the
  summary line has no tally, matching the other 3.
* ``skip_write_if_empty=True`` reproduces the ``if not all_rows: print("no
  rows produced"); return 1`` guard used by 3 of the 8 scripts — the
  helper skips the CSV write and prints the same message, returning
  ``[]``. Callers check the return value themselves and ``return 1``; this
  helper never decides the process exit code.
"""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Union

Row = dict
WorkerResult = Union[Row, "list[Row]", None]


def run_parallel_rows(
    work: "list[Any]",
    worker: Callable[[Any], WorkerResult],
    jobs: int,
    out_csv: "Path | str",
    *,
    fieldnames: "list[str] | None" = None,
    label: str = "row",
    key_fn: "Callable[[Any], Any] | None" = None,
    format_success: "Callable[[Any, list[Row]], str] | None" = None,
    is_ok: "Callable[[list[Row]], bool] | None" = None,
    skip_write_if_empty: bool = False,
    summary_leading_blank: bool = False,
) -> "list[Row]":
    """Run ``worker(item)`` for each ``item`` in ``work`` across ``jobs``
    processes, logging ``✓``/``✘`` progress per item to stderr, collecting
    each worker's returned row(s), and writing them to ``out_csv`` via
    ``csv.DictWriter(extrasaction="ignore")``.

    Args:
        work: opaque per-item argument list handed to ``worker`` (usually
            tuples, matching the existing per-script worker signatures).
        worker: called in a subprocess for each item in ``work``; returns
            a ``dict`` (one row), ``list[dict]`` (0+ rows), or a falsy
            value (no rows).
        jobs: ``ProcessPoolExecutor(max_workers=jobs)``.
        out_csv: output CSV path.
        fieldnames: explicit column order. If ``None`` (default), the
            union of all returned dict keys is used, sorted — matching
            every one of the 8 original scripts.
        label: noun used by the default ``format_success`` (e.g.
            ``"{len(rows)} {label}(s)"``, i.e. "3 row(s)").
        key_fn: maps a work item to the identifier used in progress lines
            and as the ``as_completed`` lookup key. Defaults to
            ``item[0]`` for tuple/list items, else the item itself — most
            of the 8 scripts key progress lines by ``pid`` alone; scripts
            that key by ``(pid, nodule_n)`` pass an explicit ``key_fn``.
        format_success: ``(key, rows) -> str``, the text printed after
            ``"  ✓ {key}: "`` for a successfully-completed item (no
            exception). Defaults to ``f"{len(rows)} {label}(s)"``.
        is_ok: ``(rows) -> bool`` classifying a completed item as
            "ok" vs "fail" for the summary tally. If ``None``, no tally is
            tracked or printed.
        skip_write_if_empty: if the run produced zero rows, print
            "no rows produced" and return ``[]`` without writing
            ``out_csv`` (reproduces 3 of the 8 scripts' empty-output
            guard). If ``False`` (default), an (effectively empty) CSV is
            still written, matching the other 5 scripts.
        summary_leading_blank: prefix the final "wrote N rows" summary
            line with a blank line, matching 2 of the 8 scripts.

    Returns:
        The collected rows (``[]`` if none were produced).
    """
    def _default_key_fn(w):
        return w[0] if isinstance(w, (tuple, list)) else w

    def _default_format_success(_key, rows):
        return f"{len(rows)} {label}(s)"

    _key_fn = key_fn if key_fn is not None else _default_key_fn
    _format_success = format_success if format_success is not None else _default_format_success

    all_rows: "list[Row]" = []
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(worker, w): _key_fn(w) for w in work}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                result = fut.result()
            except Exception as e:
                fail += 1
                print(f"  ✘ {key}: {e}", file=sys.stderr)
                continue

            if not result:
                rows: "list[Row]" = []
            elif isinstance(result, dict):
                rows = [result]
            else:
                rows = list(result)

            if is_ok is not None:
                if is_ok(rows):
                    ok += 1
                else:
                    fail += 1

            all_rows.extend(rows)
            print(f"  ✓ {key}: {_format_success(key, rows)}", file=sys.stderr)

    if not all_rows and skip_write_if_empty:
        print("no rows produced", file=sys.stderr)
        return all_rows

    cols = (fieldnames if fieldnames is not None
            else sorted({k for row in all_rows for k in row.keys()}))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    summary = f"wrote {len(all_rows)} rows → {out_csv}"
    if is_ok is not None:
        summary += f"  ({ok} ok / {fail} fail)"
    if summary_leading_blank:
        summary = "\n" + summary
    print(summary, file=sys.stderr)

    return all_rows
