#!/usr/bin/env python3
"""monitor.py — live TUI for in-flight qradiomics pipeline runs.

Polls each cohort's `<OUT>/.status/state.json` plus directory file counts and
renders a continuously-updating rich dashboard:

    ┌─ qradiomics pipeline status ─────────────────────────────────────┐
    │ runs/lung1   NSCLC-Radiomics                                      │
    │   ✓ catalog                                                        │
    │   ✓ fetch       142/422 series                                     │
    │   ◐ series      ████████░░░░ 312/422 (74%)  ETA 02:14              │
    │   ⌛ patient                                                       │
    │   ⌛ features                                                      │
    │   ⌛ modeling                                                       │
    │ runs/nsclc_cetuximab   NSCLC-Cetuximab                            │
    │   ...                                                              │
    └────────────────────────────────────────────────────────────────────┘

Usage:
    python monitor.py runs/                  # auto-detect cohorts under runs/
    python monitor.py runs/lung1 runs/acrin  # specific cohort dirs
    python monitor.py --once runs/           # one-shot snapshot (no live loop)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text


STAGE_NAMES = ["catalog", "fetch", "series", "patient", "features", "merge", "modeling"]
STAGE_GLYPH = {"pending": "⌛", "running": "◐", "done": "✓", "failed": "✗"}


def _read_status(out: Path) -> dict:
    """Read $OUT/.status/state.json plus counts derived from the filesystem."""
    s: dict = {"stages": {n: {"state": "pending"} for n in STAGE_NAMES}}
    sf = out / ".status" / "state.json"
    if sf.exists():
        try:
            data = json.loads(sf.read_text())
            for k, v in data.get("stages", {}).items():
                if k in s["stages"]:
                    s["stages"][k].update(v)
            s["collection"] = data.get("collection")
            s["roi"] = data.get("roi")
        except Exception:
            pass

    # Live counts (cheap glob)
    s["counts"] = {
        "ct_series_catalog": _csv_rows(out / "ct_series.csv"),
        "rt_series_catalog": _csv_rows(out / "rt_series.csv"),
        "dicom_series": _count_dirs(out / "dicom", depth=3),
        "nrrd_ct": len(list((out / "series").glob("*/*__CT.nrrd"))) if (out / "series").exists() else 0,
        "nrrd_mask": _count_masks(out / "series"),
        "patient_jsons": _count_jsons(out / "patient"),
        "manifest_rows": _csv_rows(out / "manifest.csv"),
        "feature_rows": _csv_rows(out / "features.csv"),
    }
    return s


def _csv_rows(p: Path) -> int:
    try:
        with open(p) as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def _count_dirs(p: Path, depth: int) -> int:
    if not p.exists():
        return 0
    import os
    n = 0
    base = str(p)
    base_depth = base.count(os.sep)
    for root, dirs, _ in os.walk(base):
        if root.count(os.sep) - base_depth == depth - 1:
            n += len(dirs)
            dirs[:] = []
    return n


def _count_masks(p: Path) -> int:
    if not p.exists():
        return 0
    return len([f for f in p.glob("*/*-label.nrrd")])


def _count_jsons(p: Path) -> int:
    return len(list(p.glob("*.json"))) if p.exists() else 0


def _build_panel(out: Path, status: dict) -> Panel:
    title = f"{out.name} [dim]({status.get('collection', '?')}, ROI={status.get('roi','?')})[/dim]"

    table = Table.grid(padding=(0, 1))
    table.add_column(no_wrap=True)
    table.add_column(ratio=1)
    table.add_column(no_wrap=True)

    counts = status["counts"]
    target = counts["ct_series_catalog"] or 1

    for stage in STAGE_NAMES:
        st = status["stages"][stage].get("state", "pending")
        glyph = STAGE_GLYPH.get(st, "·")
        glyph_color = {"done": "green", "running": "yellow",
                       "failed": "red", "pending": "dim"}.get(st, "white")
        detail = _stage_detail(stage, status)
        bar = _stage_bar(stage, status, target)
        table.add_row(
            Text(f"  {glyph} {stage:9s}", style=glyph_color),
            bar,
            Text(detail, style="dim"),
        )

    return Panel(table, title=title, border_style="cyan")


def _stage_detail(stage: str, status: dict) -> str:
    c = status["counts"]
    if stage == "catalog":
        return f"{c['ct_series_catalog']} CT + {c['rt_series_catalog']} RT series"
    if stage == "fetch":
        return f"{c['dicom_series']} series downloaded"
    if stage == "series":
        return f"{c['nrrd_ct']} CT.nrrd + {c['nrrd_mask']} mask.nrrd"
    if stage == "patient":
        return f"{c['patient_jsons']} patient summaries"
    if stage == "features":
        return f"manifest={c['manifest_rows']}  features={c['feature_rows']}"
    return ""


def _stage_bar(stage: str, status: dict, target: int) -> Text:
    c = status["counts"]
    if stage == "catalog":
        return Text("")
    if stage == "fetch":
        cur, tot = c["dicom_series"], (c["ct_series_catalog"] + c["rt_series_catalog"])
    elif stage == "series":
        cur, tot = c["nrrd_ct"] + c["nrrd_mask"], c["dicom_series"] or 1
    elif stage == "patient":
        cur, tot = c["patient_jsons"], c["nrrd_ct"] or 1
    elif stage == "features":
        cur, tot = c["feature_rows"], c["manifest_rows"] or 1
    else:
        return Text("")
    cur = min(cur, tot) if tot else cur
    pct = (cur / tot) if tot else 0
    width = 30
    filled = int(round(pct * width))
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if pct >= 1.0 else "yellow" if pct > 0 else "dim"
    return Text(f"{bar} {cur}/{tot}", style=color)


def _gather_cohort_dirs(args: list[str]) -> list[Path]:
    out_dirs: list[Path] = []
    for a in args:
        p = Path(a)
        if not p.exists():
            continue
        # If it's a 'runs/' parent, expand to subdirectories that look like cohort runs
        if (p / "ct_series.csv").exists() or (p / "manifest.csv").exists() or (p / "dicom").exists():
            out_dirs.append(p)
        else:
            for sub in sorted(p.iterdir()):
                if sub.is_dir() and ((sub / "ct_series.csv").exists() or
                                     (sub / "manifest.csv").exists() or
                                     (sub / "dicom").exists()):
                    out_dirs.append(sub)
    return out_dirs


def main() -> int:
    ap = argparse.ArgumentParser(description="Live status dashboard for qradiomics pipeline runs")
    ap.add_argument("paths", nargs="+", help="Cohort OUT directories (or runs/ parent)")
    ap.add_argument("--interval", type=float, default=1.0, help="Refresh seconds (default 1.0)")
    ap.add_argument("--once", action="store_true", help="Render once and exit (no live loop)")
    args = ap.parse_args()

    console = Console()
    cohorts = _gather_cohort_dirs(args.paths)
    if not cohorts:
        console.print("[red]No cohort directories found under given paths.[/red]")
        return 1

    def _render() -> Group:
        panels = [_build_panel(c, _read_status(c)) for c in cohorts]
        return Group(*panels)

    if args.once:
        console.print(_render())
        return 0

    try:
        with Live(_render(), refresh_per_second=1 / max(0.2, args.interval), console=console,
                  transient=False) as live:
            while True:
                time.sleep(args.interval)
                live.update(_render())
    except KeyboardInterrupt:
        console.print("\n[dim]monitor stopped.[/dim]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
