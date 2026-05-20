"""qr lidc — LIDC-IDRI annotation conversion (XML → NRRD).

Subcommands:
    qr lidc convert    Convert one patient (CT + XML → NRRDs + CSV).
    qr lidc convert-cohort   Walk a TCIA-style LIDC-IDRI tree and
                             convert every patient.
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click

from qradiomics.io.lidc import convert_patient, scan_lidc_dir


@click.group()
def lidc():
    """LIDC-IDRI XML annotation conversion (XML → per-reader mask NRRD)."""


@lidc.command("convert")
@click.option("--dicom-dir", "-d", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="LIDC CT series directory.")
@click.option("--xml", "xml_path",
              type=click.Path(exists=True, dir_okay=False),
              default=None, help="Matching LIDC annotation XML (omit to skip masks).")
@click.option("--output", "-o", "out_dir", required=True, type=click.Path(),
              help="Output directory for the CT NRRD + mask NRRDs + nodules.csv.")
@click.option("--pid", required=True,
              help="Patient identifier used as the filename prefix.")
def convert_one(dicom_dir, xml_path, out_dir, pid):
    """Convert a single LIDC patient (CT + XML → NRRDs + CSV).

    \b
    Example:
        qr lidc convert \\
            --dicom-dir DATA/LIDC-IDRI/LIDC-IDRI-0001/.../CT_series/ \\
            --xml      DATA/LIDC-IDRI/LIDC-IDRI-0001/.../CT_series/074.xml \\
            --output   output/lidc/LIDC-IDRI-0001 \\
            --pid      LIDC-IDRI-0001
    """
    result = convert_patient(dicom_dir, xml_path, out_dir, pid)
    click.echo(f"CT  → {result['ct_nrrd']}  size={result['ct_size']}")
    for r in result["readers"]:
        click.echo(f"  reader {r['session_index']}: "
                   f"{r['n_nodules']} nodule(s) → {r['mask_nrrd']}")
    if result["nodules_csv"]:
        click.echo(f"nodules CSV → {result['nodules_csv']}")


def _convert_one_task(args):
    pid, dicom_dir, xml_path, out_dir = args
    try:
        result = convert_patient(dicom_dir, xml_path,
                                  Path(out_dir) / pid, pid)
        return (pid, result, None)
    except Exception as e:
        return (pid, None, str(e))


@lidc.command("convert-cohort")
@click.option("--src", required=True, type=click.Path(exists=True, file_okay=False),
              help="Root of the LIDC-IDRI tree (TCIA layout).")
@click.option("--out", "out_dir", required=True, type=click.Path(),
              help="Output root; one subdir per patient.")
@click.option("--limit", type=int, default=None,
              help="Stop after N patients (smoke testing).")
@click.option("--jobs", "-j", type=int, default=4,
              help="Parallel worker processes (default 4).")
def convert_cohort(src, out_dir, limit, jobs):
    """Walk an LIDC-IDRI tree and convert every patient in parallel.

    Writes per-patient outputs under `<out>/<pid>/` plus a top-level
    `<out>/cohort_summary.csv` recording success/failure + nodule counts.
    """
    src = Path(src); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cases = list(scan_lidc_dir(src))
    if limit:
        cases = cases[:limit]
    click.echo(f"found {len(cases)} CT series under {src}")

    work = [(pid, str(d), str(x) if x else None, str(out_dir))
            for pid, d, x in cases]

    summary_rows = []
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_convert_one_task, w): w[0] for w in work}
        for fut in as_completed(futs):
            pid, result, err = fut.result()
            if err:
                n_fail += 1
                click.echo(f"  ✘ {pid}: {err}", err=True)
                summary_rows.append({"pid": pid, "status": "error", "error": err})
            else:
                n_ok += 1
                n_readers = len(result["readers"])
                n_nodules = sum(r["n_nodules"] for r in result["readers"])
                click.echo(f"  ✓ {pid}: {n_readers} reader(s), {n_nodules} nodule(s)")
                summary_rows.append({"pid": pid, "status": "ok",
                                     "n_readers": n_readers,
                                     "n_nodules": n_nodules,
                                     "ct_nrrd": result["ct_nrrd"],
                                     "nodules_csv": result["nodules_csv"] or ""})

    summary_path = out_dir / "cohort_summary.csv"
    if summary_rows:
        keys = sorted({k for row in summary_rows for k in row.keys()})
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
    click.echo(f"\nsummary: {n_ok} ok / {n_fail} fail → {summary_path}")
