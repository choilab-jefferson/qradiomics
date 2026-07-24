"""qr tcia — download imaging data from The Cancer Imaging Archive (TCIA).

Uses the public NBIA REST API (no auth needed for fully-public collections).
Series-level downloads stream a zip of DICOM files which is then extracted
into a `<output>/<patient_id>/<study_uid>/<series_uid>/` tree that the rest
of the qr CLI (`qr convert dicom-series`, `qr extract`, etc.) consumes.

References:
  https://wiki.cancerimagingarchive.net/display/Public/NBIA+Search+REST+API+Guide
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional

import click

NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def _http_get_json(url: str, params: Optional[dict] = None) -> list:
    import httpx

    with httpx.Client(timeout=120.0, follow_redirects=True) as c:
        r = c.get(url, params=params or {})
        r.raise_for_status()
        return r.json() if r.text else []


def _http_get_bytes(url: str, params: Optional[dict] = None) -> bytes:
    import httpx

    with httpx.Client(timeout=600.0, follow_redirects=True) as c:
        r = c.get(url, params=params or {})
        r.raise_for_status()
        return r.content


@click.group()
def tcia():
    """Download imaging data from The Cancer Imaging Archive (TCIA)."""


@tcia.command("collections")
def collections_cmd():
    """List public TCIA collection names."""
    data = _http_get_json(f"{NBIA_BASE}/getCollectionValues")
    for row in data:
        click.echo(row.get("Collection", row))


@tcia.command("series")
@click.option("--collection", "-c", required=True, help="TCIA collection name")
@click.option(
    "--modality",
    multiple=True,
    help="Modality filter (e.g. CT, PT, RTSTRUCT) — may be repeated",
)
@click.option(
    "--patient",
    "patient_ids",
    multiple=True,
    help="Restrict to specific PatientIDs",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output CSV (series-level table; one row per SeriesInstanceUID)",
)
def series_cmd(collection, modality, patient_ids, output):
    """List series in a TCIA collection, optionally filtered by modality / patient.

    \b
    Output: CSV with columns
        PatientID,StudyInstanceUID,SeriesInstanceUID,Modality,SeriesDescription,ImageCount
    """
    params = {"Collection": collection}
    series = _http_get_json(f"{NBIA_BASE}/getSeries", params=params)

    rows = []
    mods = set(m.upper() for m in modality) if modality else None
    pids = set(patient_ids) if patient_ids else None
    for s in series:
        if mods and s.get("Modality", "").upper() not in mods:
            continue
        if pids and s.get("PatientID") not in pids:
            continue
        rows.append(
            {
                "PatientID": s.get("PatientID"),
                "StudyInstanceUID": s.get("StudyInstanceUID"),
                "SeriesInstanceUID": s.get("SeriesInstanceUID"),
                "Modality": s.get("Modality"),
                "SeriesDescription": s.get("SeriesDescription", ""),
                "ImageCount": s.get("ImageCount", ""),
            }
        )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "PatientID",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Modality",
                "SeriesDescription",
                "ImageCount",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    click.echo(f"Wrote {len(rows)} series → {out}")


def _download_one(s: dict, out_root: Path, skip_existing: bool) -> dict:
    """Download a single series into <out_root>/<patient>/<study>/<series>/."""
    uid = s.get("SeriesInstanceUID")
    if not uid:
        return {"status": "skipped", "reason": "no_uid", "series": s}
    pid = s.get("PatientID", "UNKNOWN")
    study = s.get("StudyInstanceUID", "study")
    series_dir = out_root / pid / study / uid
    if skip_existing and series_dir.exists() and any(series_dir.iterdir()):
        return {"status": "cached", "pid": pid, "uid": uid,
                "modality": s.get("Modality", "?")}
    series_dir.mkdir(parents=True, exist_ok=True)
    try:
        blob = _http_get_bytes(f"{NBIA_BASE}/getImage", params={"SeriesInstanceUID": uid})
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            zf.extractall(series_dir)
        return {"status": "ok", "pid": pid, "uid": uid,
                "modality": s.get("Modality", "?"),
                "bytes": len(blob)}
    except Exception as e:
        return {"status": "failed", "pid": pid, "uid": uid,
                "modality": s.get("Modality", "?"), "error": str(e)}


@tcia.command("download")
@click.option("--collection", "-c", help="TCIA collection name (downloads ALL series)")
@click.option(
    "--series",
    "-s",
    help="Single SeriesInstanceUID (alternative to --collection or --manifest)",
)
@click.option(
    "--manifest",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    help="Series CSV from 'qr tcia series' (alternative to --collection)",
)
@click.option(
    "--modality",
    multiple=True,
    help="Modality filter when used with --collection",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory (DICOM files land at <out>/<patient>/<study>/<series>/)",
)
@click.option("--max-series", default=0, help="Stop after N series (0 = no limit)")
@click.option(
    "--workers",
    "-j",
    default=8,
    help="Parallel download workers (default 8). I/O-bound, so a higher number "
         "than CPU count is fine; TCIA's public NBIA tolerates ~16 concurrent.",
)
@click.option(
    "--progress/--no-progress",
    default=True,
    help="Show a rich progress bar (default on). Falls back to line output if "
         "rich isn't installed.",
)
@click.option("--skip-existing/--no-skip-existing", default=True)
def download_cmd(collection, series, manifest, modality, output, max_series,
                 workers, progress, skip_existing):
    """Download DICOM series from TCIA into a per-patient directory tree.

    \b
    Examples:
        qr tcia download --collection NSCLC-Radiomics --modality CT \\
            --modality RTSTRUCT -o /data/NSCLC-Radiomics -j 16

        qr tcia download --manifest series.csv -o /data/cohort

        qr tcia download --series <SeriesUID> -o /data/x
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not (collection or series or manifest):
        raise click.UsageError("Specify one of --collection, --series, --manifest")

    out_root = Path(output)
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve series list
    todo: list[dict] = []
    if series:
        todo.append({"SeriesInstanceUID": series})
    elif manifest:
        with open(manifest) as f:
            todo = list(csv.DictReader(f))
    else:
        series_list = _http_get_json(f"{NBIA_BASE}/getSeries", params={"Collection": collection})
        mods = set(m.upper() for m in modality) if modality else None
        for s in series_list:
            if mods and s.get("Modality", "").upper() not in mods:
                continue
            todo.append(s)

    if max_series:
        todo = todo[:max_series]

    total = len(todo)
    click.echo(f"Will download {total} series → {out_root}  (workers={workers}, "
               f"skip_existing={skip_existing})")

    import sys

    counts = {"ok": 0, "cached": 0, "failed": 0, "bytes": 0}
    # rich.Progress refreshes via terminal control codes and writes nothing
    # visible when stdout is redirected to a log file. Detect non-TTY and fall
    # back to a periodic plain-text "[n/total] pid (modality)" line so wrapper
    # scripts (test_all.sh, monitor.py, log tails) still see progress.
    use_rich = progress and sys.stdout.isatty()
    rich_bar = None
    if use_rich:
        try:
            from rich.progress import (
                BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                TextColumn, TimeElapsedColumn, TimeRemainingColumn,
                DownloadColumn,
            )
            rich_bar = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                DownloadColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("ETA"),
                TimeRemainingColumn(),
                refresh_per_second=4,
                transient=False,
            )
        except ImportError:
            rich_bar = None
            use_rich = False

    if rich_bar is not None:
        with rich_bar:
            tid = rich_bar.add_task("[cyan]downloading", total=total)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_download_one, s, out_root, skip_existing): s for s in todo}
                for fut in as_completed(futs):
                    r = fut.result()
                    status = r["status"]
                    if status == "ok":
                        counts["ok"] += 1
                        counts["bytes"] += r.get("bytes", 0)
                    elif status == "cached":
                        counts["cached"] += 1
                    else:
                        counts["failed"] += 1
                    rich_bar.update(
                        tid, advance=1,
                        description=(
                            f"[cyan]{r.get('pid','?')[:18]:<18}[/cyan] "
                            f"({r.get('modality','?')})"
                            f"  ok={counts['ok']} cached={counts['cached']} fail={counts['failed']}"
                        ),
                    )
    else:
        # Plain-text fallback: one line per completion, line-buffered so log
        # tails and the monitor see progress in real time.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_download_one, s, out_root, skip_existing): s for s in todo}
            n = 0
            for fut in as_completed(futs):
                r = fut.result(); n += 1
                status = r["status"]
                if status == "ok":
                    counts["ok"] += 1; counts["bytes"] += r.get("bytes", 0)
                elif status == "cached":
                    counts["cached"] += 1
                else:
                    counts["failed"] += 1
                # Periodic summary every 1% or 25 series, plus on the final row
                if n == total or n % max(1, total // 100) == 0 or n % 25 == 0:
                    pct = (100.0 * n / total) if total else 0
                    mb = counts["bytes"] / 1024 / 1024
                    click.echo(
                        f"  [{n:>5}/{total}] {pct:5.1f}%  "
                        f"ok={counts['ok']} cached={counts['cached']} fail={counts['failed']}  "
                        f"({mb:.0f} MB downloaded)  last={r.get('pid','?')} ({r.get('modality','?')})",
                        nl=True,
                    )
                    sys.stdout.flush()

    summary = {
        "total": total,
        "downloaded": counts["ok"],
        "cached": counts["cached"],
        "failed": counts["failed"],
        "bytes_downloaded": counts["bytes"],
    }
    click.echo("\n" + json.dumps(summary))


@tcia.command("clinical")
@click.option("--collection", "-c", required=True)
@click.option("--output", "-o", required=True, type=click.Path())
def clinical_cmd(collection, output):
    """Download clinical / patient-level metadata for a TCIA collection.

    \b
    Note: many TCIA collections distribute the clinical CSV as a wiki
    attachment rather than via the REST API. When that's the case this
    command writes an empty stub and prints the wiki URL — fetch the
    real CSV manually and pass it through `qr results merge`.
    """
    # Best-effort: try the getPatient endpoint, fall back to a stub.
    try:
        data = _http_get_json(f"{NBIA_BASE}/getPatient", params={"Collection": collection})
    except Exception:
        data = []

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if data:
        fields = sorted({k for row in data for k in row.keys()})
        # Map TCIA's PatientID to qradiomics' lowercase patient_id
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["patient_id"] + [k for k in fields if k != "PatientID"])
            w.writeheader()
            for row in data:
                row_out = {"patient_id": row.get("PatientID")}
                for k in fields:
                    if k != "PatientID":
                        row_out[k] = row.get(k, "")
                w.writerow(row_out)
        click.echo(f"Wrote {len(data)} patients → {out}")
    else:
        out.write_text("patient_id\n")
        click.echo(
            f"No REST metadata for '{collection}'. Stub written to {out}.\n"
            f"See https://www.cancerimagingarchive.net/collections/ for the wiki clinical CSV."
        )


@tcia.command("manifest")
@click.option("--series", "series_csv", required=True, type=click.Path(exists=True),
              help="series.csv from `qr tcia series` (rows = (PatientID, StudyInstanceUID, "
                   "SeriesInstanceUID, Modality, …))")
@click.option("--dicom-root", required=True, type=click.Path(exists=True, file_okay=False),
              help="Root of the downloaded DICOM tree "
                   "(<root>/<PatientID>/<StudyUID>/<SeriesUID>/*.dcm)")
@click.option("--image-modality", default="CT", show_default=True,
              help="Modality of the image series to include in the manifest "
                   "(CT, PT, MR, …)")
@click.option("--mask-modality", default="RTSTRUCT", show_default=True,
              help="Modality of the mask/RTSTRUCT series (RTSTRUCT, SEG)")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output manifest CSV (patient_id, modality, image_path, mask_path)")
def manifest_cmd(series_csv, dicom_root, image_modality, mask_modality, output):
    """Build a (patient_id, modality, image_path, mask_path) manifest from a
    TCIA `series.csv` plus the matching on-disk DICOM tree.

    `qr convert manifest-from-dir` relies on filename / directory-name globs
    (`**/CT`, `RS.*.dcm`) to classify series, which doesn't work against
    TCIA's UID-based layout (`<patient>/<study-uid>/<series-uid>/*.dcm`,
    every file numbered, no `CT`/`RS` literals anywhere). This command uses
    the Modality column already classified by `qr tcia series` and resolves
    each row to its on-disk directory by `<dicom_root>/<PatientID>/<StudyUID>/<SeriesUID>/`.

    \b
    Example:
      qr tcia manifest \\
          --series      runs/lung1/series.csv \\
          --dicom-root  runs/lung1/dicom \\
          --output      runs/lung1/manifest.csv

    For each patient with both an image-modality series and a mask-modality
    series, emits one row::

        patient_id, image_modality, <CT series dir>, <RTSTRUCT *.dcm file>

    Patients missing either side are skipped with a warning on stderr; the
    summary tallies how many patients were included / skipped.
    """
    root = Path(dicom_root)

    # Group rows by patient.
    by_pid: Dict[str, Dict[str, list]] = {}
    with open(series_csv, newline="") as f:
        for row in csv.DictReader(f):
            pid = row.get("PatientID") or row.get("patient_id")
            mod = row.get("Modality")
            if not pid or not mod:
                continue
            by_pid.setdefault(pid, {}).setdefault(mod, []).append(row)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_no_image = 0
    skipped_no_mask = 0
    skipped_path_missing = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "modality", "image_path", "mask_path"])
        for pid in sorted(by_pid):
            mods = by_pid[pid]
            if image_modality not in mods:
                skipped_no_image += 1
                continue
            if mask_modality not in mods:
                skipped_no_mask += 1
                continue

            img_row = mods[image_modality][0]
            mask_row = mods[mask_modality][0]
            img_dir = root / pid / img_row["StudyInstanceUID"] / img_row["SeriesInstanceUID"]
            mask_dir = root / pid / mask_row["StudyInstanceUID"] / mask_row["SeriesInstanceUID"]

            if not img_dir.is_dir():
                skipped_path_missing += 1
                continue
            # RTSTRUCT is a single-file series — pick the lone .dcm inside its dir.
            mask_file: Optional[Path]
            if mask_dir.is_dir():
                hits = sorted(mask_dir.glob("*.dcm"))
                mask_file = hits[0] if hits else None
            else:
                mask_file = None
            if mask_file is None or not mask_file.is_file():
                skipped_path_missing += 1
                continue

            w.writerow([pid, image_modality, str(img_dir), str(mask_file)])
            written += 1

    click.echo(
        f"Wrote {written} rows → {out} "
        f"(skipped: {skipped_no_image} no-{image_modality}, "
        f"{skipped_no_mask} no-{mask_modality}, "
        f"{skipped_path_missing} on-disk paths missing)"
    )
