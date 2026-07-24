"""qr convert — DICOM and other image-format conversions."""

from __future__ import annotations

import csv
from pathlib import Path

import click


def _glob_suffix(glob_pattern: str) -> str:
    """Return the trailing literal of a ``*``-style glob.

    ``"*_image.nrrd"`` -> ``"_image.nrrd"``. Used to recover a patient_id
    from a flat filename (``LUNG1-001_image.nrrd`` -> ``LUNG1-001``).
    """
    return glob_pattern.rsplit("*", 1)[-1] if "*" in glob_pattern else glob_pattern


@click.group()
def convert():
    """Convert medical image formats (DICOM <-> NRRD)."""


@convert.command("dicom-series")
@click.option(
    "--input",
    "-i",
    "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing a single DICOM series",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output NRRD path",
)
@click.option(
    "--modality",
    type=click.Choice(["auto", "CT", "PT", "MR", "CBCT", "OT"]),
    default="auto",
    help="Force a modality. 'auto' (default) reads it from the first slice. "
         "PT routes through SUV conversion (qradiomics.io.dicom.read_pet_suv).",
)
@click.option(
    "--raw-output",
    type=click.Path(),
    default=None,
    help="(PT only) Additionally write the raw (pre-SUV) NRRD here.",
)
def dicom_series(input_dir, output, modality, raw_output):
    """Convert a DICOM series directory to a single NRRD volume.

    \b
    Uses SimpleITK GDCM ImageSeriesReader for non-PET series. For PET
    series (Modality == PT) the slice loop in qradiomics.io.dicom.pet_suv
    is used instead so the output is in SUVbw units. Slice ordering is
    taken from the series UID, not filename.

    \b
    Example (auto-detect):
        qr convert dicom-series \\
          --input /data/ACRIN002/preCT/ \\
          --output /data/ACRIN002_preCT.nrrd
    """
    try:
        import SimpleITK as sitk
    except ImportError as e:
        click.echo(f"SimpleITK not available: {e}", err=True)
        raise SystemExit(1)

    input_path = Path(input_dir)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect modality from the first readable slice if not forced.
    detected = modality
    if modality == "auto":
        try:
            import pydicom
            for p in sorted(input_path.iterdir()):
                if not p.is_file():
                    continue
                try:
                    ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                except Exception:
                    continue
                m = str(getattr(ds, "Modality", "")).upper()
                if m:
                    detected = m
                    break
        except Exception:
            detected = "CT"

    if detected == "PT":
        from qradiomics.io.dicom import read_pet_suv
        result = read_pet_suv(input_path)
        sitk.WriteImage(result.image, str(output_path))
        if raw_output:
            Path(raw_output).parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(result.raw_image, str(raw_output))
        click.echo(
            f"Converted PET series -> {output_path} (SUVbw, "
            f"units={result.units}, factor={result.suv_factor:.6g}, "
            f"estimated={result.estimated}, "
            f"size={result.image.GetSize()}, "
            f"spacing={tuple(round(s, 3) for s in result.image.GetSpacing())})"
        )
        return

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(input_path))
    if not series_ids:
        click.echo(f"No DICOM series found under {input_path}", err=True)
        raise SystemExit(1)
    if len(series_ids) > 1:
        click.echo(
            f"Multiple series found ({len(series_ids)}); using the first. "
            "Pre-split the directory if you need a specific one.",
            err=True,
        )

    series_id = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(str(input_path), series_id)
    reader.SetFileNames(file_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_path))

    click.echo(
        f"Converted {len(file_names)} DICOM slice(s) (series {series_id}, "
        f"modality={detected}) -> {output_path} "
        f"[{image.GetSize()}, spacing {tuple(round(s, 3) for s in image.GetSpacing())}]"
    )


@convert.command("rtstruct")
@click.option(
    "--dicom-dir",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing the reference DICOM CT series",
)
@click.option(
    "--rtstruct",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="RTSTRUCT DICOM file (or directory containing one)",
)
@click.option(
    "--roi",
    default=None,
    help="ROI name to convert (default: first ROI in the structure set)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output NRRD label path",
)
def rtstruct_to_nrrd(dicom_dir, rtstruct, roi, output):
    """Convert an RTSTRUCT contour to a binary NRRD label volume.

    \b
    Requires the rt-utils package. The output mask has the same geometry
    (size, spacing, origin, direction) as the reference CT series.

    \b
    Example:
        qr convert rtstruct \\
          --dicom-dir /data/NSCLC-Cetuximab/<pat>/<date>/<study>/ \\
          --rtstruct /data/NSCLC-Cetuximab/<pat>/<date>/<study>/RTSeries/RS.<uid> \\
          --roi GTV \\
          --output /data/<pat>_GTV-label.nrrd
    """
    try:
        import SimpleITK as sitk
        import numpy as np
        from rt_utils import RTStructBuilder
    except ImportError as e:
        click.echo(f"Required package not available: {e}", err=True)
        click.echo("Install rt-utils: pip install rt-utils", err=True)
        raise SystemExit(1)

    # If --rtstruct points at a directory, locate the .dcm file inside.
    rt_path = Path(rtstruct)
    if rt_path.is_dir():
        candidates = sorted(rt_path.glob("*.dcm"))
        if not candidates:
            click.echo(f"No .dcm file in {rt_path}", err=True)
            raise SystemExit(1)
        rt_path = candidates[0]

    # Ensure RTSTRUCT has a valid preamble (often missing from TCIA downloads)
    from qradiomics.io.dicom import fix_dicom_preamble
    fix_dicom_preamble(rt_path, verbose=True)

    # rt-utils reads the reference CT series here; if that directory holds no
    # GDCM-readable DICOM (a real TCIA case — e.g. LUNG1-035, whose CT series
    # GDCM can't parse), it raises a bare Exception. Turn that into a clean,
    # non-zero exit instead of dumping a traceback mid-cohort-loop.
    try:
        rt = RTStructBuilder.create_from(
            dicom_series_path=str(dicom_dir), rt_struct_path=str(rt_path)
        )
    except Exception as e:
        click.echo(
            f'Could not build RTSTRUCT from {rt_path} against CT series in '
            f'{dicom_dir}: {e}. (Check that the reference CT is readable).',
            err=True,
        )
        raise SystemExit(1)
    roi_names = rt.get_roi_names()
    if not roi_names:
        click.echo(f"No ROIs found in {rtstruct}", err=True)
        raise SystemExit(1)

    target_roi = roi
    if target_roi is None:
        target_roi = roi_names[0]
        if len(roi_names) > 1:
            # Silently taking the first of several ROIs produces the wrong mask
            # (e.g. a lung/cord contour instead of GTV-1) with no error. Make it
            # loud so the caller notices and passes --roi.
            click.echo(
                f"WARNING: no --roi given and the structure set has {len(roi_names)} "
                f"ROIs {roi_names}; defaulting to the FIRST one ('{target_roi}'), "
                f"which is often NOT the intended target. Pass --roi explicitly.",
                err=True,
            )
        else:
            # Single-ROI RTSTRUCTs are unambiguous; stay quiet by default.
            pass
    elif target_roi not in roi_names:
        # Case-insensitive fallback — RTSTRUCT ROI names are inconsistently cased
        # across institutions (e.g., "Heart" vs "heart" vs "HEART").
        ci_match = next((n for n in roi_names if n.lower() == target_roi.lower()), None)
        if ci_match:
            click.echo(f"  ROI '{target_roi}' matched '{ci_match}' (case-insensitive)")
            target_roi = ci_match
        else:
            click.echo(
                f"ROI '{target_roi}' not in structure set. Available: {roi_names}", err=True
            )
            raise SystemExit(1)

    mask_zyx = rt.get_roi_mask_by_name(target_roi)
    # rt-utils returns (rows, cols, slices) bool array in DICOM-frame order.
    # Build a SimpleITK image with the CT's geometry so PyRadiomics treats
    # mask and image identically downstream.
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        click.echo(f"No DICOM series under {dicom_dir}", err=True)
        raise SystemExit(1)
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
    reader.SetFileNames(file_names)
    ct = reader.Execute()

    # rt_utils mask: (rows, cols, slices). SimpleITK GetArrayFromImage = (z, y, x).
    # Reorder to (slices, rows, cols) -> matches SimpleITK convention after permute.
    mask = np.transpose(mask_zyx, (2, 0, 1)).astype("uint8")

    # rt-utils occasionally returns one extra slice when the structure set
    # references a slice the CT series doesn't include. Trim or pad to the
    # CT size in the z-axis before copying geometry information.
    ct_size = ct.GetSize()  # (x, y, z)
    target_shape = (ct_size[2], ct_size[1], ct_size[0])
    if mask.shape != target_shape:
        diff = mask.shape[0] - target_shape[0]
        if diff > 0:
            mask = mask[:target_shape[0]]
        elif diff < 0:
            pad = np.zeros((-diff, target_shape[1], target_shape[2]), dtype=mask.dtype)
            mask = np.concatenate([mask, pad], axis=0)
        click.echo(
            f"  Adjusted mask z-extent by {diff:+d} slice(s) to match CT size {ct_size}",
        )

    label_img = sitk.GetImageFromArray(mask)
    label_img.CopyInformation(ct)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(label_img, str(output_path))

    n_vox = int(mask.sum())
    click.echo(
        f"Converted RTSTRUCT ROI '{target_roi}' -> {output_path} "
        f"[{label_img.GetSize()}, {n_vox} mask voxels]"
    )


@convert.command("manifest-from-dir")
@click.option(
    "--dataset-root",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory containing per-patient subdirectories",
)
@click.option(
    "--image-glob",
    default="*_CT.nrrd",
    help="Glob pattern relative to each patient dir for the image file (default: *_CT.nrrd)",
)
@click.option(
    "--mask-glob",
    default="*-label.nrrd",
    help="Glob pattern relative to each patient dir for the mask file (default: *-label.nrrd)",
)
@click.option(
    "--modality",
    default="CT",
    help="Modality string written to the manifest (default: CT)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output manifest CSV path",
)
def manifest_from_dir(dataset_root, image_glob, mask_glob, modality, output):
    """Build a manifest CSV by globbing image/mask pairs under a directory tree.

    \b
    Two layouts are supported:
      1. Per-patient subdirectories — each immediate subdirectory of
         --dataset-root is one patient (patient_id = subdirectory name);
         the first match of --image-glob and --mask-glob inside it is used.
      2. Flat layout (fallback when (1) yields nothing) — image/mask files
         sit directly in --dataset-root with a shared per-patient prefix
         (e.g. LUNG1-001_image.nrrd + LUNG1-001_mask.nrrd). The patient_id
         is recovered by stripping each glob's trailing literal.

    \b
    Example:
        qr convert manifest-from-dir \\
          --dataset-root /data/NSCLC/NSCLC/ \\
          --image-glob '*_CT.nrrd' \\
          --mask-glob '*_CT_manual_gtv-1-label.nrrd' \\
          --output /tmp/lung1_manifest.csv
    """
    import csv

    root = Path(dataset_root)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []
    for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        img = next(iter(patient_dir.glob(image_glob)), None)
        msk = next(iter(patient_dir.glob(mask_glob)), None)
        if img and msk:
            rows.append(
                {
                    "patient_id": patient_dir.name,
                    "modality": modality,
                    "image_path": str(img.resolve()),
                    "mask_path": str(msk.resolve()),
                }
            )
        else:
            skipped.append(patient_dir.name)

    # Fallback: flat layout with per-patient prefixes in the root itself
    # (e.g. LUNG1-001_image.nrrd + LUNG1-001_mask.nrrd side by side). The
    # per-patient-subdirectory walk above finds nothing there, so pair the
    # flat files by stripping each glob's trailing literal to recover the
    # patient_id.
    if not rows:
        img_suffix = _glob_suffix(image_glob)
        msk_suffix = _glob_suffix(mask_glob)
        flat_skipped = []
        for img in sorted(root.glob(image_glob)):
            if not img.is_file():
                continue
            pid = img.name[: -len(img_suffix)] if img_suffix and img.name.endswith(img_suffix) else img.stem
            msk = root / f"{pid}{msk_suffix}"
            if msk.is_file() and msk != img:
                rows.append(
                    {
                        "patient_id": pid,
                        "modality": modality,
                        "image_path": str(img.resolve()),
                        "mask_path": str(msk.resolve()),
                    }
                )
            else:
                flat_skipped.append(pid)
        skipped = flat_skipped

    if not rows:
        click.echo(f"No image/mask pairs found under {root}", err=True)
        raise SystemExit(1)

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "modality", "image_path", "mask_path"])
        w.writeheader()
        w.writerows(rows)

    click.echo(f"Wrote manifest with {len(rows)} patient(s) -> {output_path}")
    if skipped:
        click.echo(f"  skipped {len(skipped)} dir(s) without matching files (first 5): {skipped[:5]}")


@convert.command("fix-preamble")
@click.option(
    "--input",
    "-i",
    "dicom_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the DICOM/RTSTRUCT file with a missing/corrupt preamble",
)
def fix_preamble(dicom_path):
    """Verify and fix in-place the DICOM preamble of a file if missing/corrupt.

    Many DICOM and RTSTRUCT files downloaded directly from public archives (like
    TCIA) lack the standard 128-byte DICOM preamble and 'DICM' prefix, causing
    strict parsers like pydicom to fail.

    This command inspects the file, and if the preamble is missing, automatically
    reconstructs it in-place by reading the dataset with force=True, populating
    the required Media Storage headers, and writing it back with a standard header.
    """
    from qradiomics.io.dicom import fix_dicom_preamble

    p = Path(dicom_path)
    success = fix_dicom_preamble(p, verbose=True)
    if success:
        click.echo(f"✓ DICOM preamble verified/repaired successfully for: {p}")
    else:
        click.echo(f"✗ Failed to repair DICOM preamble for: {p}", err=True)
        raise SystemExit(1)


@convert.command("from-manifest")
@click.option("--manifest", "-m", "manifest_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Input manifest CSV with columns (patient_id, modality, "
                   "image_path, mask_path) — image_path is a DICOM-series directory, "
                   "mask_path is an RTSTRUCT .dcm file or directory containing one.")
@click.option("--nrrd-dir", "-d", required=True, type=click.Path(),
              help="Output NRRD root. Per-patient files land at "
                   "<nrrd-dir>/<patient_id>_<modality>.nrrd and "
                   "<nrrd-dir>/<patient_id>_<roi>-label.nrrd")
@click.option("--roi", default="GTV-1", show_default=True,
              help="RTSTRUCT ROI name to extract")
@click.option("--output", "-o", "output_manifest", required=True, type=click.Path(),
              help="Output manifest CSV pointing at the converted NRRDs")
@click.option("--skip-existing/--overwrite", default=True, show_default=True,
              help="Skip rows whose NRRDs already exist")
@click.option("--jobs", "-j", default=1, type=int, show_default=True,
              help="Parallel workers (1 = sequential). I/O- and CPU-bound; 4-8 is a "
                   "good default on a workstation, scale up to nproc for batch "
                   "conversion of large TCIA cohorts.")
def from_manifest_cmd(manifest_path, nrrd_dir, roi, output_manifest, skip_existing, jobs):
    """Batch-convert every (image_dir, rtstruct) row from an input manifest
    into an NRRD pair (image + ROI label), emitting a new manifest that
    points at the converted NRRDs.

    Pairs naturally with `qr tcia manifest` — that command produces the
    (image_dir, mask_dicom) manifest from a TCIA download tree, and this
    command turns it into the (image_nrrd, mask_nrrd) manifest that
    `qr extract` consumes.

    \b
    Example:
      qr tcia manifest \\
          --series      runs/lung1/series.csv \\
          --dicom-root  runs/lung1/dicom \\
          --output      runs/lung1/tcia_manifest.csv

      qr convert from-manifest \\
          --manifest  runs/lung1/tcia_manifest.csv \\
          --nrrd-dir  runs/lung1/nrrd \\
          --roi       GTV-1 \\
          --output    runs/lung1/manifest.csv
    """
    # Reuse the already-tested `qr convert dicom-series` and `qr convert rtstruct`
    # CLI commands via subprocess. Lets this command stay a thin loop without
    # duplicating their internal entry points.
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed

    nrrd_root = Path(nrrd_dir)
    nrrd_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read rows up-front so we can dispatch them across workers + write the
    # output manifest deterministically (sorted by input order).
    rows = []
    with open(manifest_path, newline="") as f_in:
        for row in csv.DictReader(f_in):
            pid = row.get("patient_id") or row.get("PatientID")
            modality = row.get("modality") or "CT"
            image_path = row.get("image_path")
            mask_path = row.get("mask_path")
            if pid and image_path and mask_path:
                rows.append((pid, modality, image_path, mask_path))

    def _convert_one(row):
        pid, modality, image_path, mask_path = row
        out_image = nrrd_root / f"{pid}_{modality}.nrrd"
        out_mask = nrrd_root / f"{pid}_{roi}-label.nrrd"

        if skip_existing and out_image.is_file() and out_mask.is_file():
            return {"pid": pid, "status": "cached", "row":
                    [pid, modality, str(out_image), str(out_mask)]}

        try:
            if not (skip_existing and out_image.is_file()):
                subprocess.run(
                    [sys.executable, "-m", "qradiomics.cli.main", "convert", "dicom-series",
                     "--input", image_path, "--output", str(out_image)],
                    check=True, capture_output=True,
                )
            if not (skip_existing and out_mask.is_file()):
                subprocess.run(
                    [sys.executable, "-m", "qradiomics.cli.main", "convert", "rtstruct",
                     "--dicom-dir", image_path, "--rtstruct", mask_path,
                     "--roi", roi, "--output", str(out_mask)],
                    check=True, capture_output=True,
                )
            return {"pid": pid, "status": "written", "row":
                    [pid, modality, str(out_image), str(out_mask)]}
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or b"").decode(errors="replace").strip().splitlines()[-1:] or [""]
            return {"pid": pid, "status": "failed",
                    "reason": f"exit {e.returncode}: {tail[0][-200:]}"}
        except Exception as e:  # noqa: BLE001
            return {"pid": pid, "status": "failed",
                    "reason": f"{type(e).__name__}: {e}"}

    results_by_pid = {}
    if jobs <= 1:
        for r in rows:
            res = _convert_one(r)
            results_by_pid[res["pid"]] = res
            if res["status"] == "failed":
                click.echo(f"  [{res['pid']}] failed: {res['reason']}", err=True)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {ex.submit(_convert_one, r): r[0] for r in rows}
            for fut in as_completed(futures):
                res = fut.result()
                results_by_pid[res["pid"]] = res
                if res["status"] == "failed":
                    click.echo(f"  [{res['pid']}] failed: {res['reason']}", err=True)

    # Emit the output manifest in the input row order, including cached rows.
    written = skipped = failed = 0
    with open(out_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["patient_id", "modality", "image_path", "mask_path"])
        for pid, *_ in rows:
            res = results_by_pid.get(pid)
            if res is None:
                continue
            if res["status"] == "failed":
                failed += 1
                continue
            writer.writerow(res["row"])
            if res["status"] == "cached":
                skipped += 1
            else:
                written += 1

    click.echo(
        f"qr convert from-manifest: wrote {written} new + {skipped} cached "
        f"({failed} failed) → {out_path}  [jobs={jobs}]"
    )
