"""qr convert — DICOM and other image-format conversions."""

from __future__ import annotations

from pathlib import Path

import click


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

    rt = RTStructBuilder.create_from(dicom_series_path=str(dicom_dir), rt_struct_path=str(rt_path))
    roi_names = rt.get_roi_names()
    if not roi_names:
        click.echo(f"No ROIs found in {rtstruct}", err=True)
        raise SystemExit(1)

    target_roi = roi
    if target_roi is None:
        target_roi = roi_names[0]
        click.echo(f"No --roi specified, using first ROI '{target_roi}' (available: {roi_names})")
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
    Each immediate subdirectory of --dataset-root is treated as one patient
    (patient_id = subdirectory name). The first match of --image-glob and
    --mask-glob within that patient directory is recorded.

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
