"""qr preprocess — crop to ROI bounding box and (optionally) isotropic resample.

Re-implements the `crop_image` and `resample_image_{1mm,iso}` processes
from the legacy radiomics-tools Nextflow pipelines. Output is a small
cropped+resampled CT NRRD and matching label NRRD that downstream feature
extractors (PyRadiomics, shape-let, AHSN, deep learning) all consume.

Outputs land under <output>/ as:
    <patient_id>_CT_cropped.nrrd
    <patient_id>_<ROI>_cropped-label.nrrd
optionally with `_1mm` or `_iso` suffix when resampled.
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click


def _align_mask_to_image(image, mask):
    """Resample mask onto the image grid if the two differ.

    Mirrors the legacy ``RESAMPLE_MASK_TO_IMAGE`` Nextflow module: when a
    mask is stored on its own (often smaller) grid — e.g. a heart label
    pre-cropped to the organ's bounding box while its parent CT remains
    full-body — the bbox-driven crop below would request a region larger
    than the mask itself and fail. Aligning the mask to the image grid
    via nearest-neighbour resampling (default value 0 outside the mask's
    native extent) makes the rest of the pipeline grid-agnostic without
    a separate pre-padding stage.
    """
    import SimpleITK as sitk
    if (
        mask.GetSize() == image.GetSize()
        and mask.GetSpacing() == image.GetSpacing()
        and mask.GetOrigin() == image.GetOrigin()
        and mask.GetDirection() == image.GetDirection()
    ):
        return mask
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(image)
    rf.SetInterpolator(sitk.sitkNearestNeighbor)
    rf.SetDefaultPixelValue(0)
    rf.SetOutputPixelType(mask.GetPixelID())
    return rf.Execute(mask)


def _crop_one(args):
    """Per-patient crop (+ optional resample). Returns (pid, ct_out, mask_out, err)."""
    pid, image_path, mask_path, out_dir, roi, pad_mm, resample = args
    try:
        import SimpleITK as sitk
        import numpy as np

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        mask = _align_mask_to_image(image, mask)

        # ROI bounding box (+ pad in mm converted to voxels).
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask)
        labels = list(stats.GetLabels())
        if not labels:
            return (pid, None, None, "empty_mask")
        bbox = np.array(stats.GetBoundingBox(labels[0]))   # (x, y, z, sx, sy, sz)
        pad_vox = (pad_mm / np.array(image.GetSpacing())).astype("int")
        start = bbox[:3]; size = bbox[3:]
        end = start + size
        lower = np.maximum(start - pad_vox, 0)
        upper = np.maximum(np.array(image.GetSize()) - end - pad_vox, 0)

        cropped_img = sitk.Crop(image, lower.tolist(), upper.tolist())
        cropped_msk = sitk.Crop(mask, lower.tolist(), upper.tolist())
        cropped_msk.CopyInformation(cropped_img)

        # Optional isotropic resample
        suffix = "_cropped"
        if resample:
            spacing = (float(resample), float(resample), float(resample))
            cropped_img = _resample(cropped_img, spacing, is_label=False)
            cropped_msk = _resample(cropped_msk, spacing, is_label=True)
            suffix = f"_cropped-{resample}mm" if resample != 1.0 else "_cropped-1mm"

        ct_out = Path(out_dir) / f"{pid}{suffix}_CT.nrrd"
        msk_out = Path(out_dir) / f"{pid}{suffix}_{roi}-label.nrrd"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(cropped_img, str(ct_out))
        sitk.WriteImage(cropped_msk, str(msk_out))
        return (pid, str(ct_out), str(msk_out), None)
    except Exception as e:
        return (pid, None, None, str(e))


def _resample(image, spacing_xyz, is_label):
    import SimpleITK as sitk
    import numpy as np

    rf = sitk.ResampleImageFilter()
    in_spacing = np.array(image.GetSpacing())
    in_size = np.array(image.GetSize())
    out_size = np.ceil(in_size * in_spacing / np.array(spacing_xyz)).astype(int).tolist()
    rf.SetOutputSpacing(spacing_xyz)
    rf.SetOutputOrigin(image.GetOrigin())
    rf.SetOutputDirection(image.GetDirection())
    rf.SetSize(out_size)
    rf.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return rf.Execute(image)


@click.command()
@click.option("--manifest", "-m", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Manifest CSV (patient_id, modality, image_path, mask_path)")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output directory for cropped/resampled NRRDs")
@click.option("--roi-name", default="GTV",
              help="ROI suffix used in output filenames (default: GTV)")
@click.option("--pad-mm", default=20.0, type=float,
              help="Padding around the bounding box, in millimetres (default 20)")
@click.option("--resample", default=None, type=float,
              help="Isotropic spacing in mm (e.g. 1.0). Omit to skip resampling.")
@click.option("--jobs", "-j", default=1, type=int,
              help="Worker processes (default 1).")
@click.option("--out-manifest", default=None, type=click.Path(),
              help="If set, also write a new manifest CSV pointing at the cropped pairs.")
def preprocess(manifest, output, roi_name, pad_mm, resample, jobs, out_manifest):
    """Crop and optionally resample image+mask pairs from a manifest.

    \b
    Example:
        qr preprocess -m manifest.csv -o cropped/ --roi-name GTV-1 \\
            --pad-mm 20 --resample 1.0 --jobs 8 --out-manifest cropped.csv
    """
    rows = list(csv.DictReader(open(manifest)))
    total = len(rows)
    click.echo(f"Reading manifest: {manifest}  ({total} patients, jobs={jobs})")

    work = [
        (r["patient_id"], r["image_path"], r["mask_path"], output, roi_name, pad_mm, resample)
        for r in rows
    ]

    results = []
    if jobs <= 1:
        for i, w in enumerate(work, 1):
            print(f"  preprocess [{i}/{total}] {w[0]} ...", flush=True)
            r = _crop_one(w)
            print(f"  preprocess [{i}/{total}] {r[0]} {'ok' if not r[3] else 'FAIL: ' + r[3]}",
                  flush=True)
            results.append(r)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_crop_one, w): w[0] for w in work}
            for fut in as_completed(futs):
                done += 1
                r = fut.result()
                print(f"  preprocess [{done}/{total}] {r[0]} "
                      f"{'ok' if not r[3] else 'FAIL: ' + r[3]}", flush=True)
                results.append(r)
                sys.stdout.flush()

    if out_manifest:
        with open(out_manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["patient_id", "modality", "image_path", "mask_path"])
            w.writeheader()
            for pid, ct, msk, err in results:
                if err is None and ct and msk:
                    w.writerow({"patient_id": pid, "modality": "CT",
                                "image_path": ct, "mask_path": msk})
        click.echo(f"Wrote cropped manifest → {out_manifest}")

    ok = sum(1 for r in results if not r[3])
    fail = total - ok
    click.echo(f"\nPreprocess: {ok} ok, {fail} failed → {output}")
