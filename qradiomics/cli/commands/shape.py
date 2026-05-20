"""qr shape — extract shape descriptors (AHSN, spiculation) per patient.

Wraps qradiomics.shape for batch use. Each patient's CT+mask pair runs
through:
  - AHSN 2014  (180-dim Angular Histogram of Surface Normals)
  - Spiculation 2021  (Na, Nl, Na_att, s1, s2 — mesh-based)

Both feature sets are appended into a single shape_features.csv with
the AHSN descriptor flattened into columns (ahsn_000…ahsn_179) and the
spiculation scalars as named columns.
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click


def _shape_one(args):
    pid, image_path, mask_path, do_ahsn, do_spic = args
    try:
        import SimpleITK as sitk
        import numpy as np

        result = {"patient_id": pid}

        ct = sitk.GetArrayFromImage(sitk.ReadImage(image_path)).astype("float32")
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype("float32")

        # Crop CT to mask bounding box (+ pad) so AHSN / mesh ops stay cheap
        idx = np.argwhere(mask > 0)
        if idx.size == 0:
            return (pid, None, "empty_mask")
        z0, y0, x0 = idx.min(axis=0); z1, y1, x1 = idx.max(axis=0) + 1
        z0, y0, x0 = max(0, z0 - 4), max(0, y0 - 4), max(0, x0 - 4)
        ct_c = ct[z0:z1+4, y0:y1+4, x0:x1+4]
        mask_c = mask[z0:z1+4, y0:y1+4, x0:x1+4]

        if do_ahsn:
            from qradiomics.shape import ahsn, AHSNConfig
            desc = ahsn(ct_c, mask=mask_c > 0, cfg=AHSNConfig())
            for i, v in enumerate(desc):
                result[f"ahsn_{i:03d}"] = float(v)

        if do_spic:
            try:
                from qradiomics.shape import spiculation_from_voxel
                feats, _, _, _ = spiculation_from_voxel(mask_c.astype("float32"), n_param_iter=80)
                for k, v in feats.as_dict().items() if hasattr(feats, "as_dict") else feats.__dict__.items():
                    if isinstance(v, (int, float)):
                        result[f"spic_{k}"] = float(v)
            except Exception as e:
                result["spic_error"] = str(e)

        return (pid, result, None)
    except Exception as e:
        return (pid, None, str(e))


@click.group()
def shape():
    """Shape descriptors (AHSN, spiculation) — qradiomics.shape CLI."""


@shape.command("extract")
@click.option("--manifest", "-m", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Manifest CSV (patient_id, modality, image_path, mask_path)")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output shape_features.csv")
@click.option("--ahsn/--no-ahsn", default=True, help="Compute AHSN (2014 CMPB) descriptor")
@click.option("--spiculation/--no-spiculation", default=True,
              help="Compute spiculation features (2021 CMPB)")
@click.option("--jobs", "-j", default=1, type=int)
def extract(manifest, output, ahsn, spiculation, jobs):
    """Extract shape descriptors for every patient in a manifest."""
    rows = list(csv.DictReader(open(manifest)))
    total = len(rows)
    click.echo(f"Reading manifest: {manifest}  ({total} patients, jobs={jobs})")

    work = [(r["patient_id"], r["image_path"], r["mask_path"], ahsn, spiculation) for r in rows]

    out_path = Path(output); out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0; n_fail = 0
    with open(out_path, "w", newline="") as fout:
        writer = None
        def _emit(pid, row, err):
            nonlocal writer, n_ok, n_fail
            if row is None:
                n_fail += 1; return
            if writer is None:
                writer = csv.DictWriter(fout, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row); fout.flush(); n_ok += 1

        if jobs <= 1:
            for i, w in enumerate(work, 1):
                pid = w[0]
                print(f"  shape [{i}/{total}] {pid} ...", flush=True)
                pid, row, err = _shape_one(w)
                _emit(pid, row, err)
                print(f"  shape [{i}/{total}] {pid} {'ok' if row else 'FAIL: ' + (err or '')}",
                      flush=True)
        else:
            done = 0
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(_shape_one, w): w[0] for w in work}
                for fut in as_completed(futs):
                    done += 1
                    pid, row, err = fut.result()
                    _emit(pid, row, err)
                    print(f"  shape [{done}/{total}] {pid} "
                          f"{'ok' if row else 'FAIL: ' + (err or '')}", flush=True)
                    sys.stdout.flush()

    click.echo(f"\nShape: {n_ok} ok, {n_fail} failed → {out_path}")
