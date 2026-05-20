"""LIDC AHSN nodule detection — Choi 2014 CMPB reproduction.

For each LIDC patient:
  1. Run multi-threshold candidate detection across whole CT (lung mask via
     simple HU threshold).
  2. For each candidate, extract a block, compute AHSN descriptor.
  3. Label candidate POSITIVE if its center lies inside any reader's mask
     (any nodule rasterised by qradiomics.io.lidc), NEGATIVE otherwise.
  4. Emit candidates CSV with (pid, z, y, x, scale, diameter, dot, label, ahsn_*).

Then a downstream notebook/script can train RF/SVM on `ahsn_*` and compute
sensitivity / FP-rate per case (Choi 2014 Table 4).

Note: This is a simplified reproduction. The original paper uses a more
sophisticated lung-mask + critical-section removal pipeline; here we use
a quick HU < -400 lung mask. Detection AUC is what we compare, not the
exact lung-segmentation match.
"""
from __future__ import annotations

import argparse
import csv
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from qradiomics.io.lidc import parse_lidc_xml, scan_lidc_dir
from qradiomics.io.lidc.extract import _rasterise_nodule, _uid_to_z
from qradiomics.shape import (
    AHSNConfig,
    ahsn,
    detect_candidates,
    extract_block,
)


def _quick_lung_mask(ct_arr: np.ndarray, lower_hu: float = -900.0,
                     upper_hu: float = -400.0) -> np.ndarray:
    """Quick HU-window lung mask — body interior between -900 and -400 HU."""
    from scipy.ndimage import binary_fill_holes
    mask = (ct_arr > lower_hu) & (ct_arr < upper_hu)
    # Fill holes per slice (rough lung interior)
    for z in range(mask.shape[0]):
        mask[z] = binary_fill_holes(mask[z])
    return mask


def _detect_one(args) -> list[dict]:
    pid, patient_dir, lidc_src = args
    rows: list[dict] = []
    try:
        patient_dir = Path(patient_dir)
        ct_path = patient_dir / f"{pid}_CT.nrrd"
        if not ct_path.exists():
            return [{"pid": pid, "status": "missing_ct"}]

        ct = sitk.ReadImage(str(ct_path))
        ct_arr = sitk.GetArrayFromImage(ct).astype(np.float32)
        spacing_xyz = ct.GetSpacing()                # (x, y, z) mm
        shape3d = ct_arr.shape                       # (z, y, x)

        # OR over all per-reader masks → ground-truth nodule region
        gt_mask = np.zeros(shape3d, dtype=bool)
        for msk_p in sorted(patient_dir.glob(f"{pid}_CT_Phy*-label.nrrd")):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(msk_p)))
            gt_mask |= (arr > 0)

        # Lung mask (HU window + fill)
        lung = _quick_lung_mask(ct_arr)

        # Candidate detection — diameters 3-30 voxels (≈ 3-30 mm if iso 1mm)
        cands = detect_candidates(ct_arr, lung_mask=lung,
                                   d_min=3.0, d_max=30.0,
                                   n_scales=5, nms_radius=3)
        cfg = AHSNConfig()

        for c in cands:
            block, _ = extract_block(ct_arr, c, boundary=2)
            if block.size == 0 or min(block.shape) < 5:
                continue
            try:
                desc = ahsn(block, mask=None, cfg=cfg)
            except Exception:
                continue
            label = 1 if gt_mask[c.z, c.y, c.x] else 0
            row = {
                "pid": pid,
                "z": c.z, "y": c.y, "x": c.x,
                "scale": float(c.scale),
                "diameter_vox": float(c.diameter),
                "dot": float(c.dot),
                "label": label,
            }
            for i, v in enumerate(desc.tolist()):
                row[f"ahsn_{i:03d}"] = float(v)
            rows.append(row)

        if not rows:
            rows.append({"pid": pid, "status": "no_candidates"})
    except Exception as e:
        rows = [{"pid": pid, "status": f"fatal:{type(e).__name__}",
                 "error": str(e),
                 "traceback": traceback.format_exc(limit=2)}]
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lidc-out", required=True,
                   help="Directory produced by `qr lidc convert-cohort`.")
    p.add_argument("--lidc-src", required=True,
                   help="Original LIDC-IDRI TCIA tree.")
    p.add_argument("--out", required=True, help="Output candidates CSV.")
    p.add_argument("--jobs", "-j", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    lidc_out = Path(args.lidc_out)
    patients = sorted(d for d in lidc_out.iterdir()
                      if d.is_dir() and (d / f"{d.name}_CT.nrrd").exists())
    if args.limit:
        patients = patients[: args.limit]
    print(f"found {len(patients)} converted patients", file=sys.stderr)

    work = [(pat.name, str(pat), args.lidc_src) for pat in patients]
    all_rows: list[dict] = []
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_detect_one, w): w[0] for w in work}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                fail += 1
                print(f"  ✘ {pid}: {e}", file=sys.stderr); continue
            n_cand = sum(1 for r in rows if "ahsn_000" in r)
            n_pos = sum(1 for r in rows if r.get("label") == 1)
            if n_cand:
                ok += 1
            else:
                fail += 1
            all_rows.extend(rows)
            print(f"  ✓ {pid}: {n_cand} candidate(s), {n_pos} positive",
                  file=sys.stderr)

    if not all_rows:
        print("no rows produced", file=sys.stderr); return 1

    keys = sorted({k for row in all_rows for k in row.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows → {args.out}  "
          f"({ok} ok / {fail} fail)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
