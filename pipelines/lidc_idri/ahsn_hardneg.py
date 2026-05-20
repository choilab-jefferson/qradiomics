"""LIDC AHSN with HARD negatives — closer to Choi 2014 CMPB reproduction.

The original paper's negatives are FALSE POSITIVES from a multi-threshold dot
detector — they are real, radiologist-discriminable non-nodule objects
(vessels, bronchi, junctions, etc.) that look spherical enough to pass the
detector.

LIDC XML provides ~22,000 `nonNodule` annotations — radiologist-marked
non-nodule objects in the same scans. These are exactly the hard negative
class the paper aims to discriminate against. We extract an AHSN feature
on each annotated nonNodule (centroid + block) and each annotated nodule.

This is much closer to the paper's experimental setup than random
lung-tissue sampling. Choi 2014 reports 0.996 AUC after wall elimination
(SVM-r, 180-dim AHSN, 10-fold CV on balanced 144+144 dataset, 84 patients).
Our reproduction uses 1018 multi-institutional patients with whatever
nodule + nonNodule annotations are present.
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
from qradiomics.io.lidc.extract import _uid_to_z
from qradiomics.shape import AHSNConfig, ahsn


def _patient_xml(lidc_src: Path, pid: str):
    for pid_, series_dir, xml in scan_lidc_dir(lidc_src):
        if pid_ == pid:
            return series_dir, xml
    return None, None


def _extract_block(ct_arr: np.ndarray, z: int, y: int, x: int,
                   block_size: int = 17):
    half = block_size // 2
    Z, Y, X = ct_arr.shape
    if not (half <= z < Z - half and half <= y < Y - half
            and half <= x < X - half):
        return None
    return ct_arr[z - half:z + half + 1,
                  y - half:y + half + 1,
                  x - half:x + half + 1]


def _process_one(args) -> list[dict]:
    pid, patient_dir, lidc_src, block_size = args
    rows: list[dict] = []
    try:
        patient_dir = Path(patient_dir)
        ct_path = patient_dir / f"{pid}_CT.nrrd"
        if not ct_path.exists():
            return [{"pid": pid, "status": "missing_ct"}]
        ct = sitk.ReadImage(str(ct_path))
        ct_arr = sitk.GetArrayFromImage(ct).astype(np.float32)
        Z, Y, X = ct_arr.shape

        series_dir, xml_path = _patient_xml(Path(lidc_src), pid)
        if xml_path is None:
            return [{"pid": pid, "status": "missing_xml"}]
        uid_to_z = _uid_to_z(series_dir)
        readers = parse_lidc_xml(xml_path)

        cfg = AHSNConfig()
        # ─── Positives = nodule centroids (one per reader-nodule) ───────
        for reader in readers:
            for nodule in reader.nodules:
                # Pick the middle ROI as the slice; centroid of polygon for x,y
                if not nodule.rois: continue
                roi = nodule.rois[len(nodule.rois) // 2]
                z = uid_to_z.get(roi.image_sop_uid)
                if z is None or not roi.x_coords or not roi.y_coords:
                    continue
                cx = int(round(sum(roi.x_coords) / len(roi.x_coords)))
                cy = int(round(sum(roi.y_coords) / len(roi.y_coords)))
                block = _extract_block(ct_arr, int(z), cy, cx, block_size)
                if block is None: continue
                try: desc = ahsn(block, mask=None, cfg=cfg)
                except Exception: continue
                row = {"pid": pid, "reader": reader.session_index,
                       "type": "nodule", "label": 1,
                       "object_id": nodule.nodule_id,
                       "z": int(z), "y": cy, "x": cx,
                       "malignancy": nodule.characteristics.malignancy}
                for i, v in enumerate(desc.tolist()):
                    row[f"ahsn_{i:03d}"] = float(v)
                rows.append(row)

        # ─── Negatives = nonNodule centroids ────────────────────────────
        for reader in readers:
            for nn in reader.non_nodules:
                z = uid_to_z.get(nn.image_sop_uid)
                if z is None: continue
                block = _extract_block(ct_arr, int(z),
                                        int(nn.y_coord), int(nn.x_coord),
                                        block_size)
                if block is None: continue
                try: desc = ahsn(block, mask=None, cfg=cfg)
                except Exception: continue
                row = {"pid": pid, "reader": reader.session_index,
                       "type": "nonNodule", "label": 0,
                       "object_id": nn.non_nodule_id,
                       "z": int(z), "y": int(nn.y_coord), "x": int(nn.x_coord),
                       "malignancy": 0}
                for i, v in enumerate(desc.tolist()):
                    row[f"ahsn_{i:03d}"] = float(v)
                rows.append(row)

        if not rows:
            rows.append({"pid": pid, "status": "no_annotations"})
    except Exception as e:
        rows = [{"pid": pid, "status": f"fatal:{type(e).__name__}",
                 "error": str(e), "traceback": traceback.format_exc(limit=2)}]
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lidc-out", required=True)
    p.add_argument("--lidc-src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--block-size", type=int, default=17)
    p.add_argument("--jobs", "-j", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    lidc_out = Path(args.lidc_out)
    patients = sorted(d for d in lidc_out.iterdir()
                      if d.is_dir() and (d / f"{d.name}_CT.nrrd").exists())
    if args.limit: patients = patients[: args.limit]
    print(f"found {len(patients)} converted patients", file=sys.stderr)

    work = [(p.name, str(p), args.lidc_src, args.block_size) for p in patients]
    all_rows = []
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_process_one, w): w[0] for w in work}
        for fut in as_completed(futs):
            pid = futs[fut]
            try: rows = fut.result()
            except Exception as e:
                fail += 1; print(f"  ✘ {pid}: {e}", file=sys.stderr); continue
            n_pos = sum(1 for r in rows if r.get("label") == 1)
            n_neg = sum(1 for r in rows if r.get("label") == 0)
            if n_pos + n_neg: ok += 1
            else: fail += 1
            all_rows.extend(rows)
            print(f"  ✓ {pid}: +{n_pos} / -{n_neg}", file=sys.stderr)

    keys = sorted({k for r in all_rows for k in r.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows → {args.out}  ({ok} ok / {fail} fail)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
