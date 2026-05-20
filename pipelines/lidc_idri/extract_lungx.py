"""LUNGx (SPIE-AAPM Lung CT Challenge) feature extraction.

The LUNGx dataset distributes only nodule **centroids** (x, y, image#) and
a binary benign/malignant diagnosis — no segmentation masks. To get
features that match the LIDC pipeline, we generate a per-nodule mask via
intensity-based region growing seeded at the published centroid.

Outputs one row per (pid, nodule_number) with the same column schema as
`extract_features.py`, so `reproduce_papers.py` can mix LIDC + LUNGx.

Truth files (download with `curl` from TCIA):
    CalibrationSet_NoduleData.xlsx        (14 patients, IDs CT-Training-lc001…)
    TestSet_NoduleData_PublicRelease_wTruth.xlsx  (60 patients, IDs LUNGx-CT001…)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from qradiomics.atomic import extract_features
from qradiomics.io.dicom import load_dicom_series
from qradiomics.shape import spiculation_from_voxel


def _load_truth(cal_xlsx: Path, test_xlsx: Path) -> pd.DataFrame:
    """Combine calibration + test truth into one long table."""
    cal = pd.read_excel(cal_xlsx)
    test = pd.read_excel(test_xlsx)
    # Drop legend/footer rows where Scan Number is NaN
    cal = cal[cal["Scan Number"].notna()].copy()
    test = test[test["Scan Number"].notna()].copy()
    # Calibration has no Nodule Number col (1 nodule per patient)
    cal = cal.rename(columns={"Diagnosis": "diagnosis"})
    cal["Nodule Number"] = 1
    test = test.rename(columns={"Final Diagnosis": "diagnosis"})
    cols = ["Scan Number", "Nodule Number",
            "Nodule Center x,y Position*", "Nodule Center Image", "diagnosis"]
    cal = cal[cols]; test = test[cols]
    truth = pd.concat([cal, test], ignore_index=True)
    truth = truth.rename(columns={
        "Scan Number": "scan",
        "Nodule Number": "nodule_num",
        "Nodule Center x,y Position*": "xy_str",
        "Nodule Center Image": "image_no",
    })
    truth["nodule_num"] = truth["nodule_num"].fillna(1).astype(int)
    truth["diagnosis"] = truth["diagnosis"].astype(str).str.lower().str.strip()
    truth["y_malignant"] = (truth["diagnosis"].str.contains("malig|primary lung")).astype(int)
    return truth


def _parse_xy(s) -> tuple[int, int] | None:
    """Parse xy string. Two formats:
        Test: "x, y"     (comma-separated, decimal coords)
        Cal:  "xxxYYY"   (6-digit packed: first 3 = x, last 3 = y; 512×512 image)
    """
    if pd.isna(s): return None
    s = str(s).strip()
    m = re.match(r"(\d+)[\s,]+(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Calibration set uses combined-digit format like "120325" — split halves
    s_clean = s.replace(".0", "")
    if s_clean.isdigit() and len(s_clean) == 6:
        return int(s_clean[:3]), int(s_clean[3:])
    return None


def _find_dicom_dir(lungx_root: Path, scan: str) -> Path | None:
    """LUNGx layout: <root>/<scan>/<study-uid>/<series-uid>/*.dcm"""
    base = lungx_root / scan
    if not base.exists():
        # case-insensitive lookup (CT-Training-LC001 vs CT-Training-lc001)
        for d in lungx_root.iterdir():
            if d.name.lower() == scan.lower():
                base = d; break
        else:
            return None
    # Find the CT series with the most slices
    best = None; best_n = 0
    for study in base.iterdir():
        if not study.is_dir(): continue
        for ser in study.iterdir():
            if not ser.is_dir(): continue
            n = len(list(ser.glob("*.dcm")))
            if n > best_n:
                best_n = n; best = ser
    return best


def _region_grow_seed(ct_arr: np.ndarray, z: int, y: int, x: int,
                      lower_hu: float = -500, upper_hu: float = 200,
                      max_radius: int = 12) -> np.ndarray:
    """Bounded seed region growing around (z,y,x).

    A small (max_radius*2+1)³ block is HU-thresholded and connected-component
    grown from the seed. The HU lower bound is **-500 HU** to keep the
    growth on soft-tissue density (nodule) rather than expanding into
    surrounding lung parenchyma (which sits at -700 to -900 HU).

    Returns full-volume binary mask.
    """
    from scipy.ndimage import label as cc_label

    Z, Y, X = ct_arr.shape
    z0, z1 = max(0, z - max_radius), min(Z, z + max_radius + 1)
    y0, y1 = max(0, y - max_radius), min(Y, y + max_radius + 1)
    x0, x1 = max(0, x - max_radius), min(X, x + max_radius + 1)
    block = ct_arr[z0:z1, y0:y1, x0:x1]
    bin_ = (block > lower_hu) & (block < upper_hu)
    cc, _ = cc_label(bin_)
    seed_cc = cc[z - z0, y - y0, x - x0]
    if seed_cc == 0:
        # Fall back to a small sphere around the seed (radius ~5 voxels).
        zz, yy, xx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1),
                                  np.arange(x0, x1), indexing="ij")
        sphere = ((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2) <= 5 ** 2
        out = np.zeros_like(ct_arr, dtype=np.uint8)
        out[z0:z1, y0:y1, x0:x1] = sphere.astype(np.uint8)
        return out
    out = np.zeros_like(ct_arr, dtype=np.uint8)
    out[z0:z1, y0:y1, x0:x1] = (cc == seed_cc).astype(np.uint8)
    return out


def _process_one(args) -> list[dict]:
    scan, nodule_num, z, y, x, diagnosis, y_malig, dicom_dir, do_spic = args
    rows = []
    try:
        ct = load_dicom_series(str(dicom_dir))
        ct_arr = sitk.GetArrayFromImage(ct).astype(np.float32)
        # LUNGx image# is 1-indexed slice from "bottom" — TCIA stores DICOMs by
        # InstanceNumber; we use (image_no - 1) as z. Different orderings may
        # need adjustment; verify on one case.
        Z, Y, X = ct_arr.shape
        z_idx = int(z) - 1
        if not (0 <= z_idx < Z and 0 <= y < Y and 0 <= x < X):
            return [{"scan": scan, "status": f"oob z={z_idx} y={y} x={x} "
                                              f"shape={ct_arr.shape}"}]

        mask = _region_grow_seed(ct_arr, z_idx, int(y), int(x))
        n_vox = int(mask.sum())
        if n_vox < 8:
            return [{"scan": scan, "status": f"degenerate_mask n_vox={n_vox}"}]
        msk_img = sitk.GetImageFromArray(mask)
        msk_img.CopyInformation(ct)

        row = {
            "pid": scan,
            "reader": 1,
            "nodule_id": f"Nodule {nodule_num}",
            "n_voxels": n_vox,
            "volume_mm3": n_vox * float(np.prod(ct.GetSpacing())),
            "diagnosis": diagnosis,
            "y_malignant": int(y_malig),
            "malignancy": (5 if y_malig else 1),    # cast to LIDC scale for paper-compare path
        }

        try:
            feats = extract_features(ct, msk_img, label=1, geometry_tolerance=1e-3)
            for k, v in feats.items():
                row[k] = v
            row["status_radiomics"] = "ok"
        except Exception as e:
            row["status_radiomics"] = f"error:{type(e).__name__}"

        if do_spic and n_vox < 50_000:           # skip huge masks (lung leak)
            try:
                spacing_zyx = (ct.GetSpacing()[2], ct.GetSpacing()[1], ct.GetSpacing()[0])
                sf, _p, _d, _m = spiculation_from_voxel(mask, spacing=spacing_zyx)
                row["spic_Np"] = sf.Np; row["spic_Na"] = sf.Na
                row["spic_Nl"] = sf.Nl; row["spic_Na_att"] = sf.Na_att
                row["spic_s1"] = sf.s1; row["spic_s2"] = sf.s2
                row["status_spic"] = "ok"
            except Exception as e:
                row["status_spic"] = f"error:{type(e).__name__}"
        else:
            row["status_spic"] = "skipped"

        rows.append(row)
    except Exception as e:
        rows = [{"scan": scan, "status": f"fatal:{type(e).__name__}",
                 "error": str(e), "traceback": traceback.format_exc(limit=2)}]
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lungx-root", required=True,
                   help="LUNGx DICOM root (one subdir per scan).")
    p.add_argument("--truth-cal", required=True,
                   help="CalibrationSet_NoduleData.xlsx")
    p.add_argument("--truth-test", required=True,
                   help="TestSet_NoduleData_PublicRelease_wTruth.xlsx")
    p.add_argument("--out", required=True, help="Output features CSV.")
    p.add_argument("--jobs", "-j", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-spiculation", action="store_true",
                   help="Skip spiculation features (radiomics only — much faster)")
    args = p.parse_args()

    truth = _load_truth(Path(args.truth_cal), Path(args.truth_test))
    print(f"{len(truth)} nodules in truth list", file=sys.stderr)
    if args.limit:
        truth = truth.head(args.limit)

    work = []
    for _, r in truth.iterrows():
        xy = _parse_xy(r["xy_str"])
        if xy is None:
            print(f"  ✘ {r['scan']}: cannot parse xy_str={r['xy_str']}",
                  file=sys.stderr); continue
        x, y = xy
        dicom_dir = _find_dicom_dir(Path(args.lungx_root), r["scan"])
        if dicom_dir is None:
            print(f"  ✘ {r['scan']}: dicom_dir missing", file=sys.stderr); continue
        work.append((r["scan"], int(r["nodule_num"]),
                     int(r["image_no"]), int(y), int(x),
                     r["diagnosis"], int(r["y_malignant"]),
                     str(dicom_dir), not args.skip_spiculation))
    print(f"  → {len(work)} tasks queued", file=sys.stderr)

    all_rows = []; ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_process_one, w): w[0] for w in work}
        for fut in as_completed(futs):
            scan = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                fail += 1; print(f"  ✘ {scan}: {e}", file=sys.stderr); continue
            ok_row = sum(1 for r in rows if r.get("status_radiomics") == "ok")
            if ok_row: ok += 1
            else:      fail += 1
            all_rows.extend(rows)
            print(f"  ✓ {scan}: {len(rows)} row(s)", file=sys.stderr)

    keys = sorted({k for r in all_rows for k in r.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows → {args.out}  ({ok} ok / {fail} fail)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
