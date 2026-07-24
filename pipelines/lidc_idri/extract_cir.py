"""Extract qradiomics features from CIRDataset paper-grade masks.

The CIRDataset (Zenodo 6762573 / github.com/choilab-jefferson/CIR) contains
956 radiologist-QA/QC'ed spiculation/lobulation annotations on segmented
lung nodules:

    DATA/LIDC_spiculation/<PID>/<PID>_CT_{n}-all.nrrd          (CT crop)
    DATA/LIDC_spiculation/<PID>/<PID>_CT_{n}-all-label.nrrd    (nodule mask)
    DATA/LUNGx_spiculation/<PID>/<PID>_CT_{n}-seg.nrrd         (CT crop)
    DATA/LUNGx_spiculation/<PID>/<PID>_CT_{n}-seg-label.nrrd   (nodule mask)

This script walks one cohort and extracts the same feature schema as
`extract_features.py` so it plugs into the methods-comparison harness.

Outputs (per row): pid, nodule_id, n_voxels, volume_mm3, malignancy,
diagnosis, y_malignant + 1409 radiomics + 6 spiculation.
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

# Allow direct script invocation without PYTHONPATH.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pipelines.common.parallel import run_parallel_rows
from qradiomics.atomic import extract_features
from qradiomics.shape import spiculation_from_voxel


def _walk_lidc(root: Path):
    """Yield (pid, nodule_n, ct_path, mask_path) for LIDC."""
    for pat in sorted(root.iterdir()):
        if not pat.is_dir(): continue
        for ct in pat.glob(f"{pat.name}_CT_*-all.nrrd"):
            m = re.match(rf"{pat.name}_CT_(\d+)-all\.nrrd", ct.name)
            if not m: continue
            n = int(m.group(1))
            mask = pat / f"{pat.name}_CT_{n}-all-label.nrrd"
            if mask.exists():
                yield pat.name, n, ct, mask


def _walk_lungx(root: Path):
    """Yield (pid, nodule_n, ct_path, mask_path) for LUNGx (-seg- suffix)."""
    for pat in sorted(root.iterdir()):
        if not pat.is_dir(): continue
        for ct in pat.glob(f"{pat.name}_CT_*-seg.nrrd"):
            m = re.match(rf"{pat.name}_CT_(\d+)-seg\.nrrd", ct.name)
            if not m: continue
            n = int(m.group(1))
            mask = pat / f"{pat.name}_CT_{n}-seg-label.nrrd"
            if mask.exists():
                yield pat.name, n, ct, mask


def _process_one(args) -> dict:
    pid, n, ct_path, mask_path, malignancy, y_malig, diagnosis, do_spic = args
    row = {"pid": pid, "nodule_id": f"Nodule {n}",
           "malignancy": malignancy, "y_malignant": int(y_malig),
           "diagnosis": diagnosis}
    try:
        ct = sitk.ReadImage(str(ct_path))
        mk = sitk.ReadImage(str(mask_path))
        mk_arr = sitk.GetArrayFromImage(mk)
        n_vox = int((mk_arr > 0).sum())
        if n_vox < 8:
            row["status"] = f"degenerate_mask n_vox={n_vox}"; return row
        row["n_voxels"] = n_vox
        row["volume_mm3"] = n_vox * float(np.prod(ct.GetSpacing()))

        # Binarise the label (some CIR labels are uint8 with multiple class IDs)
        bin_mk = sitk.BinaryThreshold(mk, lowerThreshold=1, upperThreshold=10**6,
                                       insideValue=1, outsideValue=0)
        bin_mk.CopyInformation(ct)

        try:
            feats = extract_features(ct, bin_mk, label=1, geometry_tolerance=1e-3)
            for k, v in feats.items(): row[k] = v
            row["status_radiomics"] = "ok"
        except Exception as e:
            row["status_radiomics"] = f"error:{type(e).__name__}:{e}"

        if do_spic and n_vox < 200_000:
            try:
                spacing_zyx = (ct.GetSpacing()[2], ct.GetSpacing()[1], ct.GetSpacing()[0])
                bin_arr = (mk_arr > 0).astype(np.uint8)
                sf, _p, _d, _m = spiculation_from_voxel(bin_arr, spacing=spacing_zyx)
                row["spic_Np"] = sf.Np; row["spic_Na"] = sf.Na
                row["spic_Nl"] = sf.Nl; row["spic_Na_att"] = sf.Na_att
                row["spic_s1"] = sf.s1; row["spic_s2"] = sf.s2
                row["status_spic"] = "ok"
            except Exception as e:
                row["status_spic"] = f"error:{type(e).__name__}"
        else:
            row["status_spic"] = "skipped"
    except Exception as e:
        row["status"] = f"fatal:{type(e).__name__}:{e}"
        row["traceback"] = traceback.format_exc(limit=2)
    return row


def _load_lidc_malig_map(qradiomics_lidc_out: Path | None,
                        pm_ids_file: Path | None) -> dict[str, float]:
    """Build PID → mean malignancy map from previous LIDC extraction CSV."""
    if not qradiomics_lidc_out or not qradiomics_lidc_out.exists():
        return {}
    df = pd.read_csv(qradiomics_lidc_out, low_memory=False,
                     usecols=["pid", "malignancy", "nodule_id"])
    g = df.groupby("pid")["malignancy"].mean().to_dict()
    return g


def _load_lungx_truth(cal_xlsx: Path, test_xlsx: Path):
    cal = pd.read_excel(cal_xlsx); test = pd.read_excel(test_xlsx)
    cal = cal[cal["Scan Number"].notna()].copy()
    test = test[test["Scan Number"].notna()].copy()
    cal = cal.rename(columns={"Diagnosis": "diagnosis"})
    test = test.rename(columns={"Final Diagnosis": "diagnosis"})
    rows = []
    for _, r in pd.concat([cal[["Scan Number", "diagnosis"]],
                            test[["Scan Number", "diagnosis"]]]).iterrows():
        scan = str(r["Scan Number"]).strip()
        diag = str(r["diagnosis"]).lower().strip()
        y_malig = 1 if ("malig" in diag or "primary lung" in diag) else 0
        rows.append((scan, diag, y_malig))
    # PID may be upper/lower case mismatched; normalise to upper for LUNGx
    out = {}
    for scan, diag, y in rows:
        out[scan] = (diag, y)
        out[scan.upper()] = (diag, y)
        out[scan.lower()] = (diag, y)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cir-root", required=True,
                   help="DATA/ directory inside extracted CIRDataset_LCSR")
    p.add_argument("--cohort", required=True, choices=["lidc", "lungx"])
    p.add_argument("--out", required=True, help="Output features CSV")
    p.add_argument("--lidc-malig-csv",
                   help="Previous LIDC features.csv (used to look up malignancy)")
    p.add_argument("--lungx-cal-xlsx", help="LUNGx CalibrationSet xlsx")
    p.add_argument("--lungx-test-xlsx", help="LUNGx TestSet xlsx")
    p.add_argument("--skip-spiculation", action="store_true")
    p.add_argument("--jobs", "-j", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    root = Path(args.cir_root)
    if args.cohort == "lidc":
        walker = _walk_lidc(root / "LIDC_spiculation")
        malig_map = (_load_lidc_malig_map(Path(args.lidc_malig_csv), None)
                     if args.lidc_malig_csv else {})
        truth_map = {}
    else:
        walker = _walk_lungx(root / "LUNGx_spiculation")
        malig_map = {}
        truth_map = _load_lungx_truth(Path(args.lungx_cal_xlsx),
                                       Path(args.lungx_test_xlsx))

    work = []
    for pid, n, ct, mk in walker:
        if args.cohort == "lidc":
            malignancy = malig_map.get(pid, 0)
            # Derive binary (≥4=malig, ≤2=benign, drop 3) for LIDC labels
            y_malig = int(malignancy >= 4) if malignancy else 0
            diag = ("malig" if malignancy >= 4
                    else ("benign" if 0 < malignancy <= 2 else "unknown"))
        else:
            diag, y_malig = truth_map.get(pid, ("unknown", 0))
            malignancy = 5 if y_malig else 1
        work.append((pid, n, str(ct), str(mk),
                     float(malignancy), y_malig, diag,
                     not args.skip_spiculation))
    if args.limit: work = work[: args.limit]
    print(f"{len(work)} (pid, nodule) tasks for {args.cohort}", file=sys.stderr)

    def _fmt(key, rows: list[dict]) -> str:
        r = rows[0] if rows else {}
        return (f"status={r.get('status_radiomics', '?')[:15]} "
                f"spic={r.get('status_spic', '?')[:10]} n_vox={r.get('n_voxels', '?')}")

    def _is_ok(rows: list[dict]) -> bool:
        return bool(rows) and rows[0].get("status_radiomics") == "ok"

    run_parallel_rows(
        work, _process_one, args.jobs, args.out,
        key_fn=lambda w: f"{w[0]}#{w[1]}",
        format_success=_fmt, is_ok=_is_ok,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
