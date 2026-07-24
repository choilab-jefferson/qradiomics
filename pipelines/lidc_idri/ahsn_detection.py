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
import sys
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Allow direct script invocation without PYTHONPATH.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pipelines.common.parallel import run_parallel_rows
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

    def _fmt(_pid, rows: list[dict]) -> str:
        n_cand = sum(1 for r in rows if "ahsn_000" in r)
        n_pos = sum(1 for r in rows if r.get("label") == 1)
        return f"{n_cand} candidate(s), {n_pos} positive"

    all_rows = run_parallel_rows(
        work, _detect_one, args.jobs, args.out,
        format_success=_fmt,
        is_ok=lambda rows: any("ahsn_000" in r for r in rows),
        skip_write_if_empty=True, summary_leading_blank=True,
    )
    if not all_rows:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
