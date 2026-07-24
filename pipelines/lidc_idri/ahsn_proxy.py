"""LIDC AHSN classification — Choi 2014 CMPB proxy reproduction.

Whole-volume Hessian-based candidate detection (5 scales × 512³) in pure
Python is too slow to run across the full LIDC cohort. This script
implements a *proxy* AHSN reproduction:

  Positives: every annotated nodule centroid from the per-reader masks.
  Negatives: random lung-tissue voxels sampled N× the positive count,
             at least 20 mm (heuristic) away from any annotated mask.

Each candidate gets an AHSN descriptor (qradiomics.shape.ahsn) computed on
a block extracted around it. This matches the *evaluation* step in the
paper (Sec. 4.2: classification of nodule vs non-nodule blocks) without
re-running the multi-threshold candidate generator.

Output: candidates.csv with (pid, z, y, x, label, ahsn_000…ahsn_K).
Then train a classifier and compute ROC AUC.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, distance_transform_edt

# Allow direct script invocation without PYTHONPATH.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pipelines.common.parallel import run_parallel_rows
from qradiomics.shape import AHSNConfig, ahsn


def _sample_candidates(patient_dir: Path, pid: str,
                       neg_per_pos: int = 3,
                       neg_min_distance_mm: float = 20.0,
                       seed: int = 0) -> tuple[list[tuple[int, int, int, int]],
                                                sitk.Image]:
    """Return [(z,y,x,label)…] candidate list and the CT image."""
    ct = sitk.ReadImage(str(patient_dir / f"{pid}_CT.nrrd"))
    ct_arr = sitk.GetArrayFromImage(ct).astype(np.float32)
    spacing = ct.GetSpacing()                            # (x, y, z) mm
    shape3d = ct_arr.shape

    # OR reader masks → ground truth nodule region
    gt = np.zeros(shape3d, dtype=bool)
    for msk_p in sorted(patient_dir.glob(f"{pid}_CT_Phy*-label.nrrd")):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(msk_p)))
        gt |= (arr > 0)
    if not gt.any():
        return [], ct

    # Positives = centroid of each connected component
    from scipy.ndimage import label as cc_label, center_of_mass
    cc, n_cc = cc_label(gt)
    pos = []
    for cc_id in range(1, n_cc + 1):
        com = center_of_mass(cc == cc_id)
        pos.append((int(round(com[0])), int(round(com[1])),
                    int(round(com[2])), 1))

    # Quick lung mask (HU window) ∩ (distance to gt > threshold)
    lung = (ct_arr > -900) & (ct_arr < -400)
    # Voxel distance in z, y, x
    vox_z, vox_y, vox_x = spacing[2], spacing[1], spacing[0]
    # Distance from gt in voxels (then scale)
    dist_vox = distance_transform_edt(~gt)
    # Convert to mm using minimum voxel spacing as conservative scaling
    min_vox = min(vox_z, vox_y, vox_x)
    far_from_gt = (dist_vox * min_vox) > neg_min_distance_mm
    neg_candidates = np.argwhere(lung & far_from_gt)

    rng = np.random.default_rng(seed)
    n_neg = neg_per_pos * len(pos)
    if len(neg_candidates) > n_neg:
        idx = rng.choice(len(neg_candidates), size=n_neg, replace=False)
        neg_candidates = neg_candidates[idx]
    neg = [(int(z), int(y), int(x), 0) for z, y, x in neg_candidates]

    return pos + neg, ct


def _extract_block(ct_arr: np.ndarray, z: int, y: int, x: int,
                    block_size: int = 17) -> np.ndarray | None:
    half = block_size // 2
    Z, Y, X = ct_arr.shape
    if not (half <= z < Z - half and half <= y < Y - half
            and half <= x < X - half):
        return None
    return ct_arr[z - half:z + half + 1,
                  y - half:y + half + 1,
                  x - half:x + half + 1]


def _process_one(args) -> list[dict]:
    pid, patient_dir, neg_per_pos, block_size = args
    rows: list[dict] = []
    try:
        cands, ct = _sample_candidates(Path(patient_dir), pid,
                                        neg_per_pos=neg_per_pos)
        if not cands:
            return [{"pid": pid, "status": "no_gt_nodules"}]
        ct_arr = sitk.GetArrayFromImage(ct).astype(np.float32)
        cfg = AHSNConfig()

        for z, y, x, label in cands:
            block = _extract_block(ct_arr, z, y, x, block_size)
            if block is None:
                continue
            try:
                desc = ahsn(block, mask=None, cfg=cfg)
            except Exception:
                continue
            row = {"pid": pid, "z": z, "y": y, "x": x, "label": label}
            for i, v in enumerate(desc.tolist()):
                row[f"ahsn_{i:03d}"] = float(v)
            rows.append(row)
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
    p.add_argument("--out", required=True, help="Output candidates CSV.")
    p.add_argument("--neg-per-pos", type=int, default=3,
                   help="How many negatives to sample per positive (default 3).")
    p.add_argument("--block-size", type=int, default=17,
                   help="Cubic block edge length in voxels (default 17 ≈ 17mm if iso).")
    p.add_argument("--jobs", "-j", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    lidc_out = Path(args.lidc_out)
    patients = sorted(d for d in lidc_out.iterdir()
                      if d.is_dir() and (d / f"{d.name}_CT.nrrd").exists())
    if args.limit:
        patients = patients[: args.limit]
    print(f"found {len(patients)} converted patients", file=sys.stderr)

    work = [(p.name, str(p), args.neg_per_pos, args.block_size)
            for p in patients]

    def _fmt(_pid, rows: list[dict]) -> str:
        n_pos = sum(1 for r in rows if r.get("label") == 1)
        n_neg = sum(1 for r in rows if r.get("label") == 0)
        return f"+{n_pos} / -{n_neg}"

    def _is_ok(rows: list[dict]) -> bool:
        n_pos = sum(1 for r in rows if r.get("label") == 1)
        n_neg = sum(1 for r in rows if r.get("label") == 0)
        return (n_pos + n_neg) > 0

    all_rows = run_parallel_rows(
        work, _process_one, args.jobs, args.out,
        format_success=_fmt, is_ok=_is_ok,
        skip_write_if_empty=True, summary_leading_blank=True,
    )
    if not all_rows:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
