"""Directly validate qradiomics.shape spiculation port against LCSR reference output.

The CIRDataset (Zenodo 6762573) bundles LCSR's processed outputs alongside
the nodule masks:

    {pid}_CT_{n}-all-label.nrrd          — input nodule mask
    {pid}_CT_{n}-all-ard.nrrd            — LCSR area-distortion map
    {pid}_CT_{n}-all-spikes-label.nrrd   — LCSR peak class (1=spic, 2=lob, 3=att)

We:
  1. Run `spiculation_from_voxel(mask)` on each nodule.
  2. Count our peaks per class (Na = spic, Nl = lob, Na_att = attachment).
  3. Compare to the LCSR reference counts (extracted by counting connected
     components of class 1, 2, 3 in the spikes-label NRRD).
  4. Emit a per-nodule comparison CSV + Spearman correlation summary.

This is a direct *port-validation* against the original MATLAB pipeline's output.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label as cc_label

# Allow direct script invocation without PYTHONPATH.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pipelines.common.parallel import run_parallel_rows
from qradiomics.shape import spiculation_from_voxel


def _walk(root: Path, cohort: str):
    if cohort == "lidc":
        sub, suffix = "LIDC_spiculation", "all"
    else:
        sub, suffix = "LUNGx_spiculation", "seg"
    for pat in sorted((root / sub).iterdir()):
        if not pat.is_dir(): continue
        for mask in pat.glob(f"{pat.name}_CT_*-{suffix}-label.nrrd"):
            m = re.match(rf"{pat.name}_CT_(\d+)-{suffix}-label\.nrrd", mask.name)
            if not m: continue
            n = int(m.group(1))
            spikes = pat / f"{pat.name}_CT_{n}-{suffix}-spikes-label.nrrd"
            ard = pat / f"{pat.name}_CT_{n}-{suffix}-ard.nrrd"
            if spikes.exists():
                yield pat.name, n, mask, spikes, ard


def _count_lcsr_peaks(spikes_path: Path) -> dict[str, int]:
    """Count connected components per class in LCSR spikes-label NRRD."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(spikes_path)))
    out = {}
    for cls, name in ((1, "lcsr_Na"), (2, "lcsr_Nl"), (3, "lcsr_Na_att")):
        cc, n_cc = cc_label((arr == cls))
        out[name] = int(n_cc)
    return out


def _process_one(args):
    pid, n, mask_path, spikes_path, ard_path = args
    try:
        mk = sitk.ReadImage(str(mask_path))
        mk_arr = (sitk.GetArrayFromImage(mk) > 0).astype(np.uint8)
        if mk_arr.sum() < 8:
            return {"pid": pid, "nodule_n": n, "status": "small_mask"}
        spacing = mk.GetSpacing()
        spacing_zyx = (spacing[2], spacing[1], spacing[0])
        sf, peaks, dist, mesh = spiculation_from_voxel(mk_arr, spacing=spacing_zyx)
        row = {
            "pid": pid, "nodule_n": n,
            "n_voxels": int(mk_arr.sum()),
            "qr_Np": sf.Np, "qr_Na": sf.Na, "qr_Nl": sf.Nl, "qr_Na_att": sf.Na_att,
            "qr_s1": sf.s1, "qr_s2": sf.s2,
            "n_mesh_vertices": mesh.n_vertices,
        }
        # Compare to LCSR if available
        row.update(_count_lcsr_peaks(spikes_path))
        # Compare our area-distortion to LCSR's ard (rough — different coordinate systems)
        if ard_path.exists():
            ard_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(ard_path)))
            row["lcsr_ard_min"] = float(np.min(ard_arr))
            row["lcsr_ard_max"] = float(np.max(ard_arr))
            row["lcsr_ard_neg_voxels"] = int(np.sum(ard_arr < 0))
            row["qr_ard_min"] = float(np.min(dist))
            row["qr_ard_max"] = float(np.max(dist))
            row["qr_ard_neg_vertices"] = int(np.sum(dist < 0))
        return row
    except Exception as e:
        return {"pid": pid, "nodule_n": n, "status": f"err:{type(e).__name__}:{e}"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cir-root", required=True)
    p.add_argument("--cohort", choices=["lidc", "lungx"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", "-j", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    work = list(_walk(Path(args.cir_root), args.cohort))
    if args.limit: work = work[: args.limit]
    print(f"{len(work)} nodules", file=sys.stderr)

    def _fmt(key, rows: list[dict]) -> str:
        r = rows[0] if rows else {}
        return (f"qr Na={r.get('qr_Na', '?')} Nl={r.get('qr_Nl', '?')} "
                f"Na_att={r.get('qr_Na_att', '?')}  "
                f"lcsr Na={r.get('lcsr_Na', '?')} Nl={r.get('lcsr_Nl', '?')} "
                f"Na_att={r.get('lcsr_Na_att', '?')}")

    run_parallel_rows(
        work, _process_one, args.jobs, args.out,
        key_fn=lambda w: f"{w[0]}#{w[1]}",
        format_success=_fmt,
    )

    # Spearman correlation between qradiomics and LCSR peak counts
    try:
        import pandas as pd
        from scipy.stats import spearmanr
        df = pd.read_csv(args.out)
        df = df.dropna(subset=["qr_Na", "lcsr_Na"])
        for ours, theirs, label in (("qr_Na", "lcsr_Na", "Spiculations (Na)"),
                                     ("qr_Nl", "lcsr_Nl", "Lobulations (Nl)"),
                                     ("qr_Na_att", "lcsr_Na_att", "Attachments (Na_att)")):
            rho, p = spearmanr(df[ours], df[theirs])
            print(f"  Spearman ρ {label}: ours×LCSR = {rho:.3f}  (p={p:.2g}, n={len(df)})",
                  file=sys.stderr)
    except Exception as e:
        print(f"correlation skipped: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
