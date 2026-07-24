"""Voxelize our mesh-based peak detections and compare to LCSR's reference.

`qradiomics.shape.spiculation_from_voxel`:
  voxel mask  →  marching cubes mesh  →  spherical parameterization  →
  area-distortion map (on mesh vertices)  →  peaks (mesh-vertex groups)

LCSR's reference `-spikes-label.nrrd` is in voxel space (class 1=spic, 2=lob,
3=att). To compare directly we **voxelize** our mesh peaks back onto the
input voxel grid:

  for each peak: take its member vertex indices → look up their mesh.vertices
  (mm coords) → convert to voxel indices via spacing → splat label.

Outputs per nodule:
  - dice_spic, dice_lob, dice_att       (Dice overlap to LCSR class voxels)
  - peak_voxel_count_qr, peak_voxel_count_lcsr (raw voxel totals)

Optionally exports our mesh as OBJ + the voxelised peak labels as NRRD so the
output matches the CIRDataset format exactly. This lets us "produce CIR-like
mesh data" directly from qradiomics-public.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Allow direct script invocation without PYTHONPATH.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pipelines.common.parallel import run_parallel_rows
from qradiomics.shape import classify_peak, spiculation_from_voxel


def _peak_voxels(peak_members: np.ndarray, mesh_vertices: np.ndarray,
                 spacing_zyx: tuple[float, float, float],
                 shape: tuple[int, int, int]) -> set[tuple[int, int, int]]:
    """Map peak vertex indices → voxel coordinates (z,y,x)."""
    pts = mesh_vertices[peak_members]      # (M, 3)  mm — order matches voxel_to_mesh (z, y, x)
    vox = np.round(pts / np.asarray(spacing_zyx)).astype(int)
    Z, Y, X = shape
    out = set()
    for z, y, x in vox:
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            out.add((int(z), int(y), int(x)))
    return out


def _voxelise_peaks(mesh, peaks, classes, spacing_zyx, shape, dilate: int = 1):
    """Build a 3D label volume from our mesh-vertex peaks.

    Each peak's vertex coords are splatted then morphologically dilated `dilate`
    iterations so the comparison is region-vs-region (LCSR's spikes-label
    voxels are dilated surface patches, not point clouds).
    """
    from scipy.ndimage import binary_dilation
    out = np.zeros(shape, dtype=np.uint8)
    class_to_label = {"spiculation": 1, "lobulation": 2, "attachment": 3}
    for peak, cls in zip(peaks, classes):
        lbl = class_to_label.get(cls)
        if lbl is None: continue
        m = np.zeros(shape, dtype=bool)
        for z, y, x in _peak_voxels(peak.members, mesh.vertices, spacing_zyx, shape):
            m[z, y, x] = True
        if dilate > 0:
            m = binary_dilation(m, iterations=dilate)
        # Don't overwrite an existing higher-priority label (spic > lob > att)
        out[m & (out == 0)] = lbl
    return out


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0; b = b > 0
    denom = a.sum() + b.sum()
    if denom == 0: return float("nan")
    return float(2 * np.logical_and(a, b).sum() / denom)


def _export_obj(mesh, out_path: Path):
    """Write a minimal Wavefront OBJ file for the mesh."""
    with open(out_path, "w") as f:
        f.write(f"# qradiomics mesh ({mesh.n_vertices} vertices, "
                f"{mesh.n_faces} faces)\n")
        for v in mesh.vertices:
            # OBJ is (x, y, z) — mesh.vertices stores (z, y, x)
            f.write(f"v {v[2]:.4f} {v[1]:.4f} {v[0]:.4f}\n")
        for face in mesh.faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def _process_one(args):
    pid, n, mask_path, ref_spikes_path, ct_path, out_dir, export = args
    try:
        mk = sitk.ReadImage(str(mask_path))
        mk_arr = (sitk.GetArrayFromImage(mk) > 0).astype(np.uint8)
        if mk_arr.sum() < 8:
            return {"pid": pid, "nodule_n": n, "status": "small_mask"}
        spacing = mk.GetSpacing()
        spacing_zyx = (spacing[2], spacing[1], spacing[0])
        sf, peaks, dist, mesh = spiculation_from_voxel(mk_arr, spacing=spacing_zyx)
        classes = [classify_peak(p) for p in peaks]
        qr_vox = _voxelise_peaks(mesh, peaks, classes, spacing_zyx, mk_arr.shape)

        ref = sitk.GetArrayFromImage(sitk.ReadImage(str(ref_spikes_path)))
        row = {
            "pid": pid, "nodule_n": n,
            "n_voxels_mask": int(mk_arr.sum()),
            "n_peaks_qr": len(peaks),
            "n_spic_qr": classes.count("spiculation"),
            "n_lob_qr":  classes.count("lobulation"),
            "n_att_qr":  classes.count("attachment"),
            "vox_spic_qr": int((qr_vox == 1).sum()),
            "vox_lob_qr":  int((qr_vox == 2).sum()),
            "vox_att_qr":  int((qr_vox == 3).sum()),
            "vox_spic_lcsr": int((ref == 1).sum()),
            "vox_lob_lcsr":  int((ref == 2).sum()),
            "vox_att_lcsr":  int((ref == 3).sum()),
            "dice_spic": _dice(qr_vox == 1, ref == 1),
            "dice_lob":  _dice(qr_vox == 2, ref == 2),
            "dice_att":  _dice(qr_vox == 3, ref == 3),
            "dice_any":  _dice(qr_vox > 0,  ref > 0),
        }

        if export:
            pat_out = Path(out_dir) / pid
            pat_out.mkdir(parents=True, exist_ok=True)
            # Voxelised peaks (matches LCSR NRRD format)
            qr_img = sitk.GetImageFromArray(qr_vox)
            qr_img.CopyInformation(mk)
            sitk.WriteImage(qr_img, str(pat_out / f"{pid}_CT_{n}-qr-spikes-label.nrrd"))
            # Mesh OBJ
            _export_obj(mesh, pat_out / f"{pid}_CT_{n}-qr-mesh.obj")
            row["exported"] = str(pat_out)

        return row
    except Exception as e:
        return {"pid": pid, "nodule_n": n, "status": f"err:{type(e).__name__}:{e}"}


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
            ct = pat / f"{pat.name}_CT_{n}-{suffix}.nrrd"
            if spikes.exists():
                yield pat.name, n, mask, spikes, ct


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cir-root", required=True)
    p.add_argument("--cohort", choices=["lidc", "lungx"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--export-dir", default=None,
                   help="If set, export OBJ mesh + voxelised peak NRRD here.")
    p.add_argument("--jobs", "-j", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    work = [(pid, n, str(m), str(s), str(c),
             args.export_dir, bool(args.export_dir))
            for pid, n, m, s, c in _walk(Path(args.cir_root), args.cohort)]
    if args.limit: work = work[: args.limit]
    print(f"{len(work)} nodules", file=sys.stderr)

    def _fmt(key, rows: list[dict]) -> str:
        r = rows[0] if rows else {}
        if 'dice_spic' in r:
            return (f"dice spic={r.get('dice_spic', '?'):.3f} "
                    f"lob={r.get('dice_lob', '?'):.3f}  "
                    f"vox qr/lcsr spic={r.get('vox_spic_qr', '?')}/{r.get('vox_spic_lcsr', '?')}")
        return f"⚠ {r}"

    run_parallel_rows(
        work, _process_one, args.jobs, args.out,
        key_fn=lambda w: f"{w[0]}#{w[1]}",
        format_success=_fmt,
    )

    # Aggregate dice summary
    try:
        import pandas as pd
        df = pd.read_csv(args.out)
        for k in ("dice_spic", "dice_lob", "dice_att", "dice_any"):
            if k in df:
                vals = df[k].dropna()
                if len(vals):
                    print(f"  {k}: median={vals.median():.3f}  mean={vals.mean():.3f}  "
                          f"n={len(vals)}", file=sys.stderr)
    except Exception as e:
        print(f"agg skipped: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
