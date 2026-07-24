"""LIDC-IDRI per-(reader, nodule) radiomics + spiculation feature extractor.

Reproduces the feature-extraction stage of Choi 2014 CMPB (AHSN nodule
classification) and Choi 2021 CMPB (spiculation quantification) using the
qradiomics-public atomic core + qradiomics.shape spiculation pipeline.

Inputs:
    --lidc-out  Directory produced by `qr lidc convert-cohort`. Expected
                tree:  <root>/<pid>/{<pid>_CT.nrrd, <pid>_CT_Phy*-label.nrrd,
                                      <pid>_nodules.csv}
                (Per-nodule masks are recomputed in memory from the XML if
                they are not present — no extra disk requirement.)
    --lidc-src  Original LIDC-IDRI tree (XML + DICOM) — used to look up XML
                for per-nodule rasterisation when per-nodule masks are absent.

Outputs:
    --out features.csv    One row per (pid, reader, nodule_id) with columns:
        * pid, reader, nodule_id, n_voxels, volume_mm3, malignancy, …
        * <radiomics>      atomic.extract_features (1409 features)
        * spic_Np, spic_Na, spic_Nl, spic_Na_att, spic_s1, spic_s2

Skips nodules whose voxel count < --min-voxels (default 8; mesh extraction
fails on degenerate masks).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Allow direct script invocation without PYTHONPATH (run.sh sets it via
# _qr_resolve.sh, but a developer poking at the script directly should
# still work).
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pipelines.common.parallel import run_parallel_rows
from qradiomics.atomic import extract_features
from qradiomics.io.lidc import (
    parse_lidc_xml,
    scan_lidc_dir,
)
from qradiomics.io.lidc.extract import _rasterise_nodule, _uid_to_z
from qradiomics.shape import spiculation_from_voxel


def _patient_xml(lidc_src: Path, pid: str) -> Path | None:
    """Locate the annotation XML for `pid` under the original LIDC tree."""
    pid_base = pid.split("-")[0:3]  # 'LIDC-IDRI-NNNN'
    pid_root = lidc_src / "-".join(pid_base) if not (lidc_src / pid).exists() else lidc_src / pid
    if not pid_root.exists():
        pid_root = lidc_src / pid
    if not pid_root.exists():
        return None
    # Pick the first XML under the first matching CT series.
    for pid_, series_dir, xml in scan_lidc_dir(lidc_src):
        if pid_ == pid:
            return xml
    return None


def _patient_dicom_dir(lidc_src: Path, pid: str) -> Path | None:
    for pid_, series_dir, _xml in scan_lidc_dir(lidc_src):
        if pid_ == pid:
            return series_dir
    return None


def _extract_one_patient(args) -> list[dict]:
    """Worker: emit one features row per (reader, nodule) for a patient."""
    pid, patient_dir, lidc_src, params_file, min_voxels = args
    rows: list[dict] = []
    try:
        patient_dir = Path(patient_dir)
        ct_nrrd = patient_dir / f"{pid}_CT.nrrd"
        if not ct_nrrd.exists():
            return [{"pid": pid, "status": "missing_ct"}]
        ct = sitk.ReadImage(str(ct_nrrd))
        spacing_xyz = ct.GetSpacing()                # (x, y, z) mm
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        size = ct.GetSize()
        shape3d = (size[2], size[1], size[0])

        # XML: try patient_dir first, fall back to lidc_src.
        xml_candidates = list(patient_dir.glob("*.xml"))
        xml_path = xml_candidates[0] if xml_candidates else (
            _patient_xml(Path(lidc_src), pid) if lidc_src else None)
        if not xml_path:
            return [{"pid": pid, "status": "missing_xml"}]

        # Need DICOM series for UID→z map.
        dicom_dir = _patient_dicom_dir(Path(lidc_src), pid) if lidc_src else None
        if dicom_dir is None:
            return [{"pid": pid, "status": "missing_dicom"}]

        uid_to_z = _uid_to_z(dicom_dir)
        readers = parse_lidc_xml(xml_path)

        for reader in readers:
            for nodule in reader.nodules:
                mask3d = _rasterise_nodule(nodule, uid_to_z, shape3d)
                n_vox = int(mask3d.sum())
                if n_vox < min_voxels:
                    continue
                msk_img = sitk.GetImageFromArray(mask3d)
                msk_img.CopyInformation(ct)

                row = {
                    "pid": pid,
                    "reader": reader.session_index,
                    "nodule_id": nodule.nodule_id,
                    "n_voxels": n_vox,
                    "volume_mm3": n_vox * float(np.prod(spacing_xyz)),
                    "malignancy": nodule.characteristics.malignancy,
                    "subtlety": nodule.characteristics.subtlety,
                    "calcification": nodule.characteristics.calcification,
                    "sphericity_score": nodule.characteristics.sphericity,
                    "margin": nodule.characteristics.margin,
                    "lobulation_score": nodule.characteristics.lobulation,
                    "spiculation_score": nodule.characteristics.spiculation,
                    "texture": nodule.characteristics.texture,
                }

                # Atomic radiomics
                try:
                    feats = extract_features(
                        ct, msk_img, params_file=params_file,
                        label=1, geometry_tolerance=1e-3)
                    for k, v in feats.items():
                        row[k] = v
                    row["status_radiomics"] = "ok"
                except Exception as e:
                    row["status_radiomics"] = f"error:{type(e).__name__}"

                # Spiculation (qradiomics.shape)
                try:
                    sfeat, _peaks, _dist, _mesh = spiculation_from_voxel(
                        mask3d.astype(np.uint8),
                        spacing=spacing_zyx,
                    )
                    row["spic_Np"] = sfeat.Np
                    row["spic_Na"] = sfeat.Na
                    row["spic_Nl"] = sfeat.Nl
                    row["spic_Na_att"] = sfeat.Na_att
                    row["spic_s1"] = sfeat.s1
                    row["spic_s2"] = sfeat.s2
                    row["status_spic"] = "ok"
                except Exception as e:
                    row["status_spic"] = f"error:{type(e).__name__}"

                rows.append(row)
        if not rows:
            rows.append({"pid": pid, "status": "no_nodules"})
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
                   help="Original LIDC-IDRI tree (XML + DICOM).")
    p.add_argument("--out", required=True, help="Output features CSV.")
    p.add_argument("--params",
                   help="PyRadiomics parameter YAML (default: bundled extractor defaults).")
    p.add_argument("--min-voxels", type=int, default=8,
                   help="Skip nodules with < this many voxels (mesh failure guard).")
    p.add_argument("--jobs", "-j", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N patients.")
    args = p.parse_args()

    lidc_out = Path(args.lidc_out)
    patients = sorted(d for d in lidc_out.iterdir()
                      if d.is_dir() and (d / f"{d.name}_CT.nrrd").exists())
    if args.limit:
        patients = patients[: args.limit]
    print(f"found {len(patients)} converted patients under {lidc_out}",
          file=sys.stderr)

    work = [(pat.name, str(pat), args.lidc_src, args.params, args.min_voxels)
            for pat in patients]

    def _is_ok(rows: list[dict]) -> bool:
        return any(r.get("status_radiomics") == "ok"
                   or "feat" in {k.split("_")[0] for k in r.keys()}
                   for r in rows)

    all_rows = run_parallel_rows(
        work, _extract_one_patient, args.jobs, args.out,
        is_ok=_is_ok, skip_write_if_empty=True, summary_leading_blank=True,
    )
    if not all_rows:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
