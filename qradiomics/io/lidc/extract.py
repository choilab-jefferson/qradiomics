"""LIDC patient → CT NRRD + per-reader mask NRRDs + characteristics CSV.

Port of `main_image_to_nrrd.m`. Uses qradiomics' existing DICOM reader
(`qradiomics.io.dicom.load_dicom_series`) for the CT volume; this file
adds:

  * `_uid_to_z(dicom_series_dir)` — map DICOM SOPInstanceUID → 0-indexed
    z slice (replaces `fn_uid_to_zindex.m`).
  * `_rasterise_polygon(x, y, shape)` — polygon → 2D binary mask
    (replaces MATLAB ``poly2mask``; uses skimage.draw.polygon).
  * `convert_patient(patient_dir, out_dir, pid)` — top-level driver that
    walks one LIDC patient, parses XML, writes CT + per-reader masks +
    nodules.csv.
  * `scan_lidc_dir(root)` — walk a TCIA-style LIDC-IDRI tree and yield
    one `(pid, dicom_series_dir, xml_path)` per CT series (replaces
    `fn_scan_pid.m`).
"""
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from .parse_xml import LIDCNodule, parse_lidc_xml


# ─── DICOM walk ─────────────────────────────────────────────────────────────

def scan_lidc_dir(root: str | Path) -> Iterator[Tuple[str, Path, Optional[Path]]]:
    """Walk a TCIA-style LIDC-IDRI tree.

    Expected layout (matches TCIA download):

        root/
          LIDC-IDRI-NNNN/
            <study-uid>/
              <series-uid>/
                *.dcm
                *.xml          (annotation, optional — non-CT series have none)

    Yields one `(pid, dicom_series_dir, xml_path_or_None)` per CT series
    found. If a patient has multiple CT series (rare in LIDC), each gets a
    suffix `-1`, `-2`, … (matching the MATLAB convention).
    """
    root = Path(root)
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("."):
            continue
        pid_base = patient_dir.name
        seen = 0
        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir() or study_dir.name.startswith("."):
                continue
            for series_dir in sorted(study_dir.iterdir()):
                if not series_dir.is_dir() or series_dir.name.startswith("."):
                    continue
                # Must contain at least one .dcm and the first one must be a CT.
                dcms = sorted(series_dir.glob("*.dcm"))
                if not dcms:
                    continue
                if not _series_is_ct(dcms[0]):
                    continue
                xml_files = list(series_dir.glob("*.xml"))
                xml_path = xml_files[0] if xml_files else None
                pid = pid_base if seen == 0 else f"{pid_base}-{seen}"
                yield pid, series_dir, xml_path
                seen += 1


def _series_is_ct(dcm_path: Path) -> bool:
    try:
        import pydicom
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    except Exception:
        return False
    if not hasattr(ds, "ImagePositionPatient"):
        return False
    return str(getattr(ds, "Modality", "")).upper() == "CT"


# ─── DICOM tag → z-slice map ────────────────────────────────────────────────

def _uid_to_z(dicom_series_dir: Path) -> dict[str, int]:
    """Return `{SOPInstanceUID: z_index}` (0-indexed) for the CT series.

    Slice ordering follows ImagePositionPatient[2] descending (head→foot),
    matching the MATLAB code that sorts the same way.
    """
    import pydicom

    rows = []
    for p in sorted(dicom_series_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        sop = getattr(ds, "SOPInstanceUID", None)
        if ipp is None or sop is None:
            continue
        try:
            z = float(ipp[2])
        except (TypeError, ValueError, IndexError):
            continue
        rows.append((str(sop), z))
    # Sort by z descending; MATLAB code sorts `if ipp(3) < jpp(3)` swap → descending.
    rows.sort(key=lambda r: -r[1])
    return {uid: idx for idx, (uid, _z) in enumerate(rows)}


# ─── polygon rasterisation ──────────────────────────────────────────────────

def _rasterise_polygon(x: List[float], y: List[float],
                       shape: Tuple[int, int]) -> np.ndarray:
    """Filled polygon → 2D uint8 mask. Replaces MATLAB ``poly2mask``.

    Args:
        x, y: pixel coordinates (LIDC XML reports them in image coordinates,
              0-indexed continuous). Same orientation as the source slice.
        shape: (rows, cols) — usually (512, 512) for LIDC.

    Returns:
        uint8 mask, 1 inside polygon.
    """
    from skimage.draw import polygon

    rr, cc = polygon(np.array(y), np.array(x), shape=shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rr, cc] = 1
    return mask


# ─── nodule rasterisation per reader ────────────────────────────────────────

def _rasterise_nodule(nodule: LIDCNodule,
                      uid_to_z: dict[str, int],
                      shape3d: Tuple[int, int, int]) -> np.ndarray:
    """OR together the per-slice polygons of one nodule into a 3D mask."""
    n_slices, n_rows, n_cols = shape3d
    mask = np.zeros(shape3d, dtype=np.uint8)
    for roi in nodule.rois:
        z = uid_to_z.get(roi.image_sop_uid)
        if z is None or not (0 <= z < n_slices):
            continue
        slice_mask = _rasterise_polygon(roi.x_coords, roi.y_coords, (n_rows, n_cols))
        mask[z] |= slice_mask
    return mask


# ─── high-level driver ─────────────────────────────────────────────────────

def convert_patient(dicom_series_dir: str | Path,
                    xml_path: Optional[str | Path],
                    out_dir: str | Path,
                    pid: str,
                    per_nodule_masks: bool = False) -> dict:
    """Convert one LIDC patient: write CT NRRD + per-reader masks + CSV.

    Args:
        dicom_series_dir: CT series directory.
        xml_path: matching ``*.xml`` annotation (may be None — then no masks
                  written, only the CT).
        out_dir: where outputs go (created if missing).
        pid: patient identifier for the output filenames.
        per_nodule_masks: if True, also write one NRRD per (reader, nodule)
                          to enable per-nodule feature extraction.

    Returns:
        A dict with what was written + nodule counts per reader.
    """
    from qradiomics.io.dicom import load_dicom_series

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. CT volume (uses qradiomics' standard DICOM reader)
    ct = load_dicom_series(str(dicom_series_dir))
    ct_path = out_dir / f"{pid}_CT.nrrd"
    sitk.WriteImage(ct, str(ct_path))
    size = ct.GetSize()              # (cols, rows, slices) in SimpleITK
    shape3d = (size[2], size[1], size[0])  # (z, y, x) for numpy

    result = {
        "pid": pid,
        "ct_nrrd": str(ct_path),
        "ct_size": list(size),
        "ct_spacing": [float(s) for s in ct.GetSpacing()],
        "readers": [],
        "nodules_csv": None,
    }

    if xml_path is None:
        return result

    # 2. Parse XML annotations
    readers = parse_lidc_xml(xml_path)
    uid_to_z = _uid_to_z(Path(dicom_series_dir))

    # 3. Per-reader 3D mask: OR over all that reader's nodules
    csv_rows: List[dict] = []
    written_masks: List[str] = []
    for reader in readers:
        reader_mask = np.zeros(shape3d, dtype=np.uint8)
        nodules_with_polygons = 0
        for nodule in reader.nodules:
            n_mask = _rasterise_nodule(nodule, uid_to_z, shape3d)
            n_voxels = int(n_mask.sum())
            if n_voxels == 0:
                continue
            nodules_with_polygons += 1
            reader_mask |= n_mask

            # Optionally write per-(reader, nodule) mask
            if per_nodule_masks:
                nm_img = sitk.GetImageFromArray(n_mask)
                nm_img.CopyInformation(ct)
                safe_nid = nodule.nodule_id.replace(" ", "_").replace("/", "-")
                nm_path = (out_dir /
                           f"{pid}_CT_Phy{reader.session_index}_{safe_nid}-label.nrrd")
                sitk.WriteImage(nm_img, str(nm_path))

            # Compute basic stats for the CSV row
            ct_arr = sitk.GetArrayFromImage(ct)        # numpy z,y,x
            voxel_intensities = ct_arr[n_mask > 0]
            spacing = ct.GetSpacing()                   # (x, y, z) in mm
            voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
            nz = np.argwhere(n_mask > 0)
            bbox = (
                int(nz[:, 2].min()), int(nz[:, 1].min()), int(nz[:, 0].min()),
                int(nz[:, 2].ptp() + 1), int(nz[:, 1].ptp() + 1), int(nz[:, 0].ptp() + 1),
            )
            centroid_idx = (float(nz[:, 2].mean()),
                            float(nz[:, 1].mean()),
                            float(nz[:, 0].mean()))
            safe_nid = nodule.nodule_id.replace(" ", "_").replace("/", "-")
            nm_rel = (f"{pid}_CT_Phy{reader.session_index}_{safe_nid}-label.nrrd"
                      if per_nodule_masks else "")
            csv_rows.append({
                "pid": pid,
                "reader": reader.session_index,
                "nodule_id": nodule.nodule_id,
                "mask_nrrd": nm_rel,
                "n_voxels": n_voxels,
                "volume_mm3": n_voxels * voxel_volume_mm3,
                "bbox_x": bbox[0], "bbox_y": bbox[1], "bbox_z": bbox[2],
                "bbox_sx": bbox[3], "bbox_sy": bbox[4], "bbox_sz": bbox[5],
                "centroid_x_idx": centroid_idx[0],
                "centroid_y_idx": centroid_idx[1],
                "centroid_z_idx": centroid_idx[2],
                "mean_intensity": float(voxel_intensities.mean()),
                "min_intensity": float(voxel_intensities.min()),
                "max_intensity": float(voxel_intensities.max()),
                **asdict(nodule.characteristics),
            })
        if nodules_with_polygons == 0:
            continue

        # Write per-reader mask volume in the CT's geometry
        msk_img = sitk.GetImageFromArray(reader_mask)
        msk_img.CopyInformation(ct)
        msk_path = out_dir / f"{pid}_CT_Phy{reader.session_index}-label.nrrd"
        sitk.WriteImage(msk_img, str(msk_path))
        written_masks.append(str(msk_path))
        result["readers"].append({
            "session_index": reader.session_index,
            "n_nodules": nodules_with_polygons,
            "mask_nrrd": str(msk_path),
        })

    # 4. Write nodules CSV
    if csv_rows:
        csv_path = out_dir / f"{pid}_nodules.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        result["nodules_csv"] = str(csv_path)

    return result
