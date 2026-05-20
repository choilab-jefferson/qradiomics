"""STAPLE consensus mask for multi-reader LIDC annotations.

Port of `STAPLEComparison` (radiomics-tools docker image) → SimpleITK's
native `STAPLEImageFilter` (binary) / `MultiLabelSTAPLEImageFilter`.

The MATLAB pipeline ran 4 per-reader masks through STAPLE to produce a
consensus binary mask plus per-reader sensitivity/specificity. We mirror
the same outputs without the docker dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import SimpleITK as sitk


def staple_consensus(reader_masks: Sequence[sitk.Image],
                     foreground_value: int = 1,
                     threshold: float = 0.5) -> tuple[sitk.Image, List[float], List[float]]:
    """Combine N per-reader binary masks into a STAPLE consensus mask.

    Args:
        reader_masks: list of SimpleITK images (binary; same geometry).
        foreground_value: label considered "nodule" in the input masks.
        threshold: STAPLE probability cutoff for the binary consensus output.

    Returns:
        (consensus_mask, sensitivities, specificities)
            consensus_mask  — uint8 binary sitk.Image (foreground = 1)
            sensitivities   — per-reader sensitivity (E-step estimate)
            specificities   — per-reader specificity (E-step estimate)
    """
    if not reader_masks:
        raise ValueError("staple_consensus requires at least one reader mask")

    cast = [sitk.Cast(m, sitk.sitkUInt8) for m in reader_masks]
    staple = sitk.STAPLEImageFilter()
    staple.SetForegroundValue(foreground_value)
    prob = staple.Execute(cast)            # float image, voxel = P(foreground)
    sens = list(staple.GetSensitivity())
    spec = list(staple.GetSpecificity())

    binary = sitk.Cast(sitk.GreaterEqual(prob, threshold), sitk.sitkUInt8)
    binary.CopyInformation(reader_masks[0])
    return binary, sens, spec


def staple_patient(patient_dir: str | Path,
                   pid: str,
                   out_path: str | Path | None = None,
                   threshold: float = 0.5) -> dict:
    """Run STAPLE across all `{pid}_CT_Phy*-label.nrrd` files in a patient dir.

    Args:
        patient_dir: where `convert_patient` wrote the per-reader masks.
        pid: patient identifier (filename prefix).
        out_path: optional consensus output path (default `{pid}_CT_STAPLE-label.nrrd`).
        threshold: probability threshold for binary consensus.

    Returns:
        Dict with consensus_path, n_readers, sensitivities, specificities.
        Returns `{"n_readers": 0}` if no per-reader masks were found.
    """
    patient_dir = Path(patient_dir)
    # Find per-reader (not per-nodule) masks: pattern `..._Phy<n>-label.nrrd`.
    reader_files = sorted(
        p for p in patient_dir.glob(f"{pid}_CT_Phy*-label.nrrd")
        if "_Phy" in p.name and p.name.split("_Phy")[-1].split("-")[0].isdigit()
    )
    if not reader_files:
        return {"n_readers": 0, "consensus_path": None,
                "sensitivities": [], "specificities": []}

    masks = [sitk.ReadImage(str(p)) for p in reader_files]
    consensus, sens, spec = staple_consensus(masks, threshold=threshold)

    if out_path is None:
        out_path = patient_dir / f"{pid}_CT_STAPLE-label.nrrd"
    out_path = Path(out_path)
    sitk.WriteImage(consensus, str(out_path))

    return {
        "n_readers": len(reader_files),
        "reader_files": [str(p) for p in reader_files],
        "consensus_path": str(out_path),
        "sensitivities": sens,
        "specificities": spec,
        "consensus_voxels": int(np.sum(sitk.GetArrayFromImage(consensus) > 0)),
    }
