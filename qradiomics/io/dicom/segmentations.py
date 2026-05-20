"""Slicer-style seg.nrrd multi-segment loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import SimpleITK as sitk

__all__ = ["SegmentInfo", "list_segments_in_seg_nrrd", "load_segment_from_seg_nrrd"]


@dataclass(frozen=True)
class SegmentInfo:
    """One segment entry inside a Slicer-style seg.nrrd file."""

    index: int  # Channel index along the segments axis
    name: str
    label_value: Optional[int] = None  # explicit LabelValue if present in metadata
    color: Optional[str] = None


def list_segments_in_seg_nrrd(seg_path: Union[str, Path]) -> List[SegmentInfo]:
    """Enumerate segments in a Slicer-style seg.nrrd file.

    Slicer stores segments in metadata keys ``SegmentN_Name``,
    ``SegmentN_LabelValue``, ``SegmentN_Color``. The channel index is
    parsed from the prefix (``Segment10_Name`` → index 10).
    """
    seg = sitk.ReadImage(str(seg_path))
    names: dict[int, str] = {}
    labels: dict[int, int] = {}
    colors: dict[int, str] = {}
    for key in seg.GetMetaDataKeys():
        if not key.startswith("Segment"):
            continue
        try:
            idx_part, attr = key.split("_", 1)
            idx = int(idx_part.replace("Segment", ""))
        except ValueError:
            continue
        value = seg.GetMetaData(key)
        if attr == "Name":
            names[idx] = value
        elif attr == "LabelValue":
            try:
                labels[idx] = int(value)
            except ValueError:
                pass
        elif attr == "Color":
            colors[idx] = value
    out: List[SegmentInfo] = []
    for idx in sorted(names):
        out.append(
            SegmentInfo(
                index=idx,
                name=names[idx],
                label_value=labels.get(idx),
                color=colors.get(idx),
            )
        )
    return out


def load_segment_from_seg_nrrd(
    seg_path: Union[str, Path],
    segment_label: Union[str, int],
    *,
    output_label: int = 1,
) -> sitk.Image:
    """Extract one segment from a Slicer-style seg.nrrd as a 3-D binary mask.

    Args:
        seg_path: Path to the multi-segment seg.nrrd file.
        segment_label: Either the segment **name** (case-insensitive,
            trimmed) or the integer channel index.
        output_label: Foreground label value in the returned mask.

    Returns:
        3-D :class:`SimpleITK.Image` with foreground voxels set to
        ``output_label`` and background to 0. Spacing / origin /
        direction match the spatial axes of the seg.nrrd.
    """
    seg = sitk.ReadImage(str(seg_path))

    if isinstance(segment_label, int):
        idx = segment_label
    else:
        idx = _match_segment_name(seg, segment_label)
        if idx is None:
            raise ValueError(
                f"Segment {segment_label!r} not found in {seg_path}. "
                f"Available: {[s.name for s in list_segments_in_seg_nrrd(seg_path)]}"
            )

    arr = sitk.GetArrayFromImage(seg)
    if arr.ndim != 4:
        raise ValueError(
            f"seg.nrrd is not 4-D (channels axis missing): shape={arr.shape}"
        )
    if idx < 0 or idx >= arr.shape[-1]:
        raise ValueError(
            f"Segment index {idx} out of range for shape {arr.shape}."
        )

    binary = (arr[..., idx] > 0).astype(np.uint8) * int(output_label)
    mask = sitk.GetImageFromArray(binary)

    spacing = list(seg.GetSpacing())[:3]
    origin = list(seg.GetOrigin())[:3]
    direction = list(seg.GetDirection())
    if len(direction) == 16:
        direction = list(np.asarray(direction).reshape(4, 4)[:3, :3].flatten())
    mask.SetSpacing(spacing)
    mask.SetOrigin(origin)
    mask.SetDirection(direction)
    return mask


def _match_segment_name(seg: sitk.Image, target: str) -> Optional[int]:
    needle = target.strip().lower()
    for key in seg.GetMetaDataKeys():
        if not key.endswith("_Name"):
            continue
        value = seg.GetMetaData(key)
        if value.strip().lower() == needle:
            idx_part = key.split("_")[0]
            try:
                return int(idx_part.replace("Segment", ""))
            except ValueError:
                return None
    return None
