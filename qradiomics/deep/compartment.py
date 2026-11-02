"""Multi-stage pyramid compartment-decomposed feature pooling.

Decomposes a 3D lesion mask into three physical tissue compartments:
  - ``whole``: all foreground ROI voxels (``mask > 0``).
  - ``solid_core`` (alias ``core``): dense tumour core (``mask > 0`` and ``HU >= solid_hu_threshold``).
  - ``boundary_shell`` (alias ``shell``): low-density lung-ingress boundary / sub-solid rim
    (``mask > 0`` and ``HU < solid_hu_threshold``, or morphological peripheral shell fallback
    when all lesion voxels exceed ``solid_hu_threshold`` and ``shell_fallback=True``).

Supports pooling across multi-stage encoder feature pyramids (default ``stages=(-3, -2, -1)``)
either from a pre-concatenated pyramid feature tensor or a sequence/dict of per-stage tensors.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import binary_erosion

from .pooling import (
    _ensure_torch,
    aggregate_feature_map,
    centroid_sample_stage,
    soft_pool_stage,
)

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

COMPARTMENTS: Tuple[str, str, str] = ("whole", "solid_core", "boundary_shell")
LEGACY_COMPARTMENTS: Tuple[str, str, str] = ("whole", "core", "shell")
DEFAULT_STAGES: Tuple[int, ...] = (-3, -2, -1)
DEFAULT_SOLID_HU_THRESHOLD: float = -500.0
DEFAULT_PYRAMID_CHANNELS: int = 896
STATS: Tuple[str, str] = ("mean", "std")

__all__ = [
    "COMPARTMENTS",
    "DEFAULT_PYRAMID_CHANNELS",
    "DEFAULT_SOLID_HU_THRESHOLD",
    "DEFAULT_STAGES",
    "LEGACY_COMPARTMENTS",
    "STATS",
    "align_pyramid_stages",
    "build_compartment_masks",
    "centroid_sample_stage",
    "extract_multistage_compartment_features",
    "pool_compartment_stage",
    "pool_pyramid_compartments",
    "soft_pool_stage",
]


def _to_numpy_3d(arr: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    if torch is not None and isinstance(arr, torch.Tensor):
        out = arr.detach().cpu().numpy()
    else:
        out = np.asarray(arr)
    while out.ndim > 3:
        out = np.squeeze(out, axis=0)
    if out.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {out.shape}")
    return out


def build_compartment_masks(
    ct_crop: Union[np.ndarray, "torch.Tensor"],
    mask_crop: Union[np.ndarray, "torch.Tensor"],
    *,
    solid_hu_threshold: float = DEFAULT_SOLID_HU_THRESHOLD,
    shell_fallback: bool = False,
) -> Dict[str, np.ndarray]:
    """Construct boolean 3D masks for ``whole``, ``solid_core`` (``core``), and ``boundary_shell`` (``shell``).

    Args:
        ct_crop: 3D CT/CBCT intensity array in Hounsfield Units (HU).
        mask_crop: 3D binary mask array of the lesion ROI.
        solid_hu_threshold: HU threshold separating solid core (``>= threshold``) from
            boundary shell (``< threshold``). Default ``-500.0`` HU.
        shell_fallback: If True and ``boundary_shell`` has 0 voxels (entire ROI >= threshold),
            use the 1-voxel morphological surface rim (``whole & ~binary_erosion(whole)``) as
            the boundary shell fallback.

    Returns:
        Dictionary with keys ``"whole"``, ``"solid_core"``, ``"boundary_shell"`` (plus aliases
        ``"core"`` and ``"shell"``).
    """
    hu = _to_numpy_3d(ct_crop).astype(np.float32)
    whole = _to_numpy_3d(mask_crop) > 0
    if hu.shape != whole.shape:
        raise ValueError(f"CT crop shape {hu.shape} does not match mask shape {whole.shape}")

    solid_core = whole & (hu >= float(solid_hu_threshold))
    boundary_shell = whole & (hu < float(solid_hu_threshold))

    if shell_fallback and whole.any() and not boundary_shell.any():
        eroded = binary_erosion(whole)
        rim = whole & ~eroded
        boundary_shell = rim if rim.any() else whole.copy()

    return {
        "whole": whole,
        "solid_core": solid_core,
        "boundary_shell": boundary_shell,
        "core": solid_core,
        "shell": boundary_shell,
    }


def align_pyramid_stages(
    stage_tensors: Union[
        "torch.Tensor",
        Sequence["torch.Tensor"],
        Mapping[int, "torch.Tensor"],
    ],
    stages: Sequence[int] = DEFAULT_STAGES,
) -> "torch.Tensor":
    """Select and trilinearly upsample multi-stage encoder outputs to the finest stage resolution.

    Args:
        stage_tensors: Either a single already-concatenated 5D tensor ``(1, C, D, H, W)``,
            a sequence of stage tensors (indexed by ``stages``, e.g. ``(-3, -2, -1)``),
            or a mapping from stage index to tensor.
        stages: Stage indices to extract and concatenate along the channel dimension.

    Returns:
        5D tensor ``(1, C_total, D_finest, H_finest, W_finest)``.
    """
    _ensure_torch()
    if isinstance(stage_tensors, torch.Tensor):
        return stage_tensors if stage_tensors.dim() == 5 else stage_tensors.unsqueeze(0)

    if isinstance(stage_tensors, Mapping):
        selected = [stage_tensors[int(s)] for s in stages]
    else:
        seq = list(stage_tensors)
        if (
            len(seq) == len(stages)
            and all(s < 0 for s in stages)
            and len(seq) <= max(abs(s) for s in stages) - 1
        ):
            selected = seq
        else:
            selected = [seq[int(s)] for s in stages]

    selected_5d = [t if t.dim() == 5 else t.unsqueeze(0) for t in selected]
    finest_idx = int(
        max(
            range(len(selected_5d)),
            key=lambda i: (
                selected_5d[i].shape[-1] * selected_5d[i].shape[-2] * selected_5d[i].shape[-3]
            ),
        )
    )
    target_spatial = selected_5d[finest_idx].shape[-3:]
    aligned = [
        (
            t
            if tuple(t.shape[-3:]) == tuple(target_spatial)
            else F.interpolate(t, size=target_spatial, mode="trilinear", align_corners=False)
        )
        for t in selected_5d
    ]
    return torch.cat(aligned, dim=1)


def pool_compartment_stage(
    stage_tensor: "torch.Tensor",
    compartment_mask: Union[np.ndarray, "torch.Tensor"],
    *,
    fallback_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pool a single feature stage over one compartment mask using soft partial-volume weights.

    Args:
        stage_tensor: Feature tensor ``(1, C, D', H', W')`` or ``(C, D', H', W')``.
        compartment_mask: 3D binary mask ``(D, H, W)`` for the compartment.
        fallback_mask: Optional fallback mask (e.g. ``whole``) when ``compartment_mask`` is empty.
            If ``None`` and ``compartment_mask`` is empty, returns ``NaN`` arrays of length ``C``.
        eps: Minimum occupancy threshold before COM fallback.

    Returns:
        ``(mean_vec, std_vec, n_voxels)`` where ``mean_vec`` and ``std_vec`` have shape ``(C,)``.
    """
    _ensure_torch()
    fm = stage_tensor if stage_tensor.dim() == 5 else stage_tensor.unsqueeze(0)
    nch = int(fm.shape[1])
    m_arr = _to_numpy_3d(compartment_mask) > 0
    nvox = int(m_arr.sum())

    if nvox == 0:
        if fallback_mask is not None:
            fb_arr = _to_numpy_3d(fallback_mask) > 0
            if fb_arr.any():
                ch_mean, ch_std = soft_pool_stage(fm, fb_arr.astype(np.float32), eps=eps)
                return (
                    ch_mean.detach().cpu().float().numpy(),
                    ch_std.detach().cpu().float().numpy(),
                    0,
                )
        nan_vec = np.full(nch, np.nan, dtype=np.float32)
        return nan_vec, nan_vec.copy(), 0

    ch_mean, ch_std = soft_pool_stage(fm, m_arr.astype(np.float32), eps=eps)
    return (
        ch_mean.detach().cpu().float().numpy(),
        ch_std.detach().cpu().float().numpy(),
        nvox,
    )


def pool_pyramid_compartments(
    fm: "torch.Tensor",
    hu: Union[np.ndarray, "torch.Tensor"],
    t_in: Union[np.ndarray, "torch.Tensor"],
    hu_cut: float = DEFAULT_SOLID_HU_THRESHOLD,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Pool pyramid feature map over focal compartments (``whole``, ``core``, ``shell``).

    Matches the JLR ``pool_pyramid_compartments`` contract:
      - Returns ``(feature_vector, nvox_dict)`` where ``feature_vector`` concatenates
        ``[mean_0..mean_{C-1}, std_0..std_{C-1}]`` across ``("whole", "core", "shell")``.
      - Empty compartments get ``NaN`` entries and ``nvox[comp] = 0``.
    """
    _ensure_torch()
    fm_5d = fm if fm.dim() == 5 else fm.unsqueeze(0)
    nch = int(fm_5d.shape[1])
    masks = build_compartment_masks(hu, t_in, solid_hu_threshold=hu_cut, shell_fallback=False)

    vec: list[np.ndarray] = []
    nvox: Dict[str, int] = {}
    for c in LEGACY_COMPARTMENTS:
        m = masks[c]
        nvox[c] = int(m.sum())
        if nvox[c] == 0:
            vec.append(np.full(len(STATS) * nch, np.nan, dtype=np.float32))
            continue
        m_t = torch.from_numpy(m.astype(np.float32)).to(fm_5d.device)
        st = aggregate_feature_map(fm_5d, m_t, soft_mask=True)
        vec.append(
            np.array([st[f"deep_{s}_{i}"] for s in STATS for i in range(nch)], dtype=np.float32)
        )
    nvox["solid_core"] = nvox["core"]
    nvox["boundary_shell"] = nvox["shell"]
    return np.concatenate(vec), nvox


def extract_multistage_compartment_features(
    stage_tensors: Union[
        "torch.Tensor",
        Sequence["torch.Tensor"],
        Mapping[int, "torch.Tensor"],
    ],
    ct_crop: Union[np.ndarray, "torch.Tensor"],
    mask_crop: Union[np.ndarray, "torch.Tensor"],
    *,
    stages: Sequence[int] = DEFAULT_STAGES,
    solid_hu_threshold: float = DEFAULT_SOLID_HU_THRESHOLD,
    shell_fallback: bool = True,
    fallback_empty_to_whole: bool = False,
) -> Dict[str, Union[float, int, np.ndarray]]:
    """Extract multi-stage pyramid compartment-decomposed features for ``whole``, ``solid_core``, and ``boundary_shell``.

    Args:
        stage_tensors: Multi-stage feature pyramid (5D concatenated tensor, sequence of stage
            tensors, or dict mapping stage index to tensor).
        ct_crop: 3D HU array/tensor aligned with ``mask_crop``.
        mask_crop: 3D binary ROI mask array/tensor.
        stages: Pyramid stages to align and concatenate (default ``(-3, -2, -1)``).
        solid_hu_threshold: HU threshold separating ``solid_core`` (``>= threshold``) from
            ``boundary_shell`` (``< threshold``). Default ``-500.0``.
        shell_fallback: When True and all ROI voxels have ``HU >= solid_hu_threshold``, use
            the 1-voxel morphological rim as ``boundary_shell``.
        fallback_empty_to_whole: When True, if a sub-compartment is still empty, substitute
            the ``whole`` compartment feature vector instead of ``NaN``.

    Returns:
        Dictionary containing:
          - ``"feature_vector"``: 1D float32 numpy array of length ``3 * 2 * C_total``.
          - ``"whole_mean"``, ``"whole_std"``, ``"solid_core_mean"``, ``"solid_core_std"``,
            ``"boundary_shell_mean"``, ``"boundary_shell_std"``: 1D arrays of length ``C_total``.
          - ``"nvox_whole"``, ``"nvox_solid_core"`` (and ``"nvox_core"``),
            ``"nvox_boundary_shell"`` (and ``"nvox_shell"``): voxel counts.
          - Per-channel keys ``pyr_{whole,core,shell}_{mean,std}_{i}`` and
            ``pyr_{whole,solid_core,boundary_shell}_{mean,std}_{i}``.
    """
    _ensure_torch()
    fm = align_pyramid_stages(stage_tensors, stages=stages)
    nch = int(fm.shape[1])
    masks = build_compartment_masks(
        ct_crop,
        mask_crop,
        solid_hu_threshold=solid_hu_threshold,
        shell_fallback=shell_fallback,
    )
    whole_mask = masks["whole"]
    if not whole_mask.any():
        raise ValueError("Empty ROI mask passed to extract_multistage_compartment_features")

    fb_mask = whole_mask if fallback_empty_to_whole else None
    out: Dict[str, Union[float, int, np.ndarray]] = {}
    vec_parts: list[np.ndarray] = []

    comp_pairs = (
        ("whole", "whole"),
        ("solid_core", "core"),
        ("boundary_shell", "shell"),
    )
    for canon_name, short_name in comp_pairs:
        c_mean, c_std, nvox = pool_compartment_stage(
            fm, masks[canon_name], fallback_mask=fb_mask if canon_name != "whole" else None
        )
        out[f"{canon_name}_mean"] = c_mean
        out[f"{canon_name}_std"] = c_std
        out[f"{short_name}_mean"] = c_mean
        out[f"{short_name}_std"] = c_std
        out[f"nvox_{canon_name}"] = int(masks[canon_name].sum())
        out[f"nvox_{short_name}"] = int(masks[canon_name].sum())
        vec_parts.append(c_mean)
        vec_parts.append(c_std)

        for i in range(nch):
            out[f"pyr_{short_name}_mean_{i}"] = float(c_mean[i])
            out[f"pyr_{short_name}_std_{i}"] = float(c_std[i])
            out[f"pyr_{canon_name}_mean_{i}"] = float(c_mean[i])
            out[f"pyr_{canon_name}_std_{i}"] = float(c_std[i])

    out["feature_vector"] = np.concatenate(vec_parts).astype(np.float32)
    return out
