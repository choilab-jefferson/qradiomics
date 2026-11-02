"""3D feature map pooling and small-lesion adaptive spacing primitives.

Provides cohort-agnostic foundation-model pooling utilities:
  - Area-integrated 3D soft-mask pooling (``F.adaptive_avg_pool3d`` over float binary
    mask ``M``, weighted aggregation ``(F_l * M_l).sum() / M_l.sum()``, plus true
    center-of-mass projection fallback when ``M_l.sum() < eps`` instead of bounding-box
    center).
  - Small-lesion adaptive zoom-in spacing calculation (``compute_adaptive_zoom_spacing``
    / ``compute_adaptive_focal_spacing``) for focal crops so small ROIs span enough
    encoder feature-map voxels after stride-16 downsampling.
  - Deterministic focal-crop cache key (``focal_crop_key``).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

DEFAULT_CHANNELS = 512
DEFAULT_FOCAL_PAD_MM = 20.0
DEFAULT_FOCAL_RESAMPLE_MM: Tuple[float, float, float] = (3.0, 1.0, 1.0)
DEFAULT_MIN_FOCAL_SPACING_MM: Tuple[float, float, float] = (0.5, 0.25, 0.25)
DEFAULT_TARGET_ROI_VOXELS: Tuple[int, int, int] = (32, 64, 64)

__all__ = [
    "DEFAULT_CHANNELS",
    "DEFAULT_FOCAL_PAD_MM",
    "DEFAULT_FOCAL_RESAMPLE_MM",
    "DEFAULT_MIN_FOCAL_SPACING_MM",
    "DEFAULT_TARGET_ROI_VOXELS",
    "aggregate_feature_map",
    "centroid_sample_stage",
    "compute_adaptive_focal_spacing",
    "compute_adaptive_zoom_spacing",
    "focal_crop_key",
    "project_mask_centroid_to_feature_grid",
    "resize_mask_to_feature_grid",
    "soft_pool_stage",
]


def _ensure_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for qradiomics.deep.pooling operations.")


def _normalize_mask_tensor(
    mask: Union["torch.Tensor", np.ndarray],
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """Convert mask to a 3D float tensor (D, H, W) in {0.0, 1.0}."""
    _ensure_torch()
    if isinstance(mask, np.ndarray):
        mask_t = torch.from_numpy(mask)
    else:
        mask_t = mask
    if device is not None:
        mask_t = mask_t.to(device)
    mask_f = (mask_t.float() > 0).float()
    while mask_f.dim() > 3:
        mask_f = mask_f.squeeze(0)
    if mask_f.dim() != 3:
        raise ValueError(f"Expected 3D mask (D, H, W), got shape {tuple(mask_f.shape)}")
    return mask_f


def project_mask_centroid_to_feature_grid(
    orig_mask_bool: "torch.Tensor",
    feat_spatial: Sequence[int],
) -> Tuple[int, int, int]:
    """Project the true center-of-mass of a 3D binary mask onto feature-map grid coordinates.

    Unlike bounding-box center approximations, this computes the exact mean coordinate
    of all foreground voxels and scales to ``(D_map, H_map, W_map)``.
    """
    _ensure_torch()
    if not orig_mask_bool.any():
        raise ValueError("Cannot project centroid of an empty mask.")
    D_map, H_map, W_map = (int(feat_spatial[0]), int(feat_spatial[1]), int(feat_spatial[2]))
    D_mask, H_mask, W_mask = (
        int(orig_mask_bool.shape[0]),
        int(orig_mask_bool.shape[1]),
        int(orig_mask_bool.shape[2]),
    )
    coords = torch.nonzero(orig_mask_bool, as_tuple=False).float()
    cz, cy, cx = coords.mean(dim=0).tolist()
    iz = int(np.clip(np.round(cz * (D_map - 1) / max(D_mask - 1, 1)), 0, D_map - 1))
    iy = int(np.clip(np.round(cy * (H_map - 1) / max(H_mask - 1, 1)), 0, H_map - 1))
    ix = int(np.clip(np.round(cx * (W_map - 1) / max(W_mask - 1, 1)), 0, W_map - 1))
    return iz, iy, ix


def resize_mask_to_feature_grid(
    mask: Union["torch.Tensor", np.ndarray],
    feat_spatial: Sequence[int],
    *,
    soft_mask: bool = True,
    eps: float = 1e-6,
) -> Tuple["torch.Tensor", bool]:
    """Resize a 3D binary mask onto ``feat_spatial`` with area-integrated soft pooling or hard threshold.

    When ``soft_mask=True`` and downsampling along all axes, uses ``F.adaptive_avg_pool3d``
    for exact partial-volume integration so sub-voxel lesions between grid sample centers
    do not vanish. If the resized mask sum is ``< eps``, falls back to activating the single
    projected center-of-mass voxel.

    Returns:
        ``(resized_mask, used_com_fallback)`` where ``resized_mask`` has shape ``feat_spatial``.
    """
    _ensure_torch()
    mask_f = _normalize_mask_tensor(mask)
    if mask_f.sum() == 0:
        raise ValueError("Empty ROI mask passed to resize_mask_to_feature_grid")

    orig_mask_bool = mask_f > 0
    orig_spatial = tuple(int(x) for x in mask_f.shape)
    target_spatial = (int(feat_spatial[0]), int(feat_spatial[1]), int(feat_spatial[2]))

    if orig_spatial != target_spatial:
        if soft_mask:
            is_downsampling = all(
                f_dim <= o_dim for f_dim, o_dim in zip(target_spatial, orig_spatial)
            )
            if is_downsampling:
                mask_f = (
                    F.adaptive_avg_pool3d(
                        mask_f.unsqueeze(0).unsqueeze(0),
                        output_size=target_spatial,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            else:
                mask_f = (
                    F.interpolate(
                        mask_f.unsqueeze(0).unsqueeze(0),
                        size=target_spatial,
                        mode="trilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            mask_f = mask_f.clamp(min=0.0, max=1.0)
        else:
            mask_f = (
                F.interpolate(
                    mask_f.unsqueeze(0).unsqueeze(0),
                    size=target_spatial,
                    mode="trilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
            mask_f = (mask_f > 0.5).float()

    used_fallback = False
    if float(mask_f.sum().item()) < eps:
        logger.warning(
            "Mask occupancy < %.1e after resizing — using projected center-of-mass voxel fallback.",
            eps,
        )
        mask_f = torch.zeros_like(mask_f)
        iz, iy, ix = project_mask_centroid_to_feature_grid(orig_mask_bool, target_spatial)
        mask_f[iz, iy, ix] = 1.0
        used_fallback = True

    return mask_f, used_fallback


def centroid_sample_stage(
    feature_map: "torch.Tensor",
    mask: Union["torch.Tensor", np.ndarray],
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Sample feature vectors at the projected center-of-mass voxel of ``mask``.

    Args:
        feature_map: Tensor of shape ``(1, C, D', H', W')`` or ``(C, D', H', W')``.
        mask: Binary 3D mask of shape ``(D, H, W)``.

    Returns:
        ``(ch_mean, ch_std)`` tensors of shape ``(C,)``, where ``ch_std`` is zeros.
    """
    _ensure_torch()
    fm = feature_map[0] if feature_map.dim() == 5 else feature_map
    mask_f = _normalize_mask_tensor(mask, device=fm.device)
    if mask_f.sum() == 0:
        raise ValueError("Empty ROI mask passed to centroid_sample_stage")
    iz, iy, ix = project_mask_centroid_to_feature_grid(mask_f > 0, fm.shape[-3:])
    vec = fm[:, iz, iy, ix]
    return vec, torch.zeros_like(vec)


def soft_pool_stage(
    feature_map: "torch.Tensor",
    mask: Union["torch.Tensor", np.ndarray],
    *,
    eps: float = 1e-6,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Area-integrated 3D soft-mask pooling over a single feature stage.

    Computes ``M_l = adaptive_avg_pool3d(M, output_size=F_l.shape[-3:])`` and aggregates
    weighted channel mean ``(F_l * M_l).sum() / M_l.sum()`` and weighted channel standard
    deviation. When ``M_l.sum() < eps``, falls back to :func:`centroid_sample_stage`.

    Args:
        feature_map: Tensor of shape ``(1, C, D', H', W')`` or ``(C, D', H', W')``.
        mask: Binary or float 3D mask of shape ``(D, H, W)``.
        eps: Minimum integrated mask weight before triggering COM fallback.

    Returns:
        ``(ch_mean, ch_std)`` tensors of shape ``(C,)``.
    """
    _ensure_torch()
    fm = feature_map[0] if feature_map.dim() == 5 else feature_map
    mask_f = _normalize_mask_tensor(mask, device=fm.device)
    if mask_f.sum() == 0:
        raise ValueError("Empty ROI mask passed to soft_pool_stage")

    resized_mask, used_fallback = resize_mask_to_feature_grid(
        mask_f, fm.shape[-3:], soft_mask=True, eps=eps
    )
    if used_fallback:
        return centroid_sample_stage(fm, mask_f)

    mask_bool = resized_mask > 0
    masked_values = fm[:, mask_bool]  # (C, N_vox)
    weights = resized_mask[mask_bool].unsqueeze(0)  # (1, N_vox)
    w_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    ch_mean = (masked_values * weights / w_sum).sum(dim=1)
    diff = masked_values - ch_mean.unsqueeze(1)
    ch_var = (weights * (diff**2) / w_sum).sum(dim=1).clamp(min=0.0)
    ch_std = torch.sqrt(ch_var)
    return ch_mean, ch_std


def aggregate_feature_map(
    feature_map: "torch.Tensor",
    mask: Union["torch.Tensor", np.ndarray],
    *,
    soft_mask: bool = False,
    prefix: str = "deep",
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Compute per-channel statistics (mean, std, max, min) over a masked 3D feature map.

    1. Resizes ``mask`` to ``feature_map`` spatial dimensions:
       - ``soft_mask=False``: trilinear interpolation followed by hard ``> 0.5`` threshold.
       - ``soft_mask=True``: exact partial-volume integration via ``F.adaptive_avg_pool3d``
         (when downsampling) so sub-voxel lesions are preserved with continuous weights in ``[0, 1]``.
    2. If the resized mask has occupancy ``< eps``, projects the true original mask
       center-of-mass to feature-map coordinates.
    3. Computes ``mean``, ``std``, ``max``, ``min`` per channel (weighted when ``soft_mask=True``).

    Args:
        feature_map: Tensor of shape ``(1, C, D', H', W')`` or ``(C, D', H', W')``.
        mask: Boolean or float tensor/array of shape ``(D, H, W)``.
        soft_mask: If True, use continuous partial-volume weights instead of hard 0.5 thresholding.
        prefix: Key prefix (default ``"deep"`` produces ``deep_mean_0`` ... ``deep_min_{C-1}``).
        eps: Occupancy threshold below which center-of-mass fallback is triggered.

    Returns:
        Dictionary with ``4 * C`` keys.
    """
    _ensure_torch()
    fm = feature_map if feature_map.dim() == 5 else feature_map.unsqueeze(0)
    device = fm.device
    mask_f = _normalize_mask_tensor(mask, device=device)
    if mask_f.sum() == 0:
        raise ValueError("Empty ROI mask passed to aggregate_feature_map")

    resized_mask, _ = resize_mask_to_feature_grid(
        mask_f, fm.shape[-3:], soft_mask=soft_mask, eps=eps
    )

    channels = int(fm.shape[1])
    mask_bool = resized_mask > 0
    masked_values = fm[0][:, mask_bool]  # (C, N_vox)

    if soft_mask:
        weights = resized_mask[mask_bool].unsqueeze(0)  # (1, N_vox)
        w_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        ch_mean = (masked_values * weights / w_sum).sum(dim=1)
        diff = masked_values - ch_mean.unsqueeze(1)
        ch_var = (weights * (diff**2) / w_sum).sum(dim=1).clamp(min=0.0)
        ch_std = torch.sqrt(ch_var)
    else:
        ch_mean = masked_values.mean(dim=1)
        ch_std = masked_values.std(dim=1, correction=0)

    ch_max = masked_values.max(dim=1)[0]
    ch_min = masked_values.min(dim=1)[0]

    result: Dict[str, float] = {}
    mean_np = ch_mean.detach().cpu().float().numpy()
    std_np = ch_std.detach().cpu().float().numpy()
    max_np = ch_max.detach().cpu().float().numpy()
    min_np = ch_min.detach().cpu().float().numpy()

    for i in range(channels):
        result[f"{prefix}_mean_{i}"] = float(mean_np[i])
    for i in range(channels):
        result[f"{prefix}_std_{i}"] = float(std_np[i])
    for i in range(channels):
        result[f"{prefix}_max_{i}"] = float(max_np[i])
    for i in range(channels):
        result[f"{prefix}_min_{i}"] = float(min_np[i])

    return result


def compute_adaptive_focal_spacing(
    mask_sitk: sitk.Image,
    base_resample_mm: Sequence[float] = DEFAULT_FOCAL_RESAMPLE_MM,
    min_spacing_mm: Sequence[float] = DEFAULT_MIN_FOCAL_SPACING_MM,
    target_roi_voxels: Sequence[int] = DEFAULT_TARGET_ROI_VOXELS,
    label: int = 1,
) -> Tuple[float, float, float]:
    """Compute ROI-adaptive zoom-in spacing (mm) so small objects span enough feature-map voxels.

    Hierarchical 3D encoders (e.g., SegResEncoder) downsample by 16x per axis. For a small
    ROI (e.g., < 10 mm), fixed ``(3, 1, 1)`` mm spacing leaves < 1 voxel after stride-16
    downsampling. Adaptive spacing supersamples small ROIs so the lesion bounding box
    spans up to ``target_roi_voxels`` input voxels (clamped between ``min_spacing_mm`` and
    ``base_resample_mm``), while leaving large ROIs at ``base_resample_mm``.
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_sitk)
    if label not in stats.GetLabels():
        return (
            float(base_resample_mm[0]),
            float(base_resample_mm[1]),
            float(base_resample_mm[2]),
        )

    bbox = stats.GetBoundingBox(label)  # (x0, y0, z0, sx, sy, sz)
    src_sp = mask_sitk.GetSpacing()
    out_sp: list[float] = []
    for i in range(3):
        roi_extent_mm = max(float(bbox[3 + i]) * float(src_sp[i]), float(min_spacing_mm[i]))
        ideal = roi_extent_mm / max(int(target_roi_voxels[i]), 1)
        clamped = min(float(base_resample_mm[i]), max(float(min_spacing_mm[i]), ideal))
        out_sp.append(round(clamped, 4))
    return (out_sp[0], out_sp[1], out_sp[2])


def compute_adaptive_zoom_spacing(
    mask_or_diameter: Union[sitk.Image, float, Sequence[float]],
    base_resample_mm: Sequence[float] = DEFAULT_FOCAL_RESAMPLE_MM,
    min_spacing_mm: Sequence[float] = DEFAULT_MIN_FOCAL_SPACING_MM,
    target_roi_voxels: Sequence[int] = DEFAULT_TARGET_ROI_VOXELS,
    label: int = 1,
) -> Tuple[float, float, float]:
    """Compute adaptive zoom-in voxel spacing from a ``SimpleITK.Image`` mask or physical extent (mm).

    - When passed a ``SimpleITK.Image``, delegates to :func:`compute_adaptive_focal_spacing`.
    - When passed a scalar diameter (mm) or 3-tuple of extents ``(dx_mm, dy_mm, dz_mm)``:
      computes ``clamp(extent_i / target_roi_voxels_i, min_spacing_mm_i, base_resample_mm_i)``
      (e.g., diameter < 10 mm yields fine sub-millimeter spacing down to ``min_spacing_mm``).
    """
    if isinstance(mask_or_diameter, sitk.Image):
        return compute_adaptive_focal_spacing(
            mask_or_diameter,
            base_resample_mm=base_resample_mm,
            min_spacing_mm=min_spacing_mm,
            target_roi_voxels=target_roi_voxels,
            label=label,
        )

    if isinstance(mask_or_diameter, (int, float, np.floating)):
        extents = (float(mask_or_diameter), float(mask_or_diameter), float(mask_or_diameter))
    else:
        extents = tuple(float(x) for x in mask_or_diameter)
        if len(extents) != 3:
            raise ValueError(f"Expected 3 extents (mm), got {len(extents)}")

    out_sp: list[float] = []
    for i in range(3):
        ext_mm = max(extents[i], float(min_spacing_mm[i]))
        ideal = ext_mm / max(int(target_roi_voxels[i]), 1)
        clamped = min(float(base_resample_mm[i]), max(float(min_spacing_mm[i]), ideal))
        out_sp.append(round(clamped, 4))
    return (out_sp[0], out_sp[1], out_sp[2])


def focal_crop_key(
    mask_sitk: sitk.Image,
    pad_mm: float = DEFAULT_FOCAL_PAD_MM,
    resample_mm: Optional[Sequence[float]] = DEFAULT_FOCAL_RESAMPLE_MM,
    label: int = 1,
) -> str:
    """Deterministic ROI discriminator for focal crop caching.

    Hashes the mask bounding box, ``pad_mm``, and ``resample_mm`` via SHA-1 so
    distinct ROIs on the same scan never collide in a feature-map cache.
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_sitk)
    bbox = stats.GetBoundingBox(label) if label in stats.GetLabels() else ()
    resample_repr = "none" if resample_mm is None else ",".join(f"{v:.6g}" for v in resample_mm)
    key_str = f"bbox={bbox}|pad_mm={pad_mm:.6g}|resample_mm={resample_repr}"
    return hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:16]
