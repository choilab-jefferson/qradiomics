"""3D solid-core shape & compartment remodeling primitives and longitudinal trajectory tracking.

Consolidates cohort-agnostic 3D geometric shape analysis of the dense solid core
(``HU >= solid_hu_threshold``, default ``-500.0`` HU), concentric clearance vs. central
cavitation regime decomposition, and longitudinal trajectory estimators (Theil-Sen slope
and early-to-late level difference).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_erosion
from scipy.ndimage import label as cc_label

DEFAULT_SOLID_HU_THRESHOLD: float = -500.0
DEFAULT_MIN_CC_VOXELS: int = 5

PRIMARY_SHAPE_ARCH_COLS: Tuple[str, ...] = (
    "Sphericity",
    "Flatness",
    "Elongation",
    "Oblateness",
    "Ellipsoid_Packing",
    "ScaleFree_Surface",
    "Max3D_to_EqSphere_Ratio",
)

BASE_SOLID_ARCH_COLS: Tuple[str, ...] = (
    "base_sf",
    "base_log_core_band_ratio",
    "ap_core_share_t0",
)

D0_COLS: Tuple[str, ...] = ("log_V_A0", "log_nvox_core_pyr")
D1_COLS: Tuple[str, ...] = (
    "IsoNorm_LeastAxis_ap",
    "IsoNorm_MinorAxis_ap",
    "log_Flatness_ap",
    "Pancake_Index_ap",
)
D2_COLS: Tuple[str, ...] = (
    "Sphericity_ap",
    "ScaleFree_Surface_ap",
    "log_Ellipsoid_Packing_ap",
    "CentroidShift_mm_A0",
    "Fragmentation_LogCC_A0",
)
D3_LINEAR_COLS: Tuple[str, ...] = (
    "SolidFraction_A0",
    "CoreToBand_LogRatio_A0",
    "ErodedCoreShare_ap",
    "CoreMinusShell_MeanHU_ap",
)
D3_REGIME_COLS: Tuple[str, ...] = (
    "SolidFraction_A0",
    "Concentric_Clearance_Pos_A0",
    "Central_Cavitation_Neg_A0",
    "Abs_CoreToBand_Remodeling_A0",
    "CoreMinusShell_MeanHU_ap",
)

DYNAMIC_REMODELING_CLASSES: Tuple[str, ...] = (
    "regime_concentric_clearance",
    "regime_central_cavitation_or_frag",
    "regime_anisotropic_pancake",
    "major_solid_shrinkage_responder",
)

__all__ = [
    "BASE_SOLID_ARCH_COLS",
    "D0_COLS",
    "D1_COLS",
    "D2_COLS",
    "D3_LINEAR_COLS",
    "D3_REGIME_COLS",
    "DEFAULT_MIN_CC_VOXELS",
    "DEFAULT_SOLID_HU_THRESHOLD",
    "DYNAMIC_REMODELING_CLASSES",
    "PRIMARY_SHAPE_ARCH_COLS",
    "SolidRemodelingMetrics",
    "classify_remodeling_regime",
    "compute_longitudinal_remodeling_delta",
    "compute_one_solid_shape",
    "compute_solid_core_shape",
    "compute_solid_remodeling_features",
    "decompose_remodeling_regimes",
    "level",
    "slope",
    "slope_ok",
    "theil_sen",
]


@dataclass
class SolidRemodelingMetrics:
    """Exact 3D solid-core shape and core-vs-band compartment remodeling descriptors."""

    LeastAxisLength: float
    MinorAxisLength: float
    MajorAxisLength: float
    Solid_Volume_mm3: float
    Geom_Volume_mm3: float
    SurfaceArea_mm2: float
    Sphericity: float
    Flatness: float
    Elongation: float
    Oblateness: float
    Ellipsoid_Packing: float
    ScaleFree_Surface: float
    Max3D_to_EqSphere_Ratio: float
    SurfaceVolumeRatio: float
    Pancake_Index: float
    IsoNorm_LeastAxis: float
    IsoNorm_MinorAxis: float
    IsoNorm_MajorAxis: float
    CentroidShift_mm: float
    Fragmentation_LogCC: float
    SolidFraction: float
    Core_Cavitation_Index: float
    CoreToBand_LogRatio: float
    ErodedCoreShare: float
    Core_Hu_Mean: float
    Band_Hu_Mean: float
    CoreMinusShell_MeanHU: float

    def to_dict(self) -> Dict[str, float]:
        """Return metrics as a dictionary with both canonical and legacy short keys."""
        d = asdict(self)
        log_v = float(np.log(max(self.Solid_Volume_mm3, 1.0)))
        log_la = float(np.log(max(self.LeastAxisLength, 1e-3)))
        log_mi = float(np.log(max(self.MinorAxisLength, 1e-3)))
        log_ma = float(np.log(max(self.MajorAxisLength, 1e-3)))
        d.update(
            {
                # Short aliases used by JLR remodeling_3d
                "la": self.LeastAxisLength,
                "mi": self.MinorAxisLength,
                "ma": self.MajorAxisLength,
                "v": self.Solid_Volume_mm3,
                "mv": self.Geom_Volume_mm3,
                "sa": self.SurfaceArea_mm2,
                "sph": self.Sphericity,
                "fl": self.Flatness,
                "el": self.Elongation,
                "ob": self.Oblateness,
                "pack": self.Ellipsoid_Packing,
                "sc_surf": self.ScaleFree_Surface,
                "m3d_eq": self.Max3D_to_EqSphere_Ratio,
                "svr": self.SurfaceVolumeRatio,
                "core_share": self.ErodedCoreShare,
                "hu_core": self.Core_Hu_Mean,
                "hu_rim": self.Band_Hu_Mean,
                "hu_diff": self.CoreMinusShell_MeanHU,
                "centroid_shift_mm": self.CentroidShift_mm,
                "frag_log_cc": self.Fragmentation_LogCC,
                "sf_a0": self.SolidFraction,
                "c2b_log_a0": self.CoreToBand_LogRatio,
                # D0-D3 feature matrix aliases
                "log_V_A0": log_v,
                "log_LeastAxis_ap": log_la,
                "log_MinorAxis_ap": log_mi,
                "log_MajorAxis_ap": log_ma,
                "IsoNorm_LeastAxis_ap": self.IsoNorm_LeastAxis,
                "IsoNorm_MinorAxis_ap": self.IsoNorm_MinorAxis,
                "IsoNorm_MajorAxis_ap": self.IsoNorm_MajorAxis,
                "log_Flatness_ap": float(np.log(max(self.Flatness, 1e-4))),
                "log_Elongation_ap": float(np.log(max(self.Elongation, 1e-4))),
                "log_Oblateness_ap": float(np.log(max(self.Oblateness, 1e-4))),
                "Pancake_Index_ap": self.Pancake_Index,
                "Sphericity_ap": self.Sphericity,
                "log_SVR_ap": float(np.log(max(self.SurfaceVolumeRatio, 1e-5))),
                "ScaleFree_Surface_ap": self.ScaleFree_Surface,
                "log_Ellipsoid_Packing_ap": float(np.log(max(self.Ellipsoid_Packing, 1e-4))),
                "log_Max3D_to_EqSphere_ap": float(np.log(max(self.Max3D_to_EqSphere_Ratio, 1e-4))),
                "CentroidShift_mm_A0": self.CentroidShift_mm,
                "Fragmentation_LogCC_A0": self.Fragmentation_LogCC,
                "SolidFraction_A0": self.SolidFraction,
                "CoreToBand_LogRatio_A0": self.CoreToBand_LogRatio,
                "ErodedCoreShare_ap": self.ErodedCoreShare,
                "CoreMinusShell_MeanHU_ap": self.CoreMinusShell_MeanHU,
            }
        )
        return d


# ---------------------------------------------------------------------------
# Trajectory math helpers (pure numpy)
# ---------------------------------------------------------------------------


def _in_range(days: np.ndarray, lo_hi: Tuple[float, float]) -> np.ndarray:
    return (days >= lo_hi[0]) & (days <= lo_hi[1])


def theil_sen(days: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Theil-Sen median pairwise slope per feature column over distinct timepoints."""
    days_arr = np.asarray(days, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    i, j = np.triu_indices(len(days_arr), 1)
    dx = days_arr[j] - days_arr[i]
    k = dx != 0
    i, j, dx = i[k], j[k], dx[k].astype(float)
    if Y_arr.ndim == 1:
        return np.median((Y_arr[j] - Y_arr[i]) / dx)
    if Y_arr.ndim == 2:
        return np.median((Y_arr[j] - Y_arr[i]) / dx[:, None], axis=0)
    return np.median((Y_arr[:, j] - Y_arr[:, i]) / dx[None, :, None], axis=1)


def slope_ok(
    days: np.ndarray,
    span: Tuple[float, float] = (1.0, 15.0),
    min_n: int = 4,
    min_span: float = 7.0,
) -> bool:
    """Return True if ``days`` within ``span`` has at least ``min_n`` points spanning ``>= min_span``."""
    days_arr = np.asarray(days, dtype=float)
    d = days_arr[_in_range(days_arr, span)]
    return len(d) >= min_n and float(d.max() - d.min()) >= float(min_span)


def slope(
    days: np.ndarray,
    Y: np.ndarray,
    span: Tuple[float, float] = (1.0, 15.0),
    min_n: int = 4,
    min_span: float = 7.0,
) -> Optional[np.ndarray]:
    """Theil-Sen slope over ``span``; returns ``None`` unless ``slope_ok`` passes."""
    days_arr = np.asarray(days, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    k = _in_range(days_arr, span)
    if not slope_ok(days_arr, span, min_n, min_span):
        return None
    return theil_sen(days_arr[k], Y_arr[k])


def level(
    days: np.ndarray,
    Y: np.ndarray,
    early: Tuple[float, float] = (1.0, 3.0),
    late: Tuple[float, float] = (8.0, 15.0),
    min_part: int = 2,
) -> Optional[np.ndarray]:
    """Compute ``median(late) - median(early)`` per column; ``None`` if either window has ``< min_part`` scans."""
    days_arr = np.asarray(days, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    e, l_arr = _in_range(days_arr, early), _in_range(days_arr, late)
    if int(e.sum()) < min_part or int(l_arr.sum()) < min_part:
        return None
    return np.median(Y_arr[l_arr], axis=0) - np.median(Y_arr[e], axis=0)


# ---------------------------------------------------------------------------
# 3D solid-core shape & compartment remodeling extraction
# ---------------------------------------------------------------------------


def compute_solid_core_shape(
    image: Union[sitk.Image, np.ndarray],
    mask: Union[sitk.Image, np.ndarray],
    *,
    spacing_mm: Optional[Sequence[float]] = None,
    solid_hu_threshold: float = DEFAULT_SOLID_HU_THRESHOLD,
    min_cc_voxels: int = DEFAULT_MIN_CC_VOXELS,
) -> Optional[SolidRemodelingMetrics]:
    """Compute exact 3D geometric shape & remodeling descriptors of solid core (``HU >= solid_hu_threshold``).

    Args:
        image: 3D CT/CBCT image as ``SimpleITK.Image`` or ``np.ndarray`` (in ZYX order).
        mask: 3D binary ROI mask as ``SimpleITK.Image`` or ``np.ndarray`` (in ZYX order).
        spacing_mm: Voxel spacing ``(sx, sy, sz)`` in mm when passing numpy arrays.
            Defaults to ``image.GetSpacing()`` for SimpleITK images or ``(1.0, 1.0, 1.0)`` for arrays.
        solid_hu_threshold: HU threshold defining the dense solid core (default ``-500.0``).
        min_cc_voxels: Minimum connected-component voxel count for fragmentation counting (default ``5``).

    Returns:
        :class:`SolidRemodelingMetrics` or ``None`` if the mask is empty.
    """
    if isinstance(image, sitk.Image):
        sp = (
            np.array(spacing_mm, dtype=float)
            if spacing_mm is not None
            else np.array(image.GetSpacing(), dtype=float)
        )
        hu_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    else:
        sp = (
            np.array(spacing_mm, dtype=float)
            if spacing_mm is not None
            else np.array([1.0, 1.0, 1.0], dtype=float)
        )
        hu_arr = np.asarray(image, dtype=np.float32)

    if isinstance(mask, sitk.Image):
        m_arr = sitk.GetArrayFromImage(mask) > 0
    else:
        m_arr = np.asarray(mask) > 0

    if hu_arr.shape != m_arr.shape:
        raise ValueError(f"Image shape {hu_arr.shape} does not match mask shape {m_arr.shape}")
    if not m_arr.any():
        return None

    sp_zyx = sp[::-1]
    vv = float(np.prod(sp))

    coords_T = np.argwhere(m_arr)
    z0, y0, x0 = coords_T.min(axis=0)
    z1, y1, x1 = coords_T.max(axis=0) + 1
    T_sub = m_arr[z0:z1, y0:y1, x0:x1]
    hu_sub = hu_arr[z0:z1, y0:y1, x0:x1]

    solid_raw = T_sub & (hu_sub >= float(solid_hu_threshold))
    shell_raw = T_sub & (hu_sub < float(solid_hu_threshold))
    n_T = int(T_sub.sum())
    n_sol_raw = int(solid_raw.sum())
    n_sh_raw = int(shell_raw.sum())

    sf_a0 = float(n_sol_raw / max(n_T, 1))
    c2b_log_a0 = float(np.log((n_sol_raw * vv + 1.0) / (n_sh_raw * vv + 1.0)))

    solid = solid_raw if n_sol_raw >= 4 else T_sub
    n_sol = int(solid.sum())

    pts_T = np.argwhere(T_sub).astype(float) * sp_zyx[None, :]
    pts_S = np.argwhere(solid).astype(float) * sp_zyx[None, :]
    cent_T = pts_T.mean(axis=0)
    cent_S = pts_S.mean(axis=0)
    centroid_shift_mm = float(np.linalg.norm(cent_S - cent_T))

    centered = pts_S - cent_S
    cov = (centered.T @ centered) / max(len(pts_S), 1)
    evals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-4))
    la = float(4.0 * np.sqrt(evals[0]))
    mi = float(4.0 * np.sqrt(evals[1]))
    ma = float(4.0 * np.sqrt(evals[2]))

    pts_hull = (
        pts_S[np.linspace(0, len(pts_S) - 1, 2000).astype(int)] if len(pts_S) > 2000 else pts_S
    )
    dirs = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, -1, 0],
            [1, 0, 1],
            [1, 0, -1],
            [0, 1, 1],
            [0, 1, -1],
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    proj = pts_hull @ dirs.T
    ext_idx = np.unique(np.concatenate([proj.argmin(axis=0), proj.argmax(axis=0)]))
    ext_pts = pts_hull[ext_idx]
    dmat = np.linalg.norm(ext_pts[:, None, :] - ext_pts[None, :, :], axis=-1)
    max3d = float(np.max(dmat)) if len(ext_pts) > 1 else la

    v = float(max(n_sol_raw, 1) * vv)
    v_geom = float(n_sol * vv)
    p_s = np.pad(solid, 1, mode="constant", constant_values=False)
    dz_faces = np.abs(p_s[1:, 1:-1, 1:-1].astype(int) - p_s[:-1, 1:-1, 1:-1].astype(int)).sum() * (
        sp[0] * sp[1]
    )
    dy_faces = np.abs(p_s[1:-1, 1:, 1:-1].astype(int) - p_s[1:-1, :-1, 1:-1].astype(int)).sum() * (
        sp[0] * sp[2]
    )
    dx_faces = np.abs(p_s[1:-1, 1:-1, 1:].astype(int) - p_s[1:-1, 1:-1, :-1].astype(int)).sum() * (
        sp[1] * sp[2]
    )
    sa = float(max(dz_faces + dy_faces + dx_faces, 1e-3))

    sph = float(np.clip(((np.pi ** (1.0 / 3.0)) * ((6.0 * v_geom) ** (2.0 / 3.0))) / sa, 0.01, 1.0))
    fl = float(la / max(ma, 1e-6))
    el = float(mi / max(ma, 1e-6))
    ob = float(la / max(mi, 1e-6))
    pack = float(np.clip(v_geom / max((np.pi / 6.0) * ma * mi * la, 1e-6), 0.01, 2.0))
    sc_surf = float(np.log(sa / max(v_geom ** (2.0 / 3.0), 1e-6)))
    eq_d = float((6.0 * v_geom / np.pi) ** (1.0 / 3.0))
    m3d_eq = float(max3d / max(eq_d, 1e-6))
    svr = float(sa / max(v_geom, 1e-6))

    log_v = float(np.log(max(v, 1.0)))
    log_la = float(np.log(max(la, 1e-3)))
    log_mi = float(np.log(max(mi, 1e-3)))
    log_ma = float(np.log(max(ma, 1e-3)))
    iso_la = float(log_la - (1.0 / 3.0) * log_v)
    iso_mi = float(log_mi - (1.0 / 3.0) * log_v)
    iso_ma = float(log_ma - (1.0 / 3.0) * log_v)
    log_fl = float(np.log(max(fl, 1e-4)))
    pancake_index = float(iso_ma - log_fl)

    labeled, n_cc = cc_label(solid)
    if n_cc > 1:
        counts = np.bincount(labeled.ravel())[1:]
        n_cc_sig = int((counts >= int(min_cc_voxels)).sum())
    else:
        n_cc_sig = 1
    frag_log_cc = float(np.log(max(n_cc_sig, 1)))

    eroded = binary_erosion(solid)
    core_share = float(eroded.sum() / max(n_sol, 1))
    rim = solid & ~eroded
    hu_core = float(hu_sub[eroded].mean()) if eroded.any() else float(hu_sub[solid].mean())
    hu_rim = (
        float(hu_sub[rim].mean())
        if rim.any()
        else (float(hu_sub[shell_raw].mean()) if shell_raw.any() else hu_core)
    )
    hu_diff = float(hu_core - hu_rim)

    # Core_Cavitation_Index: positive when dense interior dominates (concentric),
    # negative when central density drops below rim/band (cavitation/necrosis).
    # Combines core-to-band log volume ratio and interior-vs-rim HU gradient sign.
    core_cavitation_index = float(c2b_log_a0 if eroded.any() else 0.0)

    return SolidRemodelingMetrics(
        LeastAxisLength=la,
        MinorAxisLength=mi,
        MajorAxisLength=ma,
        Solid_Volume_mm3=v,
        Geom_Volume_mm3=v_geom,
        SurfaceArea_mm2=sa,
        Sphericity=sph,
        Flatness=fl,
        Elongation=el,
        Oblateness=ob,
        Ellipsoid_Packing=pack,
        ScaleFree_Surface=sc_surf,
        Max3D_to_EqSphere_Ratio=m3d_eq,
        SurfaceVolumeRatio=svr,
        Pancake_Index=pancake_index,
        IsoNorm_LeastAxis=iso_la,
        IsoNorm_MinorAxis=iso_mi,
        IsoNorm_MajorAxis=iso_ma,
        CentroidShift_mm=centroid_shift_mm,
        Fragmentation_LogCC=frag_log_cc,
        SolidFraction=sf_a0,
        Core_Cavitation_Index=core_cavitation_index,
        CoreToBand_LogRatio=c2b_log_a0,
        ErodedCoreShare=core_share,
        Core_Hu_Mean=hu_core,
        Band_Hu_Mean=hu_rim,
        CoreMinusShell_MeanHU=hu_diff,
    )


def compute_one_solid_shape(args: Any) -> Optional[Dict[str, Any]]:
    """Worker-compatible wrapper matching JLR ``compute_one_solid_shape((image_file, contour, img, mask))``."""
    try:
        image_file, contour, img_or_path, mask_or_path = args
        if isinstance(img_or_path, (sitk.Image, np.ndarray)):
            c_img = img_or_path
        else:
            c_img = sitk.ReadImage(str(img_or_path))
        if isinstance(mask_or_path, (sitk.Image, np.ndarray)):
            m_img = mask_or_path
        else:
            m_img = sitk.ReadImage(str(mask_or_path))

        metrics = compute_solid_core_shape(c_img, m_img)
        if metrics is None:
            return None
        out: Dict[str, Any] = {"image_file": image_file, "contour": contour}
        out.update(metrics.to_dict())
        return out
    except Exception:
        return None


def compute_solid_remodeling_features(
    image: Union[sitk.Image, np.ndarray],
    mask: Union[sitk.Image, np.ndarray],
    *,
    spacing_mm: Optional[Sequence[float]] = None,
    solid_hu_threshold: float = DEFAULT_SOLID_HU_THRESHOLD,
) -> Dict[str, float]:
    """Compute flat dictionary of 3D solid-core shape and compartment remodeling features."""
    metrics = compute_solid_core_shape(
        image,
        mask,
        spacing_mm=spacing_mm,
        solid_hu_threshold=solid_hu_threshold,
    )
    if metrics is None:
        raise ValueError("Cannot compute solid remodeling features on an empty ROI mask.")
    return metrics.to_dict()


def decompose_remodeling_regimes(
    delta_core_to_band_log_ratio: Union[float, np.ndarray, pd.Series],
) -> Dict[str, Union[float, np.ndarray]]:
    """Decompose signed core-to-band log-ratio change into Concentric Clearance (+) and Central Cavitation (-).

    - ``Concentric_Clearance_Pos = max(0, delta_CoreToBand_LogRatio)``: outer low-density band clears
      faster than or ahead of solid core (concentric shrinkage).
    - ``Central_Cavitation_Neg = max(0, -delta_CoreToBand_LogRatio)``: solid core hollowing / cavitation
      into low-density interior faster than periphery.
    - ``Abs_CoreToBand_Remodeling = |delta_CoreToBand_LogRatio|``.
    """
    if isinstance(delta_core_to_band_log_ratio, (int, float, np.floating)):
        val = float(delta_core_to_band_log_ratio)
        return {
            "Concentric_Clearance_Pos_A0": max(0.0, val),
            "Central_Cavitation_Neg_A0": max(0.0, -val),
            "Abs_CoreToBand_Remodeling_A0": abs(val),
        }
    arr = np.asarray(delta_core_to_band_log_ratio, dtype=float)
    return {
        "Concentric_Clearance_Pos_A0": np.maximum(0.0, arr),
        "Central_Cavitation_Neg_A0": np.maximum(0.0, -arr),
        "Abs_CoreToBand_Remodeling_A0": np.abs(arr),
    }


def classify_remodeling_regime(
    delta_features: Mapping[str, float],
    *,
    shrinkage_log_vol_threshold: float = -0.35,
    cavitation_threshold: float = 0.15,
    concentric_threshold: float = 0.15,
    pancake_threshold: float = 0.15,
    frag_threshold: float = 0.30,
) -> Dict[str, Any]:
    """Classify longitudinal 3D remodeling trajectory into mechanistic morphological regimes.

    Args:
        delta_features: Mapping of longitudinal delta (follow-up minus baseline) or slope features.
        shrinkage_log_vol_threshold: Threshold on ``log_V_A0`` delta for major solid shrinkage responder.
        cavitation_threshold: Threshold on ``Central_Cavitation_Neg_A0`` (or ``-CoreToBand_LogRatio_A0``)
            or negative ``CoreMinusShell_MeanHU_ap`` drop.
        concentric_threshold: Threshold on ``Concentric_Clearance_Pos_A0`` (or ``+CoreToBand_LogRatio_A0``).
        pancake_threshold: Threshold on ``Pancake_Index_ap`` (or ``Pancake_Index``) increase.
        frag_threshold: Threshold on ``Fragmentation_LogCC_A0`` (or ``Fragmentation_LogCC``) increase.

    Returns:
        Dictionary with boolean regime flags (``DYNAMIC_REMODELING_CLASSES``) and ``primary_regime`` label.
    """
    d_log_v = float(delta_features.get("log_V_A0", delta_features.get("delta_log_V_A0", 0.0)))
    d_c2b = float(
        delta_features.get(
            "CoreToBand_LogRatio_A0",
            delta_features.get(
                "CoreToBand_LogRatio", delta_features.get("Core_Cavitation_Index", 0.0)
            ),
        )
    )
    d_hu_diff = float(
        delta_features.get(
            "CoreMinusShell_MeanHU_ap", delta_features.get("CoreMinusShell_MeanHU", 0.0)
        )
    )
    d_pancake = float(
        delta_features.get("Pancake_Index_ap", delta_features.get("Pancake_Index", 0.0))
    )
    d_frag = float(
        delta_features.get("Fragmentation_LogCC_A0", delta_features.get("Fragmentation_LogCC", 0.0))
    )

    decomp = decompose_remodeling_regimes(d_c2b)
    concentric_pos = float(decomp["Concentric_Clearance_Pos_A0"])
    cavitation_neg = float(decomp["Central_Cavitation_Neg_A0"])

    is_major_shrinkage = bool(d_log_v <= float(shrinkage_log_vol_threshold))
    is_concentric = bool(concentric_pos >= float(concentric_threshold) and d_hu_diff >= -10.0)
    is_cavitation_or_frag = bool(
        cavitation_neg >= float(cavitation_threshold)
        or d_frag >= float(frag_threshold)
        or d_hu_diff <= -25.0
    )
    is_pancake = bool(d_pancake >= float(pancake_threshold))

    if is_cavitation_or_frag:
        primary = "central_cavitation_or_frag"
    elif is_pancake:
        primary = "anisotropic_pancake"
    elif is_concentric:
        primary = "concentric_clearance"
    elif is_major_shrinkage:
        primary = "isotropic_shrinkage"
    else:
        primary = "stable"

    return {
        "regime_concentric_clearance": is_concentric,
        "regime_central_cavitation_or_frag": is_cavitation_or_frag,
        "regime_anisotropic_pancake": is_pancake,
        "major_solid_shrinkage_responder": is_major_shrinkage,
        "Concentric_Clearance_Pos_A0": concentric_pos,
        "Central_Cavitation_Neg_A0": cavitation_neg,
        "Abs_CoreToBand_Remodeling_A0": float(decomp["Abs_CoreToBand_Remodeling_A0"]),
        "primary_regime": primary,
    }


def compute_longitudinal_remodeling_delta(
    baseline_features: Union[SolidRemodelingMetrics, Mapping[str, float]],
    followup_features: Union[SolidRemodelingMetrics, Mapping[str, float]],
) -> Dict[str, Any]:
    """Compute longitudinal delta (``followup - baseline``) and regime decomposition between two scans."""
    b_dict = (
        baseline_features.to_dict()
        if isinstance(baseline_features, SolidRemodelingMetrics)
        else dict(baseline_features)
    )
    f_dict = (
        followup_features.to_dict()
        if isinstance(followup_features, SolidRemodelingMetrics)
        else dict(followup_features)
    )

    deltas: Dict[str, Any] = {}
    for k, f_val in f_dict.items():
        if (
            k in b_dict
            and isinstance(f_val, (int, float, np.floating))
            and isinstance(b_dict[k], (int, float, np.floating))
        ):
            deltas[k] = float(f_val) - float(b_dict[k])

    regime_info = classify_remodeling_regime(deltas)
    deltas.update(regime_info)
    return deltas
