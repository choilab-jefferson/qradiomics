"""Multi-target contour-set geometry and primary-vs-nodal tumor burden decomposition.

Provides cohort-agnostic rules and feature extractors for radiotherapy / multi-lesion
contour sets:
  - Target contour classification: :func:`is_GTV_or_tumor_roi`, :func:`is_nodal_roi`,
    :func:`is_primary_tumor_roi`, :func:`classify_contour_role`.
  - Multi-target burden & topology metrics: :func:`cset_nodal_to_primary_ratio`,
    :func:`cset_nodal_vol_fraction`, :func:`cset_vol_entropy` (Shannon volume entropy),
    and :func:`extract_contour_set_metrics` (:class:`ContourSetMetrics`).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .solid_remodeling import compute_solid_core_shape

TARGET_TOKENS: Tuple[str, ...] = ("gtv", "ctv", "itv", "ptv")
GTV_TOKENS: Tuple[str, ...] = ("gtv", "itv")
PTV_TOKENS: Tuple[str, ...] = ("ptv",)

_OAR_HINTS: Tuple[str, ...] = (
    "lung",
    "heart",
    "cord",
    "esoph",
    "vessel",
    "ves",
    "great",
    "chest",
    "body",
    "skin",
    "liver",
    "kidney",
    "stomach",
    "bowel",
    "trachea",
    "bronch",
    "aorta",
    "spinal",
    "brachial",
    "duoden",
    "rib",
    "carina",
    "plexus",
    "larynx",
    "parotid",
)

_FT = r"(?:gtv|itv|ctv|ptv)"
_RX_COMPOSITE = re.compile(r"p\s*\+\s*n|total|comb|comp(?:$|[^a-z]|osite)")
_RX_OTHER_SITE = re.compile(r"spine|bone|brain|adrenal|(?:^|[^a-z])rib")
_RX_NODAL = re.compile(
    rf"{_FT}[\s_-]?n(?:\d|$|[\s_?-]|acc|od|pet)|lngtv|(?:^|[^a-z])lns?(?:[^a-z]|$)|node|nodal|lymph|elective|gross[\s_-]?n"
)
_RX_STATION = re.compile(
    r"hil|mediast|paratr|pretrach|subcar|subaortic|aortopulm|paraesoph|sclav|level|neck|trach|"
    r"(?:^|[\s_-])(?:[1-9]|1[0-4])[lr](?:$|[\s_-])"
)
_RX_PRIMARY = re.compile(
    rf"{_FT}[\s_-]?p(?!et)(?:\d|$|[\s_-]|acc|rimary)|primary|rul|rml|rll|lul|lll|lingula|(?:lt|rt)\s?ul|lobe|mass|parenchym"
)
_RX_RETIRED = re.compile(r"do[\s_-]?not[\s_-]?use|(?:^|[^a-z])dnu(?:[^a-z]|$)|^zzz")

CSET_GEO_COLS: Tuple[str, ...] = (
    "cset_log_vol_sum",
    "cset_log_vol_gtv_sum",
    "nodal_to_primary_log_ratio",
    "ptv_to_gtv_log_ratio",
    "cset_vol_entropy",
    "cset_n_total",
    "cset_n_gtv",
    "cset_sphericity_min",
    "cset_flatness_min",
    "cset_svr_max",
)

__all__ = [
    "CSET_GEO_COLS",
    "ContourSetMetrics",
    "GTV_TOKENS",
    "PTV_TOKENS",
    "TARGET_TOKENS",
    "classify_contour_role",
    "cset_nodal_to_primary_ratio",
    "cset_nodal_vol_fraction",
    "cset_vol_entropy",
    "extract_contour_set_metrics",
    "is_GTV_or_tumor_roi",
    "is_nodal_roi",
    "is_primary_tumor_roi",
]


def is_GTV_or_tumor_roi(name: str, *, include_ptv_ctv: bool = False) -> bool:
    """Return True if ``name`` designates a gross/internal tumor target ROI (or CTV/PTV when ``include_ptv_ctv=True``).

    Excludes OAR subtractions/Booleans (e.g. ``Lung-GTV``, ``Heart-PTV``), evaluation helpers
    (``zPTV``), retired contours (``GTV_do_not_use``), and explicit margin expansions (``PTV + 2cm``).
    """
    n = str(name).lower().strip()
    if not n or n.startswith("z") or _RX_RETIRED.search(n):
        return False
    if "+" in n and ("cm" in n or "mm" in n):
        return False
    # Exclude OAR subtraction booleans like "r lung-gtv" or "esoph - ptv"
    if any(oar in n for oar in _OAR_HINTS):
        # Allow primary tumor names mentioning lung lobe only if no subtraction "-" with lung/oar
        if "lung" in n and not re.search(r"lung\s*[-–]\s*(?:gtv|itv|ctv|ptv)", n):
            other_oars = [o for o in _OAR_HINTS if o != "lung"]
            if any(o in n for o in other_oars):
                return False
        else:
            return False

    tokens = TARGET_TOKENS if include_ptv_ctv else GTV_TOKENS
    return any(t in n for t in tokens)


def classify_contour_role(name: str) -> str:
    """Classify a target contour name into ``'primary'``, ``'nodal'``, ``'composite'``, ``'other_site'``, or ``'generic'``."""
    n = str(name).lower().strip()
    if _RX_COMPOSITE.search(n) and not (_RX_NODAL.search(n) and not _RX_PRIMARY.search(n)):
        return "composite"
    if _RX_OTHER_SITE.search(n):
        return "other_site"
    if _RX_NODAL.search(n) or _RX_STATION.search(n):
        return "nodal"
    if _RX_PRIMARY.search(n):
        return "primary"
    return "generic"


def is_nodal_roi(name: str) -> bool:
    """Return True if ``name`` represents a regional lymph node or nodal station target ROI.

    Matches explicit nodal tokens (``GTVn``, ``GTV_N1``, ``GTV_nodes``, ``LN_GTV``, ``CTV_elective``)
    and thoracic/head-neck nodal station names (``hilar``, ``mediastinal``, ``paratracheal``,
    ``subcarinal``, ``paraesophageal``, ``supraclavicular``, ``4R``, ``7``, etc.).
    """
    if not is_GTV_or_tumor_roi(name, include_ptv_ctv=True):
        # Also allow bare nodal target names like "LN_4R" or "Nodal_GTV"
        n = str(name).lower().strip()
        if any(o in n for o in _OAR_HINTS) or _RX_RETIRED.search(n):
            return False
        return bool(_RX_NODAL.search(n) or _RX_STATION.search(n))
    return classify_contour_role(name) == "nodal"


def is_primary_tumor_roi(name: str, *, allow_generic_gtv: bool = True) -> bool:
    """Return True if ``name`` is a primary parenchymal tumor target (not nodal, composite, or distant site).

    Args:
        name: Contour name string.
        allow_generic_gtv: If True (default), an unannotated ``GTV`` / ``ITV`` (role ``'generic'``)
            is treated as a primary tumor candidate unless nodal tokens are present.
    """
    if not is_GTV_or_tumor_roi(name, include_ptv_ctv=True):
        return False
    r = classify_contour_role(name)
    if r == "primary":
        return True
    if allow_generic_gtv and r == "generic":
        return True
    return False


def cset_nodal_to_primary_ratio(
    nodal_volume_mm3: float,
    primary_volume_mm3: float,
    *,
    log_scale: bool = False,
    eps: float = 1.0,
) -> float:
    """Compute nodal-to-primary volume ratio (or log-ratio ``log((V_primary + V_nodal) / V_primary)``).

    Args:
        nodal_volume_mm3: Total nodal target volume in mm^3.
        primary_volume_mm3: Primary tumor target volume in mm^3.
        log_scale: If True, returns ``log((primary_volume_mm3 + nodal_volume_mm3) / max(primary_volume_mm3, eps))``
            matching ``nodal_to_primary_log_ratio`` in JLR Track 1. If False, returns linear
            ``nodal_volume_mm3 / max(primary_volume_mm3, eps)``.
        eps: Positive floor for primary denominator (default ``1.0`` mm^3).
    """
    v_nod = max(float(nodal_volume_mm3), 0.0)
    v_pri = max(float(primary_volume_mm3), float(eps))
    if log_scale:
        return float(np.log((v_pri + v_nod) / v_pri))
    return float(v_nod / v_pri)


def cset_nodal_vol_fraction(
    nodal_volume_mm3: float,
    primary_volume_mm3: float,
    *,
    eps: float = 1e-6,
) -> float:
    """Compute fraction of total gross tumor volume residing in nodal targets: ``V_nodal / (V_primary + V_nodal)``."""
    v_nod = max(float(nodal_volume_mm3), 0.0)
    v_pri = max(float(primary_volume_mm3), 0.0)
    denom = v_nod + v_pri
    if denom <= eps:
        return 0.0
    return float(v_nod / denom)


def cset_vol_entropy(volumes_mm3: Sequence[float]) -> float:
    """Compute Shannon volume entropy ``-sum(p_i * log(p_i))`` across multi-target volumes ``p_i = V_i / sum(V)``."""
    v = np.asarray(volumes_mm3, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) <= 1 or float(v.sum()) <= 0.0:
        return 0.0
    p = v / float(v.sum())
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


@dataclass
class ContourSetMetrics:
    """Multi-target contour-set burden, primary-vs-nodal split, and worst-lesion irregularity descriptors."""

    cset_n_total: int
    cset_n_gtv: int
    cset_n_primary: int
    cset_n_nodal: int
    primary_vol_mm3: float
    nodal_vol_mm3: float
    total_gtv_vol_mm3: float
    total_target_vol_mm3: float
    cset_log_vol_sum: float
    cset_log_vol_gtv_sum: float
    cset_nodal_to_primary_ratio: float
    nodal_to_primary_log_ratio: float
    cset_nodal_vol_fraction: float
    ptv_to_gtv_log_ratio: float
    cset_vol_entropy: float
    cset_sphericity_min: float
    cset_flatness_min: float
    cset_svr_max: float
    cset_irregularity_weighted: float
    cset_hu_mean_weighted: float
    cset_hu_std_across_targets: float

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def extract_contour_set_metrics(
    contours: Mapping[str, Union[np.ndarray, Mapping[str, float]]],
    *,
    ct_image: Optional[np.ndarray] = None,
    spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
    primary_contour_name: Optional[str] = None,
) -> ContourSetMetrics:
    """Extract multi-target contour-set geometry, nodal-vs-primary burden, and worst-lesion shape irregularity.

    Supports two input modes:
      1. **3D masks dict**: ``contours = {contour_name: binary_3d_mask, ...}`` (with optional ``ct_image``
         3D HU array and ``spacing_mm``).
      2. **Pre-extracted per-contour feature dict**: ``contours = {contour_name: {"volume_mm3": ..., "sphericity": ..., ...}}``.

    Args:
        contours: Mapping from contour name to either a 3D binary mask array or a dict of precomputed
            per-contour shape/intensity stats.
        ct_image: Optional 3D HU array aligned with the 3D masks.
        spacing_mm: Physical voxel spacing ``(sx, sy, sz)`` in mm (default ``(1.0, 1.0, 1.0)``).
        primary_contour_name: Optional explicit primary GTV contour name; if omitted, the largest
            primary-classified GTV/ITV contour is selected as the index primary lesion.

    Returns:
        :class:`ContourSetMetrics` summarizing the contour set.
    """
    vv = float(np.prod(np.asarray(spacing_mm, dtype=float)))
    hu_arr = np.asarray(ct_image, dtype=np.float32) if ct_image is not None else None

    records: list[Dict[str, Any]] = []
    for cname, item in contours.items():
        if not is_GTV_or_tumor_roi(cname, include_ptv_ctv=True) and not is_nodal_roi(cname):
            continue

        n_lower = str(cname).lower()
        is_gtv = any(t in n_lower for t in GTV_TOKENS) or is_nodal_roi(cname)
        is_ptv = any(t in n_lower for t in PTV_TOKENS) and not is_gtv
        nodal_flag = is_nodal_roi(cname)

        if isinstance(item, Mapping):
            vol = float(
                item.get(
                    "volume_mm3",
                    item.get("original_shape_VoxelVolume", item.get("Solid_Volume_mm3", 0.0)),
                )
            )
            if vol <= 0.0:
                continue
            sph = float(
                item.get(
                    "sphericity", item.get("original_shape_Sphericity", item.get("Sphericity", 1.0))
                )
            )
            fl = float(
                item.get("flatness", item.get("original_shape_Flatness", item.get("Flatness", 1.0)))
            )
            svr = float(
                item.get(
                    "svr",
                    item.get(
                        "original_shape_SurfaceVolumeRatio", item.get("SurfaceVolumeRatio", 0.1)
                    ),
                )
            )
            hu_mean = float(item.get("hu_mean", item.get("Core_Hu_Mean", 0.0)))
        else:
            m_arr = np.asarray(item) > 0
            if not m_arr.any():
                continue
            vol = float(m_arr.sum() * vv)
            hu_for_shape = hu_arr if hu_arr is not None else np.zeros_like(m_arr, dtype=np.float32)
            sh = compute_solid_core_shape(
                hu_for_shape,
                m_arr,
                spacing_mm=spacing_mm,
                solid_hu_threshold=-10000.0 if hu_arr is None else -500.0,
            )
            if sh is not None:
                sph = sh.Sphericity
                fl = sh.Flatness
                svr = sh.SurfaceVolumeRatio
            else:
                sph, fl, svr = 1.0, 1.0, 0.1
            hu_mean = float(hu_arr[m_arr].mean()) if hu_arr is not None else 0.0

        records.append(
            {
                "name": cname,
                "is_gtv": is_gtv,
                "is_ptv": is_ptv,
                "is_nodal": nodal_flag,
                "is_primary": not nodal_flag and is_gtv,
                "vol": vol,
                "sph": sph,
                "fl": fl,
                "svr": svr,
                "hu_mean": hu_mean,
            }
        )

    if not records:
        raise ValueError("No valid target ROIs found in contour set.")

    all_vols = np.array([r["vol"] for r in records], dtype=float)
    gtv_recs = [r for r in records if r["is_gtv"]]
    ptv_recs = [r for r in records if r["is_ptv"]]
    if not gtv_recs:
        gtv_recs = records

    nodal_recs = [r for r in gtv_recs if r["is_nodal"]]
    pri_recs = [r for r in gtv_recs if not r["is_nodal"]]

    if primary_contour_name is not None:
        matched_pri = [
            r for r in records if str(r["name"]).lower() == str(primary_contour_name).lower()
        ]
        if matched_pri:
            pri_vol = float(matched_pri[0]["vol"])
        elif pri_recs:
            pri_vol = float(sum(r["vol"] for r in pri_recs))
        else:
            pri_vol = float(max(r["vol"] for r in gtv_recs))
    elif pri_recs:
        pri_vol = float(sum(r["vol"] for r in pri_recs))
    else:
        pri_vol = float(max(r["vol"] for r in gtv_recs))

    nodal_vol = float(sum(r["vol"] for r in nodal_recs))
    gtv_vols = np.array([r["vol"] for r in gtv_recs], dtype=float)
    total_gtv_vol = float(gtv_vols.sum())
    total_target_vol = float(all_vols.sum())
    max_ptv_vol = float(max((r["vol"] for r in ptv_recs), default=total_gtv_vol))

    cset_log_vol_sum = float(np.log(max(total_target_vol, 1.0)))
    cset_log_vol_gtv_sum = float(np.log(max(total_gtv_vol, 1.0)))
    nodal_ratio = cset_nodal_to_primary_ratio(nodal_vol, pri_vol, log_scale=False)
    nodal_log_ratio = cset_nodal_to_primary_ratio(nodal_vol, pri_vol, log_scale=True)
    nodal_frac = cset_nodal_vol_fraction(nodal_vol, pri_vol)
    ptv_to_gtv_log_ratio = float(np.log(max(max_ptv_vol, 1.0) / max(pri_vol, 1.0)))
    vol_ent = cset_vol_entropy(all_vols)

    sph_vals = np.array([r["sph"] for r in gtv_recs], dtype=float)
    fl_vals = np.array([r["fl"] for r in gtv_recs], dtype=float)
    svr_vals = np.array([r["svr"] for r in gtv_recs], dtype=float)
    hu_vals = np.array([r["hu_mean"] for r in gtv_recs], dtype=float)
    w_gtv = gtv_vols / max(total_gtv_vol, 1e-6)

    cset_sphericity_min = float(np.min(sph_vals))
    cset_flatness_min = float(np.min(fl_vals))
    cset_svr_max = float(np.max(svr_vals))
    cset_irregularity_weighted = float(np.sum(w_gtv * (1.0 - sph_vals)))
    cset_hu_mean_weighted = float(np.sum(w_gtv * hu_vals))
    cset_hu_std_across_targets = float(np.std(hu_vals)) if len(hu_vals) > 1 else 0.0

    return ContourSetMetrics(
        cset_n_total=len(records),
        cset_n_gtv=len(gtv_recs),
        cset_n_primary=len(pri_recs),
        cset_n_nodal=len(nodal_recs),
        primary_vol_mm3=pri_vol,
        nodal_vol_mm3=nodal_vol,
        total_gtv_vol_mm3=total_gtv_vol,
        total_target_vol_mm3=total_target_vol,
        cset_log_vol_sum=cset_log_vol_sum,
        cset_log_vol_gtv_sum=cset_log_vol_gtv_sum,
        cset_nodal_to_primary_ratio=nodal_ratio,
        nodal_to_primary_log_ratio=nodal_log_ratio,
        cset_nodal_vol_fraction=nodal_frac,
        ptv_to_gtv_log_ratio=ptv_to_gtv_log_ratio,
        cset_vol_entropy=vol_ent,
        cset_sphericity_min=cset_sphericity_min,
        cset_flatness_min=cset_flatness_min,
        cset_svr_max=cset_svr_max,
        cset_irregularity_weighted=cset_irregularity_weighted,
        cset_hu_mean_weighted=cset_hu_mean_weighted,
        cset_hu_std_across_targets=cset_hu_std_across_targets,
    )
