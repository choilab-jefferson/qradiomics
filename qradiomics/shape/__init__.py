"""qradiomics.shape — 3D shape-based feature descriptors (public release).

Public-release modules (always available):

  hessian      — Hessian eigendecomposition + surface elements    (2014 §2.2.1)
  detection    — Multi-scale dot enhancement (Sato/Li blob filter)(2014 §2.2.2)
  ahsn         — Angular Histogram of Surface Normals descriptor  (2014 §2.3.1)
  wall_elim    — Iterative wall detection and elimination          (2014 §2.3.2)
  mesh_utils   — voxel → triangular mesh + geometry primitives    (2021)
  spiculation  — Spherical parameterization, area distortion,     (2021)
                 peak detection, classification (Na/Nl/Na_att/s1/s2)
  phantoms     — Synthetic 3D lung phantoms + geometric primitives

## Private overlay (only present when `qradiomics_private` is installed)

The companion `qradiomics_private` distribution extends this namespace with
unpublished research modules (SNoH, surfacelet, decompose, discover,
voxel_spiculation, and their DL surrogates). When the overlay is installed,
its symbols become importable as if they lived in `qradiomics.shape` directly:

  >>> from qradiomics.shape import SNoHConfig, snoh   # overlay
  >>> from qradiomics.shape import Surfacelet         # overlay

Without the overlay, those imports raise ImportError. Use
`qradiomics.shape.private_overlay_loaded()` or `qr info` to check.

## Reference papers

  Choi WJ, Choi TS. *Automated pulmonary nodule detection based on three-
  dimensional shape-based feature descriptor.* Computer Methods and
  Programs in Biomedicine 2014;113(1):37-54.
  doi:10.1016/j.cmpb.2013.08.015

  Choi W, Nadeem S, Riyahi S, Deasy JO, Tannenbaum A, Lu W.
  *Reproducible and Interpretable Spiculation Quantification for Lung
  Cancer Screening.* Computer Methods and Programs in Biomedicine
  2021;200:105839. doi:10.1016/j.cmpb.2020.105839
"""

# --- Public-release surface --------------------------------------------------
from .ahsn import AHSNConfig, ahsn, find_tall_peak, normal_spherical
from .detection import Candidate, detect_candidates, dot_value, extract_block, make_scales
from .hessian import SurfaceElements, hessian_3d, hessian_eigendecomp, surface_elements
from .mesh_utils import Mesh, face_areas, vertex_areas, voxel_to_mesh
from .phantoms import (
    ALL_PHANTOMS,
    SHAPE_PRIMITIVES,
    Phantom,
    cylinder,
    make,
    make_all,
    plane,
    saddle,
    sphere,
)
from .spiculation import (
    ClassifyConfig,
    Peak,
    SpiculationFeatures,
    area_distortion,
    classify_peak,
    detect_peaks,
    peaks_to_surfacelets,
    spherical_parameterization,
    spiculation_features,
    spiculation_from_voxel,
)
from .wall_elim import wall_eliminate

# --- Private overlay (best-effort) ------------------------------------------
# Loaded only if the `qradiomics_private` distribution is installed.
# Failure to import is silent — public users see the public surface only.

_PRIVATE_LOADED: bool = False
_PRIVATE_MODULES: tuple[str, ...] = ()
_PRIVATE_VERSION: str | None = None
_PRIVATE_ALL: tuple[str, ...] = ()

try:  # pragma: no cover - depends on install state
    from qradiomics_private import __version__ as _PRIVATE_VERSION  # type: ignore[import-not-found]
    from qradiomics_private import shape as _priv_shape  # type: ignore[import-not-found]

    _PRIVATE_ALL = tuple(getattr(_priv_shape, "__all__", ()))
    _PRIVATE_MODULES = tuple(getattr(_priv_shape, "__modules__", ()))

    for _name in _PRIVATE_ALL:
        globals()[_name] = getattr(_priv_shape, _name)
    _PRIVATE_LOADED = True
    del _priv_shape
    if _PRIVATE_ALL:
        del _name
except Exception:  # ImportError or downstream import-time error
    pass


def private_overlay_loaded() -> bool:
    """Whether the qradiomics_private shape overlay is active."""
    return _PRIVATE_LOADED


def private_overlay_info() -> dict:
    """Return overlay metadata for `qr info` and introspection."""
    return {
        "loaded": _PRIVATE_LOADED,
        "version": _PRIVATE_VERSION,
        "modules": list(_PRIVATE_MODULES),
        "symbols": list(_PRIVATE_ALL),
    }


_PUBLIC_ALL: tuple[str, ...] = (
    # Public release
    "AHSNConfig", "ahsn", "find_tall_peak", "normal_spherical",
    "Candidate", "detect_candidates", "dot_value", "extract_block", "make_scales",
    "SurfaceElements", "hessian_3d", "hessian_eigendecomp", "surface_elements",
    "Mesh", "face_areas", "vertex_areas", "voxel_to_mesh",
    "ALL_PHANTOMS", "SHAPE_PRIMITIVES", "Phantom",
    "sphere", "cylinder", "plane", "saddle", "make", "make_all",
    "ClassifyConfig", "Peak", "SpiculationFeatures",
    "area_distortion", "classify_peak", "detect_peaks", "peaks_to_surfacelets",
    "spherical_parameterization", "spiculation_features", "spiculation_from_voxel",
    "wall_eliminate",
    # Overlay introspection
    "private_overlay_loaded", "private_overlay_info",
)

__all__ = list(_PUBLIC_ALL)
# Append overlay symbols at import time (runtime-extended; static analyzers see public only).
__all__ += list(_PRIVATE_ALL)  # pyright: ignore[reportUnsupportedDunderAll]
