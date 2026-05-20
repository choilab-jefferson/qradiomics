"""qradiomics.shape — published 3D shape-based feature descriptors.

This is the public-release subset. The private dev tree carries
additional unpublished research extensions (SNoH, voxel-native
spiculation, surfacelet, segmentation-less discovery, decompose) that
are intentionally not shipped here. The published methods themselves
are stable and citeable — see references below.

## Modules

  hessian      — Hessian eigendecomposition + surface elements    (2014 §2.2.1)
  detection    — Multi-scale dot enhancement (Sato/Li blob filter)(2014 §2.2.2)
  ahsn         — Angular Histogram of Surface Normals descriptor  (2014 §2.3.1)
  wall_elim    — Iterative wall detection and elimination          (2014 §2.3.2)
  mesh_utils   — voxel → triangular mesh + geometry primitives    (2021)
  spiculation  — Spherical parameterization, area distortion,     (2021)
                 peak detection, classification (Na/Nl/Na_att/s1/s2)
  phantoms     — Synthetic 3D lung phantoms for testing

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

from .ahsn import AHSNConfig, ahsn, find_tall_peak, normal_spherical
from .detection import Candidate, detect_candidates, dot_value, extract_block, make_scales
from .hessian import SurfaceElements, hessian_3d, hessian_eigendecomp, surface_elements
from .mesh_utils import Mesh, face_areas, vertex_areas, voxel_to_mesh
from .phantoms import ALL_PHANTOMS, Phantom, make, make_all
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

__all__ = [
    "AHSNConfig", "ahsn", "find_tall_peak", "normal_spherical",
    "Candidate", "detect_candidates", "dot_value", "extract_block", "make_scales",
    "SurfaceElements", "hessian_3d", "hessian_eigendecomp", "surface_elements",
    "Mesh", "face_areas", "vertex_areas", "voxel_to_mesh",
    "ALL_PHANTOMS", "Phantom", "make", "make_all",
    "ClassifyConfig", "Peak", "SpiculationFeatures",
    "area_distortion", "classify_peak", "detect_peaks", "peaks_to_surfacelets",
    "spherical_parameterization", "spiculation_features", "spiculation_from_voxel",
    "wall_eliminate",
]
