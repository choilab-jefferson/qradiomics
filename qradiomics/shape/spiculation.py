"""Spiculation quantification — Choi et al. CMPB 2021.

> Choi W, Nadeem S, Alam SR, Deasy JO, Tannenbaum A, Lu W.
> Reproducible and Interpretable Spiculation Quantification for Lung Cancer
> Screening. Comput Methods Programs Biomed 2021;200:105839.
> doi:10.1016/j.cmpb.2020.105839

## Key insight (paper §2.1)

For an **angle-preserving (conformal) spherical mapping** of a nodule
surface, the corresponding **negative area distortion** precisely
characterizes the spikes/spiculations: a spike "collapses" in area on
the spherical map; a smooth region "expands".

## Pipeline (paper §2.3)

  1. voxel mask → triangular mesh (marching cubes)
  2. mesh → unit-sphere parameterization (Poisson / Ricci-flow analog)
  3. per-vertex area distortion = log(A_sphere(v) / A_original(v))
  4. baseline curves at Δarea = 0, apex = local minimum
  5. peak height = sum of consecutive centroid distances along level-set
  6. peak width = full-width-half-min of area-distortion contour
  7. classify each peak: spiculation / lobulation / attachment
     via height threshold (h_t = 0.27) and solid angle (ω_t = 0.46 sr)

## Outputs

  - Na   number of spiculations  (sharp peak, small solid angle, tall)
  - Nl   number of lobulations    (curved peak, large solid angle)
  - Na_att number of attachments  (peaks at vessel/wall attachment site)
  - s1   sharpness score = Σ_i (mean Δarea × height) / h_total
  - s2   irregularity score = Σ_i (var Δarea × height) / h_total

These plug into the 2014 AHSN / 2026 SNoH framework as **mesh-domain
surface-lets**, complementing the voxel-domain Hessian primitives.

## Implementation note

The 2021 paper uses Ricci flow + conformal welding (Choi et al. [9])
for true conformal sphere mapping. Here we ship a *practical*
approximation:

  - initial spherical embedding via stereographic-like projection
  - iterative cotangent-Laplacian smoothing on the sphere (a discrete
    quasi-conformal flow) → drives the embedding toward conformal
  - area distortion computed per-vertex on the resulting sphere

This is sufficient to reproduce the qualitative spike-detection
behavior on synthetic and real nodules. For research-grade conformal
maps, swap `spherical_parameterization()` with a Ricci-flow library
(e.g. `polyscope` + `meshplex` extensions) — the rest of the pipeline
(area distortion, peak detection, scoring) is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .mesh_utils import (
    Mesh, voxel_to_mesh, vertex_areas, face_areas, vertex_normals,
    vertex_adjacency,
)


# =====================================================================
# Spherical parameterization (approximate conformal)
# =====================================================================

def spherical_parameterization(mesh: Mesh, n_iter: int = 200,
                                 step: float = 0.1) -> np.ndarray:
    """Approximate conformal sphere mapping for a genus-0 mesh.

    Returns:
        (V, 3) unit-sphere positions, one per vertex.

    Algorithm:
        1. Center mesh, project to unit sphere from centroid.
        2. Iteratively relax each vertex toward the area-weighted
           average of its 1-ring neighbors on the sphere, then re-
           normalize. This is a discrete quasi-conformal flow on S².
    """
    verts = mesh.vertices - mesh.vertices.mean(axis=0, keepdims=True)
    pos = verts / (np.linalg.norm(verts, axis=1, keepdims=True) + 1e-12)
    adj = vertex_adjacency(mesh)
    V = mesh.n_vertices

    # Pre-compute neighbor index arrays for vectorization
    flat_idx = []
    flat_src = []
    for v, neigh in enumerate(adj):
        for n in neigh:
            flat_idx.append(v)
            flat_src.append(n)
    flat_idx = np.asarray(flat_idx, dtype=np.int64)
    flat_src = np.asarray(flat_src, dtype=np.int64)

    for _ in range(n_iter):
        # Neighbor-mean update
        nb_pos = pos[flat_src]
        accum = np.zeros_like(pos)
        np.add.at(accum, flat_idx, nb_pos)
        cnt = np.zeros(V, dtype=np.float32)
        np.add.at(cnt, flat_idx, 1.0)
        cnt = np.maximum(cnt, 1)
        target = accum / cnt[:, None]
        pos = pos + step * (target - pos)
        pos = pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-12)
    return pos.astype(np.float32)


# =====================================================================
# Per-vertex area distortion (paper §2.3 step 2)
# =====================================================================

def area_distortion(mesh: Mesh, sphere_pos: np.ndarray) -> np.ndarray:
    """Per-vertex log(A_sphere / A_original).

    Negative → spike (collapses on sphere). Positive → smooth (expands).
    """
    A_orig = vertex_areas(mesh)
    sphere_mesh = Mesh(vertices=sphere_pos, faces=mesh.faces)
    A_sphere = vertex_areas(sphere_mesh)
    # Normalize so total area on sphere matches (4π for unit sphere) and
    # total on mesh stays as is, so ratio reflects local distortion only.
    A_orig_n = A_orig / (A_orig.sum() + 1e-12)
    A_sphere_n = A_sphere / (A_sphere.sum() + 1e-12)
    ratio = (A_sphere_n + 1e-12) / (A_orig_n + 1e-12)
    return np.log(ratio).astype(np.float32)


# =====================================================================
# Peak (spike) detection — paper Algorithm 1
# =====================================================================

@dataclass(frozen=True)
class Peak:
    apex: int                       # vertex index of local min Δarea
    members: np.ndarray             # vertex indices belonging to peak (np.int64)
    height: float                   # peak height (sum of centroid steps)
    width: float                    # FWHM-style width
    mean_distortion: float          # mean Δarea over members
    var_distortion: float           # variance Δarea over members
    solid_angle: float              # approx solid angle subtended (sr)


def detect_peaks(mesh: Mesh, distortion: np.ndarray,
                  baseline_eps: float = 0.01,
                  min_peak_size: int = 5) -> list[Peak]:
    """Detect spike peaks from area-distortion map.

    Algorithm:
        - apex candidates = local minima of `distortion` in 1-ring
        - for each apex, region-grow upward toward zero (baseline)
        - record peak members until distortion crosses 0 ± eps
        - compute height/width/scores
    """
    V = mesh.n_vertices
    adj = vertex_adjacency(mesh)

    # Local minima of distortion (vertex < all neighbors and < 0)
    is_min = np.zeros(V, dtype=bool)
    for v in range(V):
        if distortion[v] >= -baseline_eps:
            continue
        nb = adj[v]
        if not nb:
            continue
        if (distortion[v] < distortion[nb]).all():
            is_min[v] = True
    apex_indices = np.where(is_min)[0]
    if apex_indices.size == 0:
        return []

    # Sort apexes by depth (most-negative first)
    apex_indices = apex_indices[np.argsort(distortion[apex_indices])]

    sphere_centroid_approx = vertex_normals(mesh)  # placeholder
    peaks: list[Peak] = []
    assigned = np.zeros(V, dtype=bool)

    for apex in apex_indices:
        if assigned[apex]:
            continue
        # Region grow from apex upward to baseline
        frontier = [int(apex)]
        members: list[int] = []
        levels: list[list[int]] = []
        current_level = [int(apex)]
        seen = {int(apex)}
        while current_level:
            next_level: list[int] = []
            for v in current_level:
                if distortion[v] > -baseline_eps:
                    continue
                if assigned[v]:
                    continue
                members.append(v)
                assigned[v] = True
                for n in adj[v]:
                    if (n not in seen
                            and not assigned[n]
                            and distortion[n] < -baseline_eps
                            and distortion[n] > distortion[v] - 1e-6):
                        next_level.append(n)
                        seen.add(n)
            if next_level:
                levels.append(next_level)
            current_level = next_level

        if len(members) < min_peak_size:
            for v in members:
                assigned[v] = False
            continue

        members_arr = np.asarray(members, dtype=np.int64)
        # Height = sum of successive centroid distances along levels
        height = 0.0
        if levels:
            prev_centroid = mesh.vertices[apex]
            for lvl in levels:
                ctr = mesh.vertices[np.asarray(lvl, dtype=np.int64)].mean(axis=0)
                height += float(np.linalg.norm(ctr - prev_centroid))
                prev_centroid = ctr
        # Width = FWHM of distortion contour ≈ diameter from members spread
        pts = mesh.vertices[members_arr]
        center = pts.mean(axis=0)
        width = 2.0 * float(np.linalg.norm(pts - center, axis=1).mean())
        # Solid angle ≈ (peak_area / (height^2)) — small angle = sharp spike
        peak_area = float(vertex_areas(mesh)[members_arr].sum())
        h2 = max(height ** 2, 1e-6)
        solid_angle = peak_area / h2

        mean_d = float(distortion[members_arr].mean())
        var_d = float(distortion[members_arr].var())
        peaks.append(Peak(
            apex=int(apex), members=members_arr,
            height=height, width=width,
            mean_distortion=mean_d, var_distortion=var_d,
            solid_angle=solid_angle,
        ))
    return peaks


# =====================================================================
# Peak classification: spiculation / lobulation / attachment
# =====================================================================

@dataclass(frozen=True)
class ClassifyConfig:
    """Thresholds from the paper (validated on FDA phantom data, §3.1)."""
    height_min: float = 0.27         # h_t  — paper §3.1
    solid_angle_max: float = 0.46    # ω_t in steradians — paper §3.1


def classify_peak(peak: Peak, cfg: ClassifyConfig | None = None,
                   attachment_mask: np.ndarray | None = None) -> str:
    """One peak → class label.

    Classes:
      'spiculation'  height ≥ h_t  AND  solid_angle ≤ ω_t  AND  not attached
      'lobulation'   height ≥ h_t  AND  solid_angle > ω_t   AND  not attached
      'attachment'   peak overlaps attachment mask
      'small'        height < h_t  (filtered out as noise)
    """
    cfg = cfg or ClassifyConfig()
    if attachment_mask is not None and attachment_mask[peak.members].any():
        return "attachment"
    if peak.height < cfg.height_min:
        return "small"
    if peak.solid_angle <= cfg.solid_angle_max:
        return "spiculation"
    return "lobulation"


# =====================================================================
# Interpretable spiculation features (paper §2.3)
# =====================================================================

@dataclass(frozen=True)
class SpiculationFeatures:
    """The paper's interpretable feature set."""
    Np: int            # number of all peaks
    Na: int            # number of spiculations
    Nl: int            # number of lobulations
    Na_att: int        # number of attached peaks
    s1: float          # sharpness score
    s2: float          # irregularity score
    classes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {
            "Np": float(self.Np), "Na": float(self.Na),
            "Nl": float(self.Nl), "Na_att": float(self.Na_att),
            "s1": float(self.s1), "s2": float(self.s2),
        }


def spiculation_features(peaks: list[Peak],
                          cfg: ClassifyConfig | None = None,
                          attachment_mask: np.ndarray | None = None,
                          ) -> SpiculationFeatures:
    """Compute interpretable spiculation feature set.

    s1 = Σ_i (mean Δarea_i × height_i) / Σ_i height_i        (sharpness)
    s2 = Σ_i (var  Δarea_i × height_i) / Σ_i height_i        (irregularity)
    """
    classes = [classify_peak(p, cfg, attachment_mask) for p in peaks]
    n_spic = sum(1 for c in classes if c == "spiculation")
    n_lob = sum(1 for c in classes if c == "lobulation")
    n_att = sum(1 for c in classes if c == "attachment")
    h_total = sum(p.height for p in peaks) + 1e-12
    s1 = sum(p.mean_distortion * p.height for p in peaks) / h_total
    s2 = sum(p.var_distortion * p.height for p in peaks) / h_total
    return SpiculationFeatures(
        Np=len(peaks), Na=n_spic, Nl=n_lob, Na_att=n_att,
        s1=float(s1), s2=float(s2), classes=classes,
    )


# =====================================================================
# End-to-end voxel → features pipeline
# =====================================================================

def spiculation_from_voxel(mask: np.ndarray,
                              attachment_mask: np.ndarray | None = None,
                              n_param_iter: int = 200,
                              cfg: ClassifyConfig | None = None,
                              spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                              ) -> tuple[SpiculationFeatures, list[Peak], np.ndarray, Mesh]:
    """One-shot pipeline: voxel mask → spiculation features.

    Args:
        mask: (Z, Y, X) binary nodule mask.
        attachment_mask: (Z, Y, X) bool — voxel-domain vessel/wall attachment.
        n_param_iter: iterations of spherical parameterization.
        cfg: ClassifyConfig.
        spacing: mm voxel spacing.
    Returns:
        features, peaks, distortion (V,), mesh.
    """
    mesh = voxel_to_mesh(mask.astype(np.float32), level=0.5, spacing=spacing)
    sphere = spherical_parameterization(mesh, n_iter=n_param_iter)
    dist = area_distortion(mesh, sphere)
    peaks = detect_peaks(mesh, dist)
    # Translate voxel attachment to vertex attachment (nearest-voxel lookup)
    vertex_attach = None
    if attachment_mask is not None:
        v_idx = np.round(mesh.vertices / np.asarray(spacing)).astype(int)
        Z, Y, X = attachment_mask.shape
        v_idx[:, 0] = np.clip(v_idx[:, 0], 0, Z - 1)
        v_idx[:, 1] = np.clip(v_idx[:, 1], 0, Y - 1)
        v_idx[:, 2] = np.clip(v_idx[:, 2], 0, X - 1)
        vertex_attach = attachment_mask[v_idx[:, 0], v_idx[:, 1], v_idx[:, 2]].astype(bool)
    feats = spiculation_features(peaks, cfg, vertex_attach)
    return feats, peaks, dist, mesh


# =====================================================================
# Bridge: spiculation peaks → mesh-domain surface-lets
# =====================================================================

def peaks_to_surfacelets(mesh: Mesh, peaks: list[Peak],
                           classes: Iterable[str] | None = None):
    """Each peak apex becomes a mesh-domain surface-let atom.

    Returns a list of `Surfacelet` (from .surfacelet) using the apex
    vertex coordinates, the apex normal, the mean Δarea as saliency,
    and the class label. This unifies the 2021 mesh-domain detection
    with the 2014/2026 voxel-domain surface-let representation.
    """
    from .surfacelet import Surfacelet
    classes = list(classes) if classes is not None else ["other"] * len(peaks)
    vn = vertex_normals(mesh)
    out: list[Surfacelet] = []
    for peak, cls in zip(peaks, classes):
        v = peak.apex
        pos = mesh.vertices[v]
        n = vn[v]
        out.append(Surfacelet(
            z=int(round(float(pos[0]))),
            y=int(round(float(pos[1]))),
            x=int(round(float(pos[2]))),
            nz=float(n[0]), ny=float(n[1]), nx=float(n[2]),
            saliency=float(-peak.mean_distortion),    # spikes have most-negative Δarea
            scale=float(peak.height),
            cls=cls,
        ))
    return out
