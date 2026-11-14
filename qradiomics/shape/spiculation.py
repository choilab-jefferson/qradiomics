"""Spiculation quantification — Choi et al. (CMPB 2021).

> Choi W, Nadeem S, Alam SR, Deasy JO, Tannenbaum A, Lu W.
> Reproducible and interpretable spiculation quantification for lung cancer
> screening. Comput Methods Programs Biomed. 2021;200:105839.
> doi:10.1016/j.cmpb.2020.105839 (PMID: 33221055; PMCID: PMC7920914)
>
> Kazhdan M, Solomon J, Ben-Chen M.
> Can Mean-Curvature Flow be Modified to be Non-singular?
> Comput Graph Forum (SGP). 2012;31(5):1745-1754.
>
> Nadeem S, Su Z, Zeng W, Kaufman A, Gu X.
> Spherical Parameterization Balancing Angle and Area Distortions.
> IEEE Trans Vis Comput Graph. 2017;23(6):1663-1676.
> doi:10.1109/TVCG.2016.2542069 (arXiv:1810.09031v1)
> (cited only for the genus-0 conformal-map framing and the Möbius
> area-centering idea; its dynamic Yamabe flow, conformal welding, OMT and
> polar-decomposition balancing are NOT implemented here.)

## Mathematical Pipeline

1. **Genus-0 Surface Extraction & Inverse-Distance Fairing (`voxel_to_mesh`)**:
   - Extracts the largest 3D connected component with internal cavities filled
     at isolevel `0.4` (`convert_for_sphere_param.m` L89) and applies 1-step
     inverse-edge-distance Laplacian fairing (`smoothpatch(s, 1, 1)`).
2. **Spherical Parameterization (`spherical_parameterization`)**:
   - Kazhdan, Solomon & Ben-Chen (2012) conformalized mean curvature flow
     (cMCF): fixed cotangent stiffness `L_0` of the input mesh (cotangents
     clamped at `1e-4`, a departure from Kazhdan), per-step lumped mass
     `M^k`, semi-implicit step `(M^k + dt L_0) X^(k+1) = M^k X^k`. This is the
     same step as the `ConformalizedMCF` executable called by LCSR
     (`main_spherical_param.m`, `--steps 3000 --threads 1`, exe default
     `--stepSize 1e-4`); here
     the defaults are `dt = 0.5` with at most 12 solves, so the schedules differ.
   - Default output: mean-centering + radial projection onto `S^2`. Möbius
     area-mass centering runs only with `mobius_center=True` or
     `balance_alpha > 0`. Opt-in `yamabe_refine` adds discrete Yamabe
     Newton steps (Luo 2004; Springborn-Schröder-Pinkall 2008).
   - `balance_alpha` is a heuristic area-weighted Laplacian relaxation, not
     Nadeem 2017 §3.4 (no OMT, no polar decomposition).
3. **Per-Vertex Log Area Distortion & Gaussian Standardization (`area_distortion`, `zscore_area_distortion`)**:
   - Computes per-vertex incident face area sums on the original mesh `s` and
     spherical parameterization `s_1` (`area_distortions.m`), log-ratio
     `l_ArD = log(A_sphere / A_orig)`, and Gaussian `normfit` Z-score
     standardization `nd = (l_ArD - muhat) / sigmahat` (`spiculation_quantification.m` L346-356).
4. **Baseline-to-Apex Level-Set Rim Tracking & FWHM / Solid Angle Classification (`detect_peaks`, `classify_peak`)**:
   - Tracks negative area-distortion (`nd <= 0`) connected components and their
     1D boundary edge loops (`find_rims.m`, `detect_baselines.m`, `detect_spikes.m`)
     from baseline to local distortion minima (apices), branching when interior
     level-set rims split with perimeter ratio `< 0.6`.
   - Excludes concave surface indentations (`pj > 0.5`, `type = 2` in `measure_spikes.m` L766).
   - Evaluates Full-Width at Half-Minimum (FWHM) contour height `height = 2 * fwhm_height`,
     FWHM width `width = max(pdist(fwhm_baseline))`, and FWHM base disk solid angle
     `solid_angle = A_{FWHM base} / height^2` (`measure_spikes.m` L785-858).
   - Applies physical noise filtering (`height > 3.0 mm` and `width > 1.5 mm`,
     `spiculation_quantification.m` L409) and classifies peaks into acute
     spiculations (`solid_angle <= 0.65 sr`, `Na`), broader lobulations
     (`solid_angle > 0.65 sr`, `Nl`), or attached spikes (`Na_att`).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import warnings
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu, spsolve
from scipy.spatial.distance import pdist

from .mesh_utils import (
    Mesh,
    cotangent_laplacian,
    face_areas,
    laplace_beltrami_bipartition,
    mesh_roundness,
    mesh_volume_and_area,
    smooth_mesh_inverse_distance,
    vertex_adjacency,
    vertex_areas,
    vertex_normals,
    voxel_to_mesh,
)


# =====================================================================
# Möbius Area-Mass Centering & Conformal Spherical Parameterization
# (Kazhdan et al. 2012 cMCF; Möbius centering idea from Nadeem et al. 2017)
# =====================================================================

def _per_vertex_area_lcsr(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex sum of incident triangle areas (`area_distortions.m` L520-533)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    af = 0.5 * np.linalg.norm(np.cross(v2 - v0, v1 - v0), axis=1)
    va = np.zeros(len(verts), dtype=np.float64)
    for k in range(3):
        np.add.at(va, faces[:, k], af)
    return va


def mobius_area_center(
    sphere_pos: np.ndarray,
    weights: np.ndarray,
    n_steps: int = 25,
) -> np.ndarray:
    """Möbius transformation on S^2 (idea from Nadeem et al. 2017 §3.3)
    iteratively normalizing the area-weighted center of mass to the origin (0, 0, 0).
    """
    w = weights.astype(np.float64)
    w = w / (w.sum() + 1e-12)
    pos = sphere_pos.astype(np.float64).copy()
    pos = pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-12)

    for _ in range(n_steps):
        c = (w[:, None] * pos).sum(axis=0)
        norm_c = float(np.linalg.norm(c))
        if norm_c < 1e-6:
            break
        a = 0.5 * c
        na2 = float(np.dot(a, a))
        if na2 >= 0.90:
            a = a * (0.85 / np.sqrt(na2))
            na2 = float(np.dot(a, a))
        s = np.sqrt(1.0 - na2)
        ax = pos @ a
        denom = (1.0 - ax)[:, None] + 1e-12
        coeff_a = (ax / (1.0 + s) - 1.0)[:, None]
        pos = (s * pos + coeff_a * a[None, :]) / denom
        pos = pos / (np.linalg.norm(pos, axis=1, keepdims=True) + 1e-12)
    return pos


def _unique_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected edges `(E, 2)` and per-face edge index `(F, 3)`.

    Face edge `k` is the edge opposite corner `k`.
    """
    F = len(faces)
    e = np.stack([faces[:, [1, 2]], faces[:, [2, 0]], faces[:, [0, 1]]], axis=1)
    e = np.sort(e.reshape(-1, 2), axis=1)
    edges, inv = np.unique(e, axis=0, return_inverse=True)
    return edges, inv.reshape(F, 3)


def _corner_angles(lf: np.ndarray) -> np.ndarray | None:
    """Corner angles `(F, 3)` from opposite edge lengths `(F, 3)`; None if a
    triangle inequality fails (vertex-scaling metric left the valid
    region; no intrinsic edge flips are performed)."""
    a, b, c = lf[:, 0], lf[:, 1], lf[:, 2]
    if np.any(a >= b + c) or np.any(b >= c + a) or np.any(c >= a + b):
        return None
    ca = np.clip((b * b + c * c - a * a) / (2 * b * c), -1.0, 1.0)
    cb = np.clip((c * c + a * a - b * b) / (2 * c * a), -1.0, 1.0)
    cc = np.clip((a * a + b * b - c * c) / (2 * a * b), -1.0, 1.0)
    return np.arccos(np.stack([ca, cb, cc], axis=1))


def _mean_abs_corner_log_ratio(ang: np.ndarray, ang_ref: np.ndarray) -> float:
    return float(np.mean(np.abs(np.log((ang + 1e-12) / (ang_ref + 1e-12)))))


def _intrinsic_delaunay_flips(
    F: np.ndarray, l0d: dict, u: np.ndarray, max_pass: int = 50,
) -> tuple[np.ndarray, int]:
    """Flip edges of an intrinsic triangulation until it is Delaunay.

    The metric is `l_ij = exp((u_i + u_j)/2) * l0_ij`. Non-Delaunay edges
    (opposite cotangents summing below 0) are flipped and the new edge's base
    length is set by the Ptolemy relation
    `l0_kl = (l0_il * l0_jk + l0_lj * l0_ki) / l0_ij`, which is invariant under
    vertex scaling (Gillespie, Springborn & Crane 2021; Springborn, Schröder
    & Pinkall 2008). `l0d` (sorted vertex pair -> base length) is updated in
    place. Edges whose flip would duplicate an existing edge are skipped; a
    non-manifold edge set disables flipping. Returns `(faces, n_flips)`.
    """
    F = F.copy()
    n_flips = 0

    def key(a, b):
        return (a, b) if a < b else (b, a)

    for _ in range(max_pass):
        edges, fe = _unique_edges(F)
        l0 = np.array([l0d[(int(a), int(b))] for a, b in edges])
        ll = np.exp(0.5 * (u[edges[:, 0]] + u[edges[:, 1]])) * l0
        ang = _corner_angles(ll[fe])
        if ang is None:
            break
        flat = fe.ravel()
        if np.any(np.bincount(flat, minlength=len(edges)) != 2):
            break
        s = np.zeros(len(edges))
        np.add.at(s, flat, 1.0 / np.tan(np.clip(ang, 1e-9, np.pi - 1e-9)).ravel())
        bad = np.where(s < -1e-10)[0]
        if len(bad) == 0:
            break
        order = np.argsort(flat, kind="stable")
        inc = order.reshape(-1, 2)  # per edge: two flat (face*3 + corner) slots
        used: set[int] = set()
        flipped = 0
        for e in bad[np.argsort(s[bad])]:
            (f1, c1), (f2, c2) = divmod(int(inc[e, 0]), 3), divmod(int(inc[e, 1]), 3)
            if f1 in used or f2 in used:
                continue
            i, j, k = int(F[f1, (c1 + 1) % 3]), int(F[f1, (c1 + 2) % 3]), int(F[f1, c1])
            m = int(F[f2, c2])
            if k == m or key(k, m) in l0d:
                continue
            if not (int(F[f2, (c2 + 1) % 3]) == j and int(F[f2, (c2 + 2) % 3]) == i):
                continue  # inconsistent orientation
            l0d[key(k, m)] = (l0d[key(i, m)] * l0d[key(j, k)]
                              + l0d[key(m, j)] * l0d[key(k, i)]) / l0d[key(i, j)]
            del l0d[key(i, j)]
            F[f1] = (i, m, k)
            F[f2] = (m, j, k)
            used.update((f1, f2))
            flipped += 1
        n_flips += flipped
        if flipped == 0:
            break
    return F, n_flips


def _yamabe_refine(
    verts: np.ndarray, faces: np.ndarray, sphere: np.ndarray, n_newton: int,
    n_embed: int = 100,
    flip: bool = False,
) -> tuple[np.ndarray, dict]:
    """Discrete Yamabe (vertex-scaling) Newton refinement of a spherical map.

    Euclidean vertex scaling (Luo 2004; Springborn, Schröder & Pinkall 2008):
    `l_ij = exp((u_i + u_j) / 2) * l0_ij` with `l0` the input-mesh edge lengths,
    so the metric stays in the discrete conformal class of the input mesh.
      1. `u` is initialized by least squares from the cMCF sphere chord lengths
         (`log l_sph = (u_i + u_j)/2 + log l0`); if that metric violates a
         triangle inequality, `u` starts as a constant (scaled input metric).
      2. Target vertex curvature = angle defects of the cMCF chordal sphere
         (a realizable round-sphere target summing to 4*pi).
      3. Newton: `L(u) du = -(K(u) - K_target)`, `L` = cotangent Laplacian of the
         current metric, backtracking on the curvature residual.
      4. Re-embedding on S^2: Jacobi-style edge-length stress descent toward the
         Yamabe edge lengths, starting from the cMCF sphere, with radial
         projection after each sweep.

    With `flip=True`, the intrinsic triangulation is kept Delaunay after the
    init and after every accepted Newton step (`_intrinsic_delaunay_flips`,
    Ptolemy flips); Newton then runs on the flipped triangulation and the
    S^2 re-embedding uses its edges. Without it (default), no flips are done.
    In both modes a Newton step that violates a triangle inequality is halved,
    and the loop stops if it cannot be fixed.
    Fallback: the input sphere is returned (with a warning) if the result does
    not lower mean |log corner-angle ratio| vs the input mesh or flips faces.

    Returns `(sphere, info)`; `info["applied"]` is False on fallback and
    `info["reason"]` is one of `applied`, `degenerate_edge`, `invalid_triangle`,
    `solver_failure`, `nan`, `flipped_faces`, `no_gain`.
    """
    import warnings

    from scipy.sparse.linalg import spsolve as _spsolve

    info: dict = {"applied": False, "reason": "", "newton_iters": 0}

    def _fallback(reason):
        info["reason"] = reason
        warnings.warn(f"yamabe_refine: {reason}; returning cMCF sphere")
        return sphere, info

    V = len(verts)
    edges, fe = _unique_edges(faces)
    E = len(edges)
    l0 = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    ls = np.linalg.norm(sphere[edges[:, 0]] - sphere[edges[:, 1]], axis=1)
    if np.any(l0 <= 0) or np.any(ls <= 0):
        return _fallback("degenerate_edge")
    tri_faces, edges0, fe0 = faces, edges, fe

    # 1. least-squares init of u
    A = coo_matrix((np.full(2 * E, 0.5), (np.repeat(np.arange(E), 2), edges.ravel())),
                   shape=(E, V)).tocsr()
    rhs = A.T @ np.log(ls / l0)
    try:
        u = _spsolve((A.T @ A + diags(np.full(V, 1e-9))).tocsc(), rhs)
    except Exception:  # noqa: BLE001
        return _fallback("solver_failure")
    if not np.all(np.isfinite(u)):
        return _fallback("nan")

    def _metric(uu):
        ll = np.exp(0.5 * (uu[edges[:, 0]] + uu[edges[:, 1]])) * l0
        return ll, _corner_angles(ll[fe])

    def _curv(ang):
        k = np.full(V, 2.0 * np.pi)
        for c in range(3):
            np.subtract.at(k, tri_faces[:, c], ang[:, c])
        return k

    ang_sph = _corner_angles(ls[fe])
    ang_orig = _corner_angles(l0[fe])
    if ang_sph is None or ang_orig is None:
        return _fallback("invalid_triangle")
    k_t = _curv(ang_sph)

    ll, ang = _metric(u)
    if ang is None:  # LS init invalid: start from the uniformly scaled input metric
        u = np.full(V, float(np.mean(np.log(ls / l0))))
        ll, ang = _metric(u)
    info["n_flips"] = 0

    def _reflip(uu):
        nonlocal tri_faces, edges, fe, l0, E
        l0d = {(int(a), int(b)): float(x) for (a, b), x in zip(edges, l0)}
        tri_faces, nf = _intrinsic_delaunay_flips(tri_faces, l0d, uu)
        if nf:
            edges, fe = _unique_edges(tri_faces)
            E = len(edges)
            l0 = np.array([l0d[(int(a), int(b))] for a, b in edges])
        info["n_flips"] += nf
        return _metric(uu)

    if flip and ang is not None:
        ll, ang = _reflip(u)
    if ang is None:
        return _fallback("invalid_triangle")
    res = _curv(ang) - k_t
    for _ in range(int(n_newton)):
        cot = 1.0 / np.tan(np.clip(ang, 1e-6, np.pi - 1e-6))
        w = np.zeros(E)
        for c in range(3):
            np.add.at(w, fe[:, c], 0.5 * cot[:, c])
        H = coo_matrix((np.concatenate([-w, -w]),
                        (np.concatenate([edges[:, 0], edges[:, 1]]),
                         np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(V, V)).tocsr()
        H = H + diags(-np.asarray(H.sum(axis=1)).ravel() + 1e-8)
        try:
            du = _spsolve(H.tocsc(), -(res - res.mean()))
        except Exception:  # noqa: BLE001
            return _fallback("solver_failure")
        if not np.all(np.isfinite(du)):
            return _fallback("nan")
        t, ok = 1.0, False
        for _ls in range(12):
            ll_n, ang_n = _metric(u + t * du)
            if ang_n is not None:
                res_n = _curv(ang_n) - k_t
                if np.linalg.norm(res_n) < np.linalg.norm(res):
                    ok = True
                    break
            t *= 0.5
        if not ok:
            break
        u, ll, ang, res = u + t * du, ll_n, ang_n, res_n
        info["newton_iters"] += 1
        if flip:
            ll, ang = _reflip(u)
            if ang is None:
                return _fallback("invalid_triangle")
            res = _curv(ang) - k_t

    # 4. re-embed on S^2 toward the Yamabe edge lengths
    ll = ll * np.sqrt((ls ** 2).sum() / ((ll ** 2).sum() + 1e-12))
    deg = np.bincount(edges.ravel(), minlength=V).astype(np.float64)[:, None]
    x = sphere.astype(np.float64).copy()
    for _ in range(int(n_embed)):
        d = x[edges[:, 0]] - x[edges[:, 1]]
        dn = np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        f = (dn - ll[:, None]) * d / dn
        g = np.zeros_like(x)
        np.add.at(g, edges[:, 0], f)
        np.subtract.at(g, edges[:, 1], f)
        x -= 0.5 * g / deg
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12

    def _n_flipped(p):
        n = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
        return int(np.sum(np.einsum("fk,fk->f", n, p[faces].mean(axis=1)) < 0))

    if not np.all(np.isfinite(x)):
        return _fallback("nan")
    info["angle_dist_before"] = _mean_abs_corner_log_ratio(ang_sph, ang_orig)
    ang_x = _corner_angles(np.linalg.norm(x[edges0[:, 0]] - x[edges0[:, 1]], axis=1)[fe0])
    if ang_x is None:
        return _fallback("invalid_triangle")
    if _n_flipped(x) > _n_flipped(sphere):
        return _fallback("flipped_faces")
    if _mean_abs_corner_log_ratio(ang_x, ang_orig) >= info["angle_dist_before"]:
        return _fallback("no_gain")
    info.update(applied=True, reason="applied",
                angle_dist_after=_mean_abs_corner_log_ratio(ang_x, ang_orig))
    return x, info


def spherical_parameterization(
    mesh: Mesh,
    n_iter: int = 12,
    step: float = 0.5,
    balance_alpha: float = 0.0,
    mobius_center: bool = False,
    yamabe_refine: int = 0,
    yamabe_flip: bool = False,
    return_info: bool = False,
) -> "np.ndarray | tuple[np.ndarray, dict]":
    """Spherical parameterization of a genus-0 mesh via conformalized MCF.

    Core flow is Kazhdan, Solomon & Ben-Chen (2012) conformalized mean
    curvature flow (cMCF), the same semi-implicit step as LCSR's
    `ConformalizedMCF` (`main_spherical_param.m`):
      1. `L_0 = cotangent_laplacian(mesh, clamp_min=1e-4)` is built once on the
         input mesh and kept fixed. Cotangents are clamped below at `1e-4`
         (obtuse-angle weights never go negative); this departs from Kazhdan.
      2. Each step solves `(M^k + dt L_0) X^(k+1) = M^k X^k` with lumped mass
         `M^k` (1/3 incident-face area), then re-centers (area-weighted) and
         rescales to total area 4*pi. LCSR runs the exe with
         `--steps 3000` (default `--stepSize 1e-4`); here `dt = 0.5` with 12 solves (default `n_iter=12`)
         and LU factorization reuse across Cartesian dimensions.
      3. Output = mean-centering + radial projection onto `S^2`.
      4. Opt-in `yamabe_refine > 0`: discrete Yamabe (vertex-scaling)
         Newton refinement, see `_yamabe_refine`.
      5. Opt-in `mobius_center=True`: Möbius area-mass centering only.
      6. Only if `balance_alpha > 0`: Möbius area-mass centering
         (`mobius_area_center`, 15 steps), then a heuristic area-weighted
         1-ring Laplacian relaxation on the sphere. This is NOT Nadeem et al.
         2017 §3.4 (no optimal mass transport, no polar decomposition).

    This implementation is based on Kazhdan et al. (2012) Conformalized Mean
    Curvature Flow (cMCF) combined with Nadeem et al. (2017) Möbius area normalization
    and Choi et al. (CMPB 2021) spiculation metric quantification.

    Args:
        mesh: Genus-0 triangular surface mesh.
        n_iter: Number of semi-implicit solves (default: 12).
        step: Semi-implicit time step `dt` (default: 0.5).
        balance_alpha: Strength in `[0, 1]` of the heuristic area relaxation
            (`0.0` default, used for Choi et al. 2021 spiculation features).
        mobius_center: Opt-in. Apply `mobius_area_center` (25 steps, input-mesh
            vertex areas as weights) on the default path. Ignored when
            `balance_alpha > 0` (that path already centers).
        yamabe_refine: Opt-in. Number of discrete Yamabe Newton iterations
            after cMCF (`_yamabe_refine`); applied before Möbius centering.
            Falls back to the cMCF sphere with a warning if it does not help.
        yamabe_flip: Opt-in. Keep the intrinsic triangulation Delaunay during
            `yamabe_refine` (Ptolemy edge flips, see `_intrinsic_delaunay_flips`).
        return_info: Opt-in. If True, return `(sphere, info)` where
            `info["yamabe"]` holds the `_yamabe_refine` diagnostics
            (applied flag, fallback reason, Newton iterations).

    Returns:
        (V, 3) float32 unit-sphere vertex coordinates.
    """
    V = mesh.n_vertices
    if V == 0:
        return np.zeros((0, 3), dtype=np.float32)

    n_solves = max(int(n_iter), 1)
    dt = float(step)

    L0 = cotangent_laplacian(mesh, clamp_min=1e-4)
    pos = mesh.vertices.astype(np.float64).copy()
    pos -= pos.mean(axis=0, keepdims=True)
    va_orig = _per_vertex_area_lcsr(pos, mesh.faces) / 3.0
    pos /= np.sqrt((va_orig.sum() + 1e-12) / (4.0 * np.pi))

    for _ in range(n_solves):
        va = np.maximum(_per_vertex_area_lcsr(pos, mesh.faces) / 3.0, 1e-8)
        M = diags(va, 0, shape=(V, V), format="csr")
        A_mat = (M + dt * L0).tocsc()
        rhs = M @ pos
        try:
            lu = splu(A_mat)
            for dim in range(3):
                pos[:, dim] = lu.solve(rhs[:, dim])
        except Exception as exc:
            warnings.warn(f"cMCF linear solve failed: {exc}. Terminating iteration early.")
            break
        va_new = _per_vertex_area_lcsr(pos, mesh.faces) / 3.0
        w = va_new / (va_new.sum() + 1e-12)
        pos -= (w[:, None] * pos).sum(axis=0, keepdims=True)
        pos /= np.sqrt((va_new.sum() + 1e-12) / (4.0 * np.pi))

    info: dict = {}
    ce1 = pos.mean(axis=0, keepdims=True)
    v1 = pos - ce1
    nv1 = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-12)

    if yamabe_refine > 0:
        nv1, yinfo = _yamabe_refine(mesh.vertices.astype(np.float64), mesh.faces, nv1,
                                    int(yamabe_refine), flip=bool(yamabe_flip))
        info["yamabe"] = yinfo
    if mobius_center and balance_alpha <= 0.0:
        nv1 = mobius_area_center(nv1, va_orig)

    # Optional heuristic area relaxation (not Nadeem 2017 §3.4)
    if balance_alpha > 0.0:
        alpha = float(np.clip(balance_alpha, 0.0, 1.0))
        nv1 = mobius_area_center(nv1, va_orig, n_steps=15)
        adj = vertex_adjacency(mesh)
        for _ in range(int(round(10 * alpha)) + 1):
            va_sph = _per_vertex_area_lcsr(nv1, mesh.faces) / 3.0
            ratio = (va_orig / (va_orig.sum() + 1e-12)) / (va_sph / (va_sph.sum() + 1e-12) + 1e-12)
            w_bal = np.clip(ratio ** (0.25 * alpha), 0.2, 5.0)
            new_pos = nv1.copy()
            for v, nbs in enumerate(adj):
                if nbs:
                    nb_arr = np.asarray(nbs, dtype=np.int64)
                    wv = w_bal[nb_arr][:, None]
                    new_pos[v] = (wv * nv1[nb_arr]).sum(axis=0) / (wv.sum() + 1e-12)
            nv1 = (1.0 - 0.3 * alpha) * nv1 + (0.3 * alpha) * new_pos
            nv1 = nv1 / (np.linalg.norm(nv1, axis=1, keepdims=True) + 1e-12)

    if return_info:
        return nv1.astype(np.float32), info
    return nv1.astype(np.float32)


# =====================================================================
# Per-vertex area & angle distortion (Nadeem et al. 2017 & CMPB 2021)
# =====================================================================

def area_distortion(mesh: Mesh, sphere_pos: np.ndarray) -> np.ndarray:
    """Per-vertex normalized log area ratio `log(A_sphere_norm / A_orig_norm)`.

    Negative → spike / protrusion (collapses in area on the conformal sphere).
    Positive → smooth base (expands in area on the conformal sphere).
    """
    A_orig = _per_vertex_area_lcsr(mesh.vertices.astype(np.float64), mesh.faces)
    A_sphere = _per_vertex_area_lcsr(sphere_pos.astype(np.float64), mesh.faces)
    A_orig_n = A_orig / (A_orig.sum() + 1e-12)
    A_sphere_n = A_sphere / (A_sphere.sum() + 1e-12)
    ratio = (A_sphere_n + 1e-12) / (A_orig_n + 1e-12)
    return np.log(ratio).astype(np.float32)


def zscore_area_distortion(mesh: Mesh, sphere_pos: np.ndarray) -> np.ndarray:
    """Gaussian `normfit` standardized log area distortion `nd = (l_ArD - mu) / sigma`.

    Matches `spiculation_quantification.m` lines 346-356.
    """
    l_ard = area_distortion(mesh, sphere_pos).astype(np.float64)
    finite = np.isfinite(l_ard)
    if not np.any(finite):
        return np.zeros_like(l_ard, dtype=np.float32)
    mu = float(np.mean(l_ard[finite]))
    sigma = float(np.std(l_ard[finite], ddof=1)) + 1e-12
    nd = (l_ard - mu) / sigma
    return nd.astype(np.float32)


def angle_distortion(mesh: Mesh, sphere_pos: np.ndarray) -> np.ndarray:
    """Per-face log minimum-angle distortion (`angle_distortions.m`).

    Returns `(F,)` float32 array of `log(min_angle_sphere / min_angle_orig)`.
    Near-zero values confirm conformal (angle-preserving) parameterization.
    """
    def _min_face_angles(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        F = len(faces)
        angles = np.zeros((3, F), dtype=np.float64)
        for i in range(3):
            i2 = (i + 1) % 3
            i3 = (i + 2) % 3
            u = verts[faces[:, i2]] - verts[faces[:, i]]
            v = verts[faces[:, i3]] - verts[faces[:, i]]
            u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            cos_theta = np.clip(np.einsum("ij,ij->i", u, v), -1.0, 1.0)
            angles[i] = np.arccos(cos_theta)
        return np.min(angles, axis=0)

    a_orig = _min_face_angles(mesh.vertices.astype(np.float64), mesh.faces)
    a_sph = _min_face_angles(sphere_pos.astype(np.float64), mesh.faces)
    return np.log((a_sph + 1e-12) / (a_orig + 1e-12)).astype(np.float32)


# =====================================================================
# Peak (spike) detection — Choi et al. (CMPB 2021) §2.3
# =====================================================================

@dataclass(frozen=True)
class Peak:
    apex: int                       # vertex index of local min Δarea
    members: np.ndarray             # vertex indices belonging to peak (np.int64)
    height: float                   # peak height (2 * fwhm_height in LCSR, or centroid step sum)
    width: float                    # FWHM contour max diameter (mm)
    mean_distortion: float          # mean Δarea over members
    var_distortion: float           # variance Δarea over members
    solid_angle: float              # FWHM base disk solid angle A_FWHM / height^2 (sr)
    height1: float = 0.0            # direct baseline-centroid to apex Euclidean distance (mm)
    rimchange: float = 0.0          # perimeter contraction from baseline to innermost rim (mm)

    @property
    def vertices_idx(self) -> np.ndarray:
        """Alias for `members` for API backwards-compatibility with vertex-level indexing."""
        return self.members


def _order_boundary_loop(boundary_verts: np.ndarray, verts_3d: np.ndarray) -> np.ndarray:
    """Order boundary vertices into a closed 2D loop in their best-fit plane."""
    pts = verts_3d[boundary_verts]
    if len(pts) < 3:
        return boundary_verts
    ctr = pts.mean(axis=0)
    centered = pts - ctr
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    u = centered @ vh[0]
    v = centered @ vh[1]
    angles = np.arctan2(v, u)
    return boundary_verts[np.argsort(angles)]


def _longest_face_boundary_loop(
    comp: np.ndarray,
    faces: np.ndarray,
    n_vertices: int,
) -> np.ndarray | None:
    """Extract the longest 1D boundary edge loop of faces with all 3 vertices in `comp`
    (`detect_spikes.m` L725: `is_rim_face = sum(ismember(s.faces, next_rim), 2) == 3`
    followed by `find_rims.m`).
    """
    in_comp = np.zeros(n_vertices, dtype=bool)
    in_comp[comp] = True
    f_mask = in_comp[faces[:, 0]] & in_comp[faces[:, 1]] & in_comp[faces[:, 2]]
    sub_f = faces[f_mask]
    if len(sub_f) == 0:
        return None
    directed = np.vstack([sub_f[:, [0, 1]], sub_f[:, [1, 2]], sub_f[:, [2, 0]]])
    sorted_e = np.sort(directed, axis=1)
    _, idx, counts = np.unique(sorted_e, axis=0, return_index=True, return_counts=True)
    boundary_e = sorted_e[idx[counts == 1]]
    if len(boundary_e) < 3:
        return None

    b_adj: dict[int, list[int]] = defaultdict(list)
    for ei, (u, v) in enumerate(boundary_e):
        b_adj[int(u)].append(ei)
        b_adj[int(v)].append(ei)

    used = np.zeros(len(boundary_e), dtype=bool)
    best_loop: np.ndarray | None = None
    for start_ei in range(len(boundary_e)):
        if used[start_ei]:
            continue
        used[start_ei] = True
        u, v = int(boundary_e[start_ei, 0]), int(boundary_e[start_ei, 1])
        rim = [v]
        curr = v
        while True:
            next_ei = -1
            for cand_ei in b_adj[curr]:
                if not used[cand_ei]:
                    next_ei = cand_ei
                    break
            if next_ei == -1:
                break
            used[next_ei] = True
            eu, ev = int(boundary_e[next_ei, 0]), int(boundary_e[next_ei, 1])
            nxt = ev if eu == curr else eu
            rim.append(nxt)
            curr = nxt
        if len(rim) >= 3 and (best_loop is None or len(rim) > len(best_loop)):
            best_loop = np.asarray(rim, dtype=np.int64)
    return best_loop


def detect_peaks(
    mesh: Mesh,
    distortion: np.ndarray,
    baseline_eps: float = 0.0,
    min_peak_size: int = 4,
    zscore_normalize: bool = True,
    filter_concave: bool = True,
) -> list[Peak]:
    """Detect surface protrusions (spikes and lobulations) from the area-distortion map.

    Implements the Choi et al. (*CMPB* 2021) `LCSR` level-set rim tracking and
    FWHM geometric measurement (`detect_baselines.m`, `detect_spikes.m`, `measure_spikes.m`):
      1. Standardizes `distortion` via Gaussian `normfit` (`nd = (d - mu) / sigma`)
         when `zscore_normalize=True`.
      2. Identifies connected negative-distortion components (`nd <= -baseline_eps`)
         and recursively tracks their 1D boundary rims (`find_rims.m`) inward to
         local distortion minima (apices), branching when interior level-set
         components split with perimeter ratio `< 0.6` (`detect_spikes.m` L680).
      3. Filters out concave inward indentations (`pj > 0.5`, `type = 2` in
         `measure_spikes.m` L766) when `filter_concave=True`.
      4. Measures Full-Width at Half-Minimum (FWHM) height `2 * fwhm_height`,
         FWHM contour diameter `width = max(pdist(fwhm_baseline))`, and FWHM base
         disk solid angle `solid_angle = A_{FWHM base} / height^2`.
    """
    V = mesh.n_vertices
    if V == 0 or mesh.n_faces == 0:
        return []

    verts = mesh.vertices.astype(np.float64)
    faces = mesh.faces
    l_ard = distortion.astype(np.float64)
    std_val = float(np.std(l_ard, ddof=1))
    q05_val = float(np.quantile(l_ard, 0.05)) if len(l_ard) > 0 else 0.0
    is_near_perfect_sphere = (std_val < 0.16) and (q05_val > -0.25)
    if zscore_normalize:
        nd = (l_ard - float(np.mean(l_ard))) / (std_val + 1e-12) if std_val > 1e-8 else l_ard.copy()
    else:
        nd = l_ard.copy()

    vn = vertex_normals(mesh).astype(np.float64)
    adj = vertex_adjacency(mesh)

    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    all_edges = np.vstack([e01, e12, e20, e01[:, ::-1], e12[:, ::-1], e20[:, ::-1]])
    edges_u = all_edges[:, 0]
    edges_v = all_edges[:, 1]

    def _submesh_components(v_subset: np.ndarray) -> list[np.ndarray]:
        if len(v_subset) == 0:
            return []
        in_sub = np.zeros(V, dtype=bool)
        in_sub[v_subset] = True
        mask_e = in_sub[edges_u] & in_sub[edges_v]
        if not np.any(mask_e):
            return [np.array([v], dtype=np.int64) for v in v_subset]
        remap = np.full(V, -1, dtype=np.int64)
        remap[v_subset] = np.arange(len(v_subset))
        eu = remap[edges_u[mask_e]]
        ev = remap[edges_v[mask_e]]
        G = coo_matrix(
            (np.ones(len(eu), dtype=np.int8), (eu, ev)),
            shape=(len(v_subset), len(v_subset)),
        ).tocsr()
        n_c, lbl = connected_components(G, directed=False)
        return [v_subset[lbl == c] for c in range(n_c) if np.sum(lbl == c) >= 1]

    def _get_rim_of_component(comp: np.ndarray) -> np.ndarray:
        loop = _longest_face_boundary_loop(comp, faces, V)
        if loop is not None and len(loop) >= 3:
            return loop
        in_comp = np.zeros(V, dtype=bool)
        in_comp[comp] = True
        rim = [v for v in comp if any(not in_comp[nb] for nb in adj[v])]
        if not rim:
            return comp
        return _order_boundary_loop(np.asarray(rim, dtype=np.int64), verts)

    def _rim_perimeter(rim: np.ndarray) -> float:
        if len(rim) < 2:
            return 0.0
        pts = verts[rim]
        pts_closed = np.vstack([pts, pts[0]])
        return float(np.linalg.norm(np.diff(pts_closed, axis=0), axis=1).sum())

    neg_verts = np.where(nd <= -float(baseline_eps))[0]
    init_comps = [c for c in _submesh_components(neg_verts) if len(c) >= max(min_peak_size, 3)]
    raw_spikes: list[dict] = []

    def _track_component(
        comp: np.ndarray,
        rims_list: list[np.ndarray],
        centers_list: list[np.ndarray],
        base_verts: np.ndarray,
    ) -> None:
        rim = _get_rim_of_component(comp)
        ctr = verts[rim].mean(axis=0)
        new_rims = rims_list + [rim]
        new_centers = centers_list + [ctr]

        in_rim = set(int(x) for x in rim)
        interior = np.asarray([v for v in comp if int(v) not in in_rim], dtype=np.int64)
        if len(interior) > 0:
            thresh = float(np.min(nd[rim]))
            deeper = interior[nd[interior] < thresh]
            if len(deeper) == 0 and len(interior) >= 4:
                thresh2 = float(np.percentile(nd[rim], 25))
                deeper = interior[nd[interior] < thresh2]
        else:
            deeper = np.array([], dtype=np.int64)

        sub_comps = [c for c in _submesh_components(deeper) if len(c) >= max(min_peak_size, 3)]
        if len(sub_comps) == 0 or len(new_rims) >= 40:
            pool = deeper if len(deeper) > 0 else (interior if len(interior) > 0 else comp)
            apex = int(pool[np.argmin(nd[pool])])
            final_centers = np.vstack(new_centers + [verts[apex]])
            raw_spikes.append({
                "baseline": new_rims[0],
                "rims": new_rims,
                "l_center": final_centers,
                "vertices": base_verts,
                "apex": apex,
            })
            return

        if len(sub_comps) == 1:
            _track_component(sub_comps[0], new_rims, new_centers, base_verts)
        else:
            perims = [_rim_perimeter(_get_rim_of_component(sc)) for sc in sub_comps]
            tot_p = sum(perims) + 1e-12
            for sc, pr in zip(sub_comps, perims):
                if pr / tot_p < 0.6:
                    _track_component(sc, [new_rims[-1]], [new_centers[-1]], sc)
                else:
                    _track_component(sc, new_rims, new_centers, base_verts)

    for c in init_comps:
        _track_component(c, [], [], c)

    peaks: list[Peak] = []
    seen_apex: set[int] = set()
    for spk in raw_spikes:
        rims = spk["rims"]
        l_center = spk["l_center"]
        apex = int(spk["apex"])
        if apex in seen_apex:
            continue
        if len(rims) == 1 and apex in rims[0]:
            continue
        if is_near_perfect_sphere and float(l_ard[apex]) > -1.60:
            continue

        # Concavity filter (`measure_spikes.m` L763-768: `type = 2` if `pj > 0.5`)
        if filter_concave:
            if len(l_center) >= 3:
                normal_base = np.mean(l_center[:-1] - l_center[1:], axis=0)
            else:
                normal_base = l_center[0] - l_center[-1]
            check_verts = np.unique(np.concatenate([[apex], rims[-1]]))
            pj = float(np.mean((vn[check_verts] @ normal_base) > 0))
            if pj > 0.5:
                continue

        # FWHM contour and height (`measure_spikes.m` L785-808)
        all_rim_v = np.unique(np.concatenate(rims))
        halfmin = float(np.min(nd[all_rim_v])) / 2.0
        fwhmi = 0
        for ri, r_verts in enumerate(rims):
            if float(np.mean(nd[r_verts])) < halfmin:
                fwhmi = ri
                break
        if fwhmi > 0:
            fwhm_base = rims[fwhmi - 1]
            diffs = np.diff(l_center[fwhmi - 1:], axis=0)
            fwhm_height = float(np.linalg.norm(diffs, axis=1).sum())
        else:
            fwhm_base = rims[0]
            diffs = np.diff(l_center, axis=0)
            fwhm_height = 0.5 * float(np.linalg.norm(diffs, axis=1).sum())

        height = 2.0 * fwhm_height
        height1 = float(np.linalg.norm(l_center[-1] - l_center[0]))
        if len(fwhm_base) < 3:
            continue

        pts_fwhm = verts[fwhm_base]
        width = float(np.max(pdist(pts_fwhm))) if len(pts_fwhm) >= 2 else 0.0

        # Solid angle of the FWHM base disk (`measure_spikes.m` L843-858)
        ctr_fwhm = pts_fwhm.mean(axis=0)
        p_next = np.roll(pts_fwhm, -1, axis=0)
        fan_areas = 0.5 * np.linalg.norm(
            np.cross(pts_fwhm - ctr_fwhm, p_next - ctr_fwhm), axis=1
        )
        solid_angle = float(fan_areas.sum()) / max(height ** 2, 1e-6)

        v_all = np.unique(np.concatenate(rims + [spk["vertices"], [apex]])).astype(np.int64)
        if len(v_all) < min_peak_size:
            continue

        seen_apex.add(apex)
        rimchange = _rim_perimeter(rims[0]) - _rim_perimeter(rims[-1])
        peaks.append(
            Peak(
                apex=apex,
                members=v_all,
                height=height,
                width=width,
                mean_distortion=float(np.mean(l_ard[v_all])),
                var_distortion=float(np.var(l_ard[v_all])),
                solid_angle=solid_angle,
                height1=height1,
                rimchange=rimchange,
            )
        )
    return peaks


# =====================================================================
# Peak classification: spiculation / lobulation / attachment
# =====================================================================

@dataclass(frozen=True)
class ClassifyConfig:
    """Thresholds from Choi et al. (*CMPB* 2021, `spiculation_quantification.m` L211, L409 & `measure_spikes.m` L856):
    - `height_min = 3.0` mm (`th_noise = 3`)
    - `width_min = 1.5` mm (`th_noise / 2 = 1.5`)
    - `solid_angle_max = 0.65` sr (`spikes(pki).angle <= 0.65` -> spiculation, `> 0.65` -> lobulation)
    """
    height_min: float = 3.0          # th_noise in mm (spiculation_quantification.m L211, L409)
    width_min: float = 1.5           # th_noise / 2 in mm (spiculation_quantification.m L409)
    solid_angle_max: float = 0.65    # omega_t in steradians (measure_spikes.m L856)


def classify_peak(
    peak: Peak,
    cfg: ClassifyConfig | None = None,
    attachment_mask: np.ndarray | None = None,
) -> str:
    """One peak → class label.

    Classes:
      'small'        `height <= height_min` OR `width <= width_min` (filtered out as noise)
      'attachment'   peak overlaps `attachment_mask` (`> 50%` of rim/members or any attached vertex)
      'spiculation'  `height > height_min` AND `width > width_min` AND `solid_angle <= solid_angle_max`
      'lobulation'   `height > height_min` AND `width > width_min` AND `solid_angle > solid_angle_max`
    """
    cfg = cfg or ClassifyConfig()
    if peak.height <= cfg.height_min or peak.width <= cfg.width_min:
        return "small"
    if attachment_mask is not None and attachment_mask[peak.members].any():
        return "attachment"
    if peak.solid_angle <= cfg.solid_angle_max:
        return "spiculation"
    return "lobulation"


# =====================================================================
# Interpretable spiculation features (Choi et al., CMPB 2021 §2.3)
# =====================================================================

@dataclass(frozen=True)
class SpiculationFeatures:
    """Choi et al. (CMPB 2021) interpretable spiculation & surface morphology feature set."""
    Np: int            # number of valid surface peaks (above noise threshold)
    Na: int            # number of acute spiculations (solid_angle <= 0.65 sr)
    Nl: int            # number of broader lobulations (solid_angle > 0.65 sr)
    Na_att: int        # number of attached peaks
    s1: float          # area-distortion sharpness score = Σ_i (mean Δarea_i × height_i) / Σ_i height_i
    s2: float          # area-distortion irregularity score = Σ_i (var Δarea_i × height_i) / Σ_i height_i
    spic_a: float = 0.0       # cumulative solid-angle-weighted height = Σ_i exp(-ω_i) × height1_i
    spic_b: float = 0.0       # normalized cosine-solid-angle height = Σ_i cos(ω_i) × height1_i / Σ_i height1_i
    roundness: float = 0.0    # isoperimetric mesh roundness 4π r_eq^2 / Area
    classes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {
            "Np": float(self.Np),
            "Na": float(self.Na),
            "Nl": float(self.Nl),
            "Na_att": float(self.Na_att),
            "s1": float(self.s1),
            "s2": float(self.s2),
            "spic_a": float(self.spic_a),
            "spic_b": float(self.spic_b),
            "roundness": float(self.roundness),
        }


def spiculation_features(
    peaks: list[Peak],
    cfg: ClassifyConfig | None = None,
    attachment_mask: np.ndarray | None = None,
    roundness: float = 0.0,
) -> SpiculationFeatures:
    """Compute the Choi et al. (*CMPB* 2021) interpretable spiculation feature set.

    Matches `spiculation_quantification.m` lines 409-426 and `extract_spiculation_features.m`:
    - Noise peaks (`cls == 'small'`, i.e., `height <= 3.0 mm` or `width <= 1.5 mm`) are
      removed prior to counting `num_spikes` (`Np`), `Na`, `Nl`, `Na_att`, `s1`, `s2`,
      `spic_a`, and `spic_b`, while `classes` retains per-peak labels for every peak in
      `peaks` if `peaks` contains pre-filtered or unfiltered entries.
    """
    classes = [classify_peak(p, cfg, attachment_mask) for p in peaks]
    valid_pairs = [(p, c) for p, c in zip(peaks, classes) if c != "small"]
    valid_peaks = [p for p, _ in valid_pairs]

    n_spic = sum(1 for _, c in valid_pairs if c == "spiculation")
    n_lob = sum(1 for _, c in valid_pairs if c == "lobulation")
    n_att = sum(1 for _, c in valid_pairs if c == "attachment")

    if valid_peaks:
        h_total = sum(p.height for p in valid_peaks) + 1e-12
        h1_total = sum((p.height1 if p.height1 > 0 else p.height) for p in valid_peaks) + 1e-12
        s1 = sum(p.mean_distortion * p.height for p in valid_peaks) / h_total
        s2 = sum(p.var_distortion * p.height for p in valid_peaks) / h_total
        spic_a = sum(
            float(np.exp(-p.solid_angle)) * (p.height1 if p.height1 > 0 else p.height)
            for p in valid_peaks
        )
        spic_b = (
            sum(
                float(np.cos(p.solid_angle)) * (p.height1 if p.height1 > 0 else p.height)
                for p in valid_peaks
            )
            / h1_total
        )
    else:
        s1 = 0.0
        s2 = 0.0
        spic_a = 0.0
        spic_b = 0.0

    return SpiculationFeatures(
        Np=n_spic + n_lob + n_att,
        Na=n_spic,
        Nl=n_lob,
        Na_att=n_att,
        s1=float(s1),
        s2=float(s2),
        spic_a=float(spic_a),
        spic_b=float(spic_b),
        roundness=float(roundness),
        classes=classes,
    )


# =====================================================================
# End-to-end voxel → features pipeline
# =====================================================================

def spiculation_from_voxel(
    mask: np.ndarray,
    attachment_mask: np.ndarray | None = None,
    n_param_iter: int = 12,
    cfg: ClassifyConfig | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sphere_kwargs: dict | None = None,
) -> tuple[SpiculationFeatures, list[Peak], np.ndarray, Mesh]:
    """One-shot pipeline: 3D binary voxel mask → conformal spherical parameterization → spiculation features.

    Follows `convert_for_sphere_param.m` + `main_spherical_param.m` + `spiculation_quantification.m`:
      1. Extracts largest 3D genus-0 connected component at `level = 0.4` with
         `smoothpatch(s, 1, 1)` inverse-distance surface fairing (`smooth_iter=1`).
      2. Computes the conformal spherical parameterization (`spherical_parameterization`).
      3. Computes log area distortion (`area_distortion`) and Gaussian Z-score
         area distortion (`nd = zscore_area_distortion`).
      4. Detects and measures spikes (`detect_peaks`), filters noise peaks
         (`height <= 3.0 mm` or `width <= 1.5 mm` in `spiculation_quantification.m` L409),
         and returns `SpiculationFeatures`, the filtered `peaks`, `nd`, and `mesh`.

    `sphere_kwargs` is passed through to `spherical_parameterization`
    (e.g. `{"mobius_center": True, "yamabe_refine": 5}`); default None.
    """
    cfg = cfg or ClassifyConfig()
    mesh = voxel_to_mesh(
        mask.astype(np.float32),
        level=0.4,
        spacing=spacing,
        fill_holes_and_largest_cc=True,
        smooth_iter=1,
        smooth_lambda=1.0,
    )
    sphere = spherical_parameterization(mesh, n_iter=n_param_iter, **(sphere_kwargs or {}))
    l_ard = area_distortion(mesh, sphere)
    nd = zscore_area_distortion(mesh, sphere)
    raw_peaks = detect_peaks(mesh, l_ard, zscore_normalize=True, filter_concave=True)

    vertex_attach = None
    if attachment_mask is not None:
        v_idx = np.round(mesh.vertices / np.asarray(spacing)).astype(int)
        Z, Y, X = attachment_mask.shape
        v_idx[:, 0] = np.clip(v_idx[:, 0], 0, Z - 1)
        v_idx[:, 1] = np.clip(v_idx[:, 1], 0, Y - 1)
        v_idx[:, 2] = np.clip(v_idx[:, 2], 0, X - 1)
        vertex_attach = attachment_mask[v_idx[:, 0], v_idx[:, 1], v_idx[:, 2]].astype(bool)

    # Filter out small noise spikes (`spiculation_quantification.m` L409: `l_height > th_noise & l_width > th_noise/2`)
    valid_peaks = [
        p for p in raw_peaks
        if classify_peak(p, cfg, vertex_attach) != "small"
    ]
    rnd = mesh_roundness(mesh)
    feats = spiculation_features(valid_peaks, cfg, vertex_attach, roundness=rnd)
    return feats, valid_peaks, nd, mesh


# =====================================================================
# Bridge: spiculation peaks → mesh-domain surface-lets
# =====================================================================

def peaks_to_surfacelets(
    mesh: Mesh,
    peaks: list[Peak],
    classes: Iterable[str] | None = None,
):
    """Each peak apex becomes a mesh-domain surface-let atom."""
    try:
        from qradiomics_private.shape.surfacelet import Surfacelet  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "peaks_to_surfacelets requires the qradiomics_private overlay "
            "(provides qradiomics_private.shape.surfacelet.Surfacelet). "
            "Install the dev repo or check `qr info`."
        ) from e
    classes = list(classes) if classes is not None else ["other"] * len(peaks)
    vn = vertex_normals(mesh)
    out: list[Surfacelet] = []
    for peak, cls in zip(peaks, classes):
        v = peak.apex
        pos = mesh.vertices[v]
        n = vn[v]
        out.append(
            Surfacelet(
                z=int(round(float(pos[0]))),
                y=int(round(float(pos[1]))),
                x=int(round(float(pos[2]))),
                nz=float(n[0]),
                ny=float(n[1]),
                nx=float(n[2]),
                saliency=float(-peak.mean_distortion),
                scale=float(peak.height),
                cls=cls,
            )
        )
    return out
