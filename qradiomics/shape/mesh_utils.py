"""Mesh utilities — voxel → triangular surface mesh + differential geometry.

Implements the mesh pre-processing, inverse-distance surface fairing
(`smoothpatch`), volume/roundness calculation (`stlVolume`), cotangent
Laplace-Beltrami operator, and Fiedler eigenfunction hemisphere bipartition
used by:

1. Nadeem S, Su Z, Zeng W, Kaufman A, Gu X. *Spherical Parameterization
   Balancing Angle and Area Distortions.* IEEE Trans Vis Comput Graph.
   2017;23(6):1663-1676. doi:10.1109/TVCG.2016.2542069 (arXiv:1810.09031v1)
2. Choi W, Nadeem S, Alam SR, Deasy JO, Tannenbaum A, Lu W. *Reproducible
   and interpretable spiculation quantification for lung cancer screening.*
   Comput Methods Programs Biomed. 2021;200:105839.
   doi:10.1016/j.cmpb.2020.105839
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes, label as cc_label
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from skimage.measure import marching_cubes


@dataclass(frozen=True)
class Mesh:
    """Triangular surface mesh."""
    vertices: np.ndarray   # (V, 3) float32
    faces: np.ndarray      # (F, 3) int32 — vertex indices

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def smooth_mesh_inverse_distance(
    mesh: Mesh,
    n_iter: int = 1,
    lam: float = 1.0,
) -> Mesh:
    """Inverse-edge-distance weighted Laplacian smoothing (`smoothpatch(s, 1, 1)`).

    Matches the pre-parameterization surface fairing in `convert_for_sphere_param.m`
    (Dirk-Jan Kroon's `smoothpatch` mode=1, iter=1, lambda=1.0).
    """
    if n_iter <= 0 or mesh.n_vertices == 0 or mesh.n_faces == 0:
        return mesh
    V = mesh.n_vertices
    F = mesh.faces
    e01 = F[:, [0, 1]]
    e12 = F[:, [1, 2]]
    e20 = F[:, [2, 0]]
    edges = np.vstack([e01, e12, e20, e01[:, ::-1], e12[:, ::-1], e20[:, ::-1]])
    edge_keys = edges[:, 0].astype(np.int64) * V + edges[:, 1].astype(np.int64)
    _, uniq_idx = np.unique(edge_keys, return_index=True)
    edges = edges[uniq_idx]

    pos = mesh.vertices.astype(np.float64).copy()
    for _ in range(n_iter):
        d = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1) + 1e-8
        w = 1.0 / d
        W = coo_matrix((w, (edges[:, 0], edges[:, 1])), shape=(V, V)).tocsr()
        row_sum = np.asarray(W.sum(axis=1)).ravel() + 1e-12
        avg = (W @ pos) / row_sum[:, None]
        pos = (1.0 - lam) * pos + lam * avg
    return Mesh(vertices=pos.astype(np.float32), faces=mesh.faces.copy())


def voxel_to_mesh(
    volume: np.ndarray,
    level: float = 0.5,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fill_holes_and_largest_cc: bool = False,
    smooth_iter: int = 0,
    smooth_lambda: float = 1.0,
) -> Mesh:
    """Extract isosurface mesh via marching cubes, with optional genus-0 cleaning
    and `smoothpatch` inverse-distance fairing.

    Args:
        volume: (Z, Y, X) binary or scalar volume.
        level: isolevel for marching cubes (`0.4` in LCSR `convert_for_sphere_param.m`).
        spacing: (dz, dy, dx) voxel spacing in mm.
        fill_holes_and_largest_cc: if True, fills internal 3D cavities and keeps the
            largest 3D connected component before isosurface extraction.
        smooth_iter: number of inverse-edge-distance fairing iterations (`1` in LCSR).
        smooth_lambda: relaxation weight for inverse-distance fairing (`1.0` in LCSR).
    """
    vol = volume.astype(np.float32)
    if fill_holes_and_largest_cc:
        bw = vol > level
        if np.any(bw) and not np.all(bw):
            bw_filled = binary_fill_holes(bw)
            if np.all(bw_filled):
                # Inverted mask (background=1 on all borders, object=0 inside)
                bw_filled = bw
            lbl, n_cc = cc_label(bw_filled)
            if n_cc > 1:
                counts = np.bincount(lbl.ravel())
                counts[0] = 0
                bw_filled = lbl == int(counts.argmax())
            if not np.all(bw_filled) and np.any(bw_filled):
                vol = bw_filled.astype(np.float32)

    verts, faces, _, _ = marching_cubes(vol, level=level, spacing=spacing)
    mesh = Mesh(vertices=verts.astype(np.float32), faces=faces.astype(np.int32))
    if smooth_iter > 0:
        mesh = smooth_mesh_inverse_distance(mesh, n_iter=smooth_iter, lam=smooth_lambda)
    return mesh


# ---- Basic geometric quantities -------------------------------------------

def face_areas(mesh: Mesh) -> np.ndarray:
    """(F,) triangle area."""
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).astype(np.float32)


def vertex_areas(mesh: Mesh) -> np.ndarray:
    """(V,) per-vertex area = sum of (1/3) × incident-face areas."""
    af = face_areas(mesh)
    V = mesh.n_vertices
    out = np.zeros(V, dtype=np.float32)
    for i in range(3):
        np.add.at(out, mesh.faces[:, i], af / 3.0)
    return out


def face_normals(mesh: Mesh, normalize: bool = True) -> np.ndarray:
    """(F, 3) face normals."""
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    if normalize:
        norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        n = n / norm
    return n.astype(np.float32)


def vertex_normals(mesh: Mesh) -> np.ndarray:
    """(V, 3) area-weighted outward vertex normals."""
    fn = face_normals(mesh, normalize=False).astype(np.float64)
    V = mesh.n_vertices
    out = np.zeros((V, 3), dtype=np.float64)
    for i in range(3):
        for k in range(3):
            np.add.at(out[:, k], mesh.faces[:, i], fn[:, k])
    norm = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    vn = out / norm
    cent = mesh.vertices.astype(np.float64).mean(axis=0, keepdims=True)
    if np.mean(np.sum(vn * (mesh.vertices.astype(np.float64) - cent), axis=1)) < 0:
        vn = -vn
    return vn.astype(np.float32)


def mesh_volume_and_area(mesh: Mesh) -> tuple[float, float]:
    """Signed tetrahedral divergence-theorem volume and surface area (`stlVolume`).

    Matches `spiculation_quantification.m` line 313: `[volume, area] = stlVolume(...)`.
    """
    verts = mesh.vertices.astype(np.float64)
    verts_centered = verts - verts.mean(axis=0, keepdims=True)
    v0 = verts_centered[mesh.faces[:, 0]]
    v1 = verts_centered[mesh.faces[:, 1]]
    v2 = verts_centered[mesh.faces[:, 2]]
    cross_12 = np.cross(v1, v2)
    vol = abs(float(np.sum(np.einsum("ij,ij->i", v0, cross_12)) / 6.0))
    area = float(face_areas(mesh).sum())
    return vol, area


def mesh_roundness(mesh: Mesh) -> float:
    """Isoperimetric roundness `min((4 * pi * esr^2) / area, 1)` (`spiculation_quantification.m` L326)."""
    vol, area = mesh_volume_and_area(mesh)
    if area <= 1e-12 or vol <= 1e-12:
        return 0.0
    esr = ((3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)) * (vol ** (1.0 / 3.0))
    return float(min((4.0 * np.pi * (esr ** 2)) / area, 1.0))


# ---- Cotangent Laplacian & Laplace-Beltrami Eigen-Bipartition -------------

def cotangent_laplacian(mesh: Mesh, clamp_min: float | None = None) -> csr_matrix:
    """Discrete cotangent-weighted Laplace-Beltrami operator (V × V sparse).

    Standard cotangent weights (e.g. Pinkall-Polthier 1993; Nadeem 2017 Eq. 2):
        w_ij = 0.5 * (cot alpha_ij + cot beta_ij)
        L[i, j] = -w_ij  for edge (i, j)
        L[i, i] = sum_{j != i} w_ij
    """
    V = mesh.n_vertices
    F = mesh.faces
    verts = mesh.vertices.astype(np.float64)
    rows, cols, data = [], [], []

    for k in range(3):
        i = F[:, (k + 1) % 3]
        j = F[:, (k + 2) % 3]
        o = F[:, k]  # opposite vertex
        e1 = verts[i] - verts[o]
        e2 = verts[j] - verts[o]
        dot = np.einsum("fk,fk->f", e1, e2)
        crs = np.cross(e1, e2)
        cot = 0.5 * dot / (np.linalg.norm(crs, axis=1) + 1e-12)
        if clamp_min is not None:
            cot = np.maximum(cot, clamp_min)
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([-cot, -cot])
    rows_arr = np.concatenate(rows)
    cols_arr = np.concatenate(cols)
    data_arr = np.concatenate(data)
    L = coo_matrix((data_arr, (rows_arr, cols_arr)), shape=(V, V)).tocsr()
    diag = -np.asarray(L.sum(axis=1)).ravel()
    L = L + diags(diag, 0, shape=(V, V), format="csr")
    return L


def laplace_beltrami_bipartition(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Nadeem et al. (2017) Algorithm 2 Step 1: First non-trivial eigenfunction
    `f_0` (Fiedler vector) of the cotangent Laplace-Beltrami operator `L f = lambda M f`.

    The zero-level set `{f_0 = 0}` defines the closed equatorial cutting loop `gamma`
    that partitions the genus-0 mesh into two hemisphere disks `M_1 = {f_0 >= 0}`
    and `M_2 = {f_0 < 0}`.

    Returns:
        f0: (V,) float64 first non-trivial eigenfunction values.
        hemisphere_labels: (V,) int32 hemisphere assignment (`0` for `f_0 >= 0`, `1` for `f_0 < 0`).
    """
    V = mesh.n_vertices
    L = cotangent_laplacian(mesh, clamp_min=1e-4)
    va = np.maximum(vertex_areas(mesh).astype(np.float64), 1e-8)
    M = diags(va, 0, shape=(V, V), format="csr")
    try:
        # Smallest 2 algebraic eigenvalues of generalized SPD problem L f = lambda M f
        _, vecs = eigsh(L, k=2, M=M, sigma=-1e-5, which="LM")
        f0 = np.real(vecs[:, 1]).astype(np.float64)
    except Exception:
        # Fallback to coordinate axis along principal eigenvector of vertices
        pts = mesh.vertices.astype(np.float64) - mesh.vertices.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(pts, full_matrices=False)
        f0 = pts @ vh[0]
    labels = (f0 < 0.0).astype(np.int32)
    return f0, labels


# ---- Vertex 1-ring neighbors ---------------------------------------------

def vertex_adjacency(mesh: Mesh) -> list[list[int]]:
    """Return per-vertex list of neighbor vertex indices."""
    V = mesh.n_vertices
    adj: list[set[int]] = [set() for _ in range(V)]
    for f in mesh.faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return [sorted(s) for s in adj]
