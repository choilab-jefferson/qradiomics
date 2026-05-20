"""Mesh utilities — voxel → triangular surface mesh + basic geometry.

Used by `spiculation.py` (2021 CMPB conformal-mapping pipeline).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
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


def voxel_to_mesh(volume: np.ndarray, level: float = 0.5,
                  spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Mesh:
    """Extract isosurface mesh via marching cubes.

    Args:
        volume: (Z, Y, X) binary or scalar volume.
        level: isolevel for marching cubes.
        spacing: (dz, dy, dx) voxel spacing in mm.
    """
    verts, faces, _, _ = marching_cubes(volume.astype(np.float32),
                                        level=level, spacing=spacing)
    return Mesh(vertices=verts.astype(np.float32),
                faces=faces.astype(np.int32))


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
    """(V, 3) area-weighted vertex normals."""
    fn = face_normals(mesh, normalize=False)
    V = mesh.n_vertices
    out = np.zeros((V, 3), dtype=np.float32)
    for i in range(3):
        for k in range(3):
            np.add.at(out[:, k], mesh.faces[:, i], fn[:, k])
    norm = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return (out / norm).astype(np.float32)


# ---- Cotangent Laplacian (for conformal flows) ----------------------------

def cotangent_laplacian(mesh: Mesh) -> csr_matrix:
    """Discrete cotangent-weighted Laplace-Beltrami operator (V × V sparse).

    L[i, j] = -0.5 * (cot α_ij + cot β_ij)  for edge (i,j)
    L[i, i] = -sum_{j ≠ i} L[i, j]
    """
    V = mesh.n_vertices
    F = mesh.faces
    verts = mesh.vertices
    rows, cols, data = [], [], []

    # For each face, for each of the 3 edges, accumulate cotangent weights
    for k in range(3):
        i = F[:, (k + 1) % 3]
        j = F[:, (k + 2) % 3]
        o = F[:, k]  # opposite vertex
        e1 = verts[i] - verts[o]
        e2 = verts[j] - verts[o]
        dot = np.einsum("fk,fk->f", e1, e2)
        crs = np.cross(e1, e2)
        cot = 0.5 * dot / (np.linalg.norm(crs, axis=1) + 1e-12)
        rows.extend([i, j]); cols.extend([j, i]); data.extend([-cot, -cot])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    L = coo_matrix((data, (rows, cols)), shape=(V, V)).tocsr()
    # Diagonal: -sum of row (excluding diagonal)
    diag = -np.asarray(L.sum(axis=1)).ravel()
    L = L + csr_matrix((diag, (np.arange(V), np.arange(V))), shape=(V, V))
    return L


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
