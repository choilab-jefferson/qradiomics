"""Regenerate cmcf_baseline.npz (default spherical_parameterization output).

Run only from a commit whose default path is the validated baseline.
"""
from pathlib import Path

import numpy as np

from qradiomics.shape import spherical_parameterization, voxel_to_mesh


def ellipsoid_mask(size: int = 25) -> np.ndarray:
    c = (size - 1) / 2
    z, y, x = np.indices((size, size, size), dtype=np.float32)
    rho = ((z - c) / 9.0) ** 2 + ((y - c) / 6.0) ** 2 + ((x - c) / 4.5) ** 2
    return (rho <= 1.0).astype(np.float32)


if __name__ == "__main__":
    mesh = voxel_to_mesh(ellipsoid_mask())
    sph = spherical_parameterization(mesh)
    out = Path(__file__).with_name("cmcf_baseline.npz")
    np.savez_compressed(out, vertices=mesh.vertices, faces=mesh.faces, sphere=sph)
    print(out, mesh.n_vertices, mesh.n_faces)
