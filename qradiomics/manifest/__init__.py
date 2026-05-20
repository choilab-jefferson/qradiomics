"""qradiomics.manifest — flat CSV manifest of atomic units.

A manifest is a tabular cross-product of (Patient, [Course], Study,
ImageSeries, ROI) — one row per atomic feature-extraction unit. It is
the format that workflow runners consume: each row is a self-contained
job description (image path + mask path + tags) that the atomic layer
turns into a feature vector.

See [[wiki/architecture/ATOMIC_UNIT_AND_STAGES]] §1.1 for the formal
hierarchy and §1.2 for the manifest schema.
"""

from .io import flatten_cohort, read_manifest, write_manifest, MANIFEST_COLUMNS

__all__ = [
    "MANIFEST_COLUMNS",
    "flatten_cohort",
    "read_manifest",
    "write_manifest",
]
