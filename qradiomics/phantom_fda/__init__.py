"""qradiomics.phantom_fda — Phantom-FDA quantification benchmark.

Public ground-truth manifest for the TCIA Phantom-FDA dataset (manufacturer
specs + precision-weighing measurements, not patient data). Used to measure
bias / variance / robustness of shape descriptors, both the public ones in
:mod:`qradiomics.shape` and the private ones in
:mod:`qradiomics_private.shape` (SNoH, Surfacelet, voxel-spiculation).

Wiki spec: ``wiki/architecture/PHANTOM_FDA_QUANTIFICATION.md`` (M0..M5).
Dataset reference: ``wiki/data/PHANTOM_FDA.md``.

Modules:

* :mod:`.layouts`       — per-layout ground-truth manifest (M0).
* (future) loader       — TCIA fetch → NRRD + per-nodule sub-volume (M1).
* (future) quantify     — descriptor extract → GT match → metrics (M2/M3).
* (future) metrics      — ICC, CCC, shape-confusion, Surfacelet Jaccard.
"""

from __future__ import annotations

from .layouts import LAYOUT_4, Layout, Nodule

__all__ = ["LAYOUT_4", "Layout", "Nodule"]
