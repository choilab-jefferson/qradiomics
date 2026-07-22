# qradiomics Python API

The atomic core and shape descriptors, for programmatic use or when writing a new
pipeline step. Prefer these over reimplementing I/O or extraction. Import as
`from qradiomics.x import Y` (not `import qradiomics.x` aliases). Signatures below are
condensed — read the module docstrings for the full parameter set.

## Table of contents

- [qradiomics.atomic — I/O, extraction, preprocessing](#qradiomicsatomic)
- [qradiomics.io — DICOM / LIDC / PACS](#qradiomicsio)
- [qradiomics.shape — AHSN, spiculation, detection](#qradiomicsshape)
- [Testing with phantoms (no patient data)](#testing-with-phantoms)

---

## qradiomics.atomic

The core file-in / features-out layer. `__all__`: `load_image_and_mask`,
`check_geometry`, `extract_features`, `preprocess_pair`, `histogram_match_hu`,
`register_pair`, `resample_to_fixed`.

```python
from qradiomics.atomic import extract_features, load_image_and_mask

# Load an image+mask NRRD pair as SimpleITK images. By default it enforces
# compatible geometry; pass require_compatible_geometry=False to tolerate small
# mismatches (this is what `qr extract` does under --jobs).
image, mask = load_image_and_mask(
    "CT.nrrd", "GTV-label.nrrd",
    require_compatible_geometry=True,   # spacing_tolerance / origin_tolerance also tunable
)

# Run PyRadiomics. Returns a dict of ~1409 features (diagnostics_* stripped).
# params_file is a PyRadiomics YAML (a pattern); None uses the built-in default.
# label selects which mask value to extract; geometry_tolerance relaxes the check.
features = extract_features(image, mask, params_file=None, label=1)
```

Supporting helpers:
- `check_geometry(image, mask, ...)` — validate that an image/mask pair line up.
- `preprocess_pair(image, mask, ...)` — resample / normalize before extraction.
- `histogram_match_hu(...)` — cross-scanner HU harmonization (behind `qr hu-correct`).
- `register_pair(...)`, `resample_to_fixed(...)` — register a moving pair onto a fixed
  image (behind `qr register`).

**Always** do image array I/O through this layer or SimpleITK `ReadImage`/`WriteImage`.
Do not pull pixel arrays out of pydicom — pydicom is for tags, not image geometry.

## qradiomics.io

`qradiomics.io` exposes the `dicom` and `pacs` subpackages; `lidc` is importable as a
subpackage too.

```python
from qradiomics.io.dicom import load_dicom_series          # DICOM series dir → SimpleITK image
from qradiomics.io.lidc import (                            # LIDC-IDRI XML handling
    parse_lidc_xml,          # XML → list[LIDCReader] (per-reader nodule annotations)
    convert_patient,         # DICOM series + XML → CT NRRD + per-reader mask NRRDs
    scan_lidc_dir,           # walk a TCIA-layout LIDC tree → (pid, ct_dir, xml)
    staple_consensus,        # list[sitk.Image] → single STAPLE consensus mask
    staple_patient,          # patient dir → consensus mask across the 4 readers
)
```

LIDC-IDRI ships nodule contours as per-reader XML rather than RTSTRUCT, which is why
it needs this dedicated path (and the `qr lidc` command group). A single binary label
usually comes from a STAPLE consensus of the four readers.

## qradiomics.shape

Published 3D shape descriptors from Choi 2014 (AHSN) and Choi 2021 (spiculation). This
is the public subset; some research extensions are intentionally not shipped.

**Spiculation (Choi 2021)** — the most common entry point is the one-shot voxel pipeline:

```python
from qradiomics.shape import spiculation_from_voxel

# mask: (Z, Y, X) binary nodule mask (numpy). spacing in mm.
features, peaks, distortion, mesh = spiculation_from_voxel(
    mask, spacing=(0.7, 0.7, 0.7),
    attachment_mask=None,      # optional voxel-domain vessel/wall attachment
    n_param_iter=200,          # spherical-parameterization iterations
)
# features: SpiculationFeatures(Np, Na, Nl, Na_att, s1, s2)
#   Np=all peaks, Na=spiculations, Nl=lobulations, Na_att=attached peaks,
#   s1=sharpness, s2=irregularity. features.as_dict() → flat {name: float}.
```

Lower-level pieces if you need to build the pipeline yourself: `voxel_to_mesh`,
`spherical_parameterization`, `area_distortion`, `detect_peaks`, `classify_peak`,
`spiculation_features`, `ClassifyConfig`, `Peak`.

**AHSN (Choi 2014)** — 180-dim Angular Histogram of Surface Normals on a cropped ROI:

```python
from qradiomics.shape import ahsn, AHSNConfig

descriptor = ahsn(block, mask=None, cfg=AHSNConfig())   # block: 3D ROI numpy array
```

**Detection (Choi 2014)** — multi-scale dot/blob enhancement for candidate nodules:

```python
from qradiomics.shape import detect_candidates       # → list[Candidate]
from qradiomics.shape import hessian_3d, surface_elements, make_scales, dot_value
```

## Testing with phantoms

The shape and atomic tests run on synthetic 3D phantoms, so they need no real patient
data — follow this for any new test:

```python
from qradiomics.shape import make, make_all, ALL_PHANTOMS, Phantom
# make("sphere") / make_all() → labeled synthetic nodules with known geometry.
```

See `tests/shape/test_phantoms.py`, `tests/shape/test_spiculation.py`, and
`tests/test_atomic.py` for the patterns. Every new pipeline module should ship a
`tests/test_<module>.py` with at least one synthetic-data unit test, per `AGENTS.md §7`.
