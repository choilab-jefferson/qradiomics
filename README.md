# qradiomics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.9.2-orange.svg)](https://github.com/choilab-jefferson/qradiomics/releases)

## Kick-off

Single backend (`scripts/kickoff.sh`) for both flows. It clones the
repo (if not already), creates a `.venv`, `pip install -e .`, runs
`qr info`, and runs the smoke tests.

**For users — one-liner:**

```bash
curl -sSL https://raw.githubusercontent.com/choilab-jefferson/qradiomics/main/scripts/kickoff.sh | bash
```

**For AI coding agents — drop-in prompt:**

> You are working on qradiomics, a lung-radiomics CLI toolkit at
> `github.com/choilab-jefferson/qradiomics`. Do this:
>
> 1. Run the kick-off (clones, installs, verifies):
>    ```
>    curl -sSL https://raw.githubusercontent.com/choilab-jefferson/qradiomics/main/scripts/kickoff.sh | bash
>    ```
> 2. `cd qradiomics` and read `AGENTS.md` for operations (conventions,
>    layout, don'ts, citations). Do **not** log PHI, do **not** push
>    to remotes, do **not** invent new public dataset URLs — use only
>    the ones documented in AGENTS.md.
> 3. Ask the user what they want to build.

Env knobs accepted by the script: `QR_REPO_URL`, `QR_REPO_DIR`,
`QR_BRANCH`, `QR_PYTHON`, `QR_VENV` (set to `-` to skip the venv),
`QR_SKIP_SMOKE=1` to skip pytest.

**Plain `pip` install (e.g. Colab):**

```bash
pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"
pip install qradiomics
```

Install pyradiomics **first**. `pip install qradiomics` on its own fails to
resolve its `pyradiomics>=3.1.0` requirement: pyradiomics ships no PyPI wheel
for Python 3.10+ and its 3.1.0 sdist has broken metadata (declares `3.0.1a1`),
so pip discards it. This is a pyradiomics packaging limitation, not a
qradiomics one — installing pyradiomics from its upstream git first gives pip a
build that satisfies the requirement.

---

> **Active successor for three earlier Choi Lab radiomics codebases.** The
> C++/MATLAB pipelines in
> [taznux/radiomics-tools](https://github.com/taznux/radiomics-tools),
> [taznux/lung-image-analysis](https://github.com/taznux/lung-image-analysis), and
> [choilab-jefferson/LungCancerScreeningRadiomics](https://github.com/choilab-jefferson/LungCancerScreeningRadiomics)
> are **superseded** by this repo. The feature extractors are now in
> `qradiomics.feature.rtools` (Python ITK port, numerically exact to the
> C++ binary). New work should land here.

**Radiomics research CLI.** `qr` does two things equally well:

1. **Atomic tasks** — convert DICOM, extract features, merge clinical, fit
   a model. Each is a single command, files in / files out.
2. **Workflow assembly** — generate, mutate, scaffold, and run multi-step
   pipelines from those atomic tasks. Default executor is **Nextflow**
   (per-patient parallel + cache + HPC); **Prefect** is the secondary
   executor; `inline` is the small-cohort fallback.

The canonical radiomics data flow has four stages — `data → image →
features → modeling` — and one `qr workflow plan` call instantiates the
whole chain:

```bash
# Atomic tasks
qr convert dicom-series / rtstruct / manifest-from-dir
qr extract        -m manifest.csv -p <pattern> -o features.csv
qr results merge  -f features.csv -c clinical.csv -o analysis_ready.csv
qr analyze {survival,classify,importance} -i analysis_ready.csv ...
qr ml {train,predict,evaluate} ...

# Workflow assembly
qr workflow plan      -t dicom_to_ml -d <cohort> -c <clinical> -o plan.json
qr workflow scaffold  -p plan.json -e nextflow   -o pipeline.nf
qr workflow run       plan.json --executor nextflow   # default
```

## Background — three earlier projects, unified

`qradiomics` is the modern Python successor of three earlier Choi Lab
radiomics codebases. The MATLAB pipelines, the ITK / Ruffus C++ tools,
and the Docker-based screening workflow are distilled here into a single
Click CLI built on PyRadiomics, scikit-learn, and lifelines:

| Earlier project | Stack | Role | This repo |
|---|---|---|---|
| [taznux/lung-image-analysis](https://github.com/taznux/lung-image-analysis) | MATLAB · MIT | LIDC-IDRI nodule detection / segmentation / characterization | superseded |
| [taznux/radiomics-tools](https://github.com/taznux/radiomics-tools) | C++/Python (ITK, Ruffus) · MIT | DICOM tools, GrowCut segmentation, feature extraction pipeline | superseded |
| [choilab-jefferson/LungCancerScreeningRadiomics](https://github.com/choilab-jefferson/LungCancerScreeningRadiomics) | MATLAB / Python · GPL-3.0 | LIDC + LUNGx end-to-end screening workflow with AutoML | superseded (this repo re-implements the open subset under MIT using PyRadiomics) |

The AHSN shape descriptor pipeline (CMPB 2014) and the spiculation
quantification pipeline (CMPB 2021, companion to
[choilab-jefferson/CIR](https://github.com/choilab-jefferson/CIR))
are re-integrated in `qradiomics.shape` (see Shape Analysis below).
The longitudinal CBCT / delta-radiomics workflows (ASTRO / AAPM 2026)
will be released here after publication.

## Install

```bash
pip install -e .            # core CLI + library
pip install -e .[rtstruct]  # plus rt-utils for `qr convert rtstruct`
```

Python 3.11 or newer is required. PyRadiomics, SimpleITK, lifelines,
scikit-learn, statsmodels, scipy, and pandas are pulled in as
dependencies.

After install, `qr`, `qradiomics`, and `qrdx` are available on `$PATH`
and point at `qradiomics.cli.main:cli`.

## DICOM Conversion

Many TCIA cohorts ship as DICOM (CT/PET/MR series + RTSTRUCT). Two
helpers convert into the NRRD form the rest of the pipeline consumes:

```bash
# 1. CT/PET/MR DICOM series → single NRRD volume
qr convert dicom-series \
  -i <dataset_root>/<patient>/<study>/CT/ \
  -o <out>/<patient>_CT.nrrd

# 2. RTSTRUCT contour → binary label NRRD (same geometry as the reference CT)
qr convert rtstruct \
  -d <dataset_root>/<patient>/<study>/CT/ \
  -r <dataset_root>/<patient>/<study>/RTSeries/RS.<uid>.dcm \
  --roi GTV \
  -o <out>/<patient>_GTV-label.nrrd

# 3. (Optional) build a manifest by globbing image/mask pairs in a tree
qr convert manifest-from-dir \
  -d <out>/ \
  --image-glob '*_CT.nrrd' \
  --mask-glob '*-label.nrrd' \
  -o manifest.csv
```

RTSTRUCT conversion uses `rt-utils` (install via
`pip install qradiomics[rtstruct]`). ROI lookup is case-insensitive (a
`--roi Heart` request matches a structure set with `heart`). The mask
is auto-reshaped to the CT geometry, with a ±1-slice z-axis trim/pad
when the structure set references slices outside the series.

## End-to-end Example — TCIA NSCLC-Radiomics (Lung1) from scratch

Starts from nothing — pulls DICOM straight from TCIA, converts, extracts,
joins clinical, and reports the Cox PH ranking. The same chain runs for
any TCIA collection by swapping the first argument.

```bash
# 0. One-time: install + workspace
pip install -e .[rtstruct]
export USER_DATA=/data/$USER          # ≥ 30 GB free for Lung1 (~422 patients)
mkdir -p $USER_DATA/{Lung1,Lung1-out}

# 1. DICOM pull from TCIA  (CT + RTSTRUCT for the same series instance UIDs)
qr tcia download \
  --collection NSCLC-Radiomics --modality CT \
  -o $USER_DATA/Lung1 -j 16
qr tcia download \
  --collection NSCLC-Radiomics --modality RTSTRUCT \
  -o $USER_DATA/Lung1 -j 16

# 2. DICOM → NRRD per patient: CT volume + GTV-1 binary mask
for pat in $USER_DATA/Lung1/*/; do
  pid=$(basename "$pat")
  qr convert dicom-series \
    -i  "$pat"*/CT \
    -o  "$USER_DATA/Lung1-out/${pid}_CT.nrrd"
  qr convert rtstruct \
    -d  "$pat"*/CT \
    -r  "$pat"*/RTSeries/*.dcm \
    --roi GTV-1 \
    -o  "$USER_DATA/Lung1-out/${pid}_GTV-label.nrrd"
done

# 3. Manifest from the converted tree
qr convert manifest-from-dir \
  -d "$USER_DATA/Lung1-out" \
  --image-glob '*_CT.nrrd' \
  --mask-glob  '*_GTV-label.nrrd' \
  -o  "$USER_DATA/Lung1-out/manifest.csv"

# 4. PyRadiomics extraction under the nsclc-survival pattern
#    (Original + LoG + Wavelet + Square + SquareRoot + Logarithm ×
#     {firstorder, shape, glcm, glrlm, glszm, gldm, ngtdm} ≈ 1130 features)
qr extract \
  -m "$USER_DATA/Lung1-out/manifest.csv" \
  -p nsclc-survival \
  -o "$USER_DATA/Lung1-out/features.csv"

# 5. Join with the published clinical table (download once)
#    Header has Survival.time + deadstatus → results merge auto-renames.
curl -sLo "$USER_DATA/Lung1-out/clinical.csv" \
  "https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
qr results merge \
  -f "$USER_DATA/Lung1-out/features.csv" \
  -c "$USER_DATA/Lung1-out/clinical.csv" \
  --clinical-id-col PatientID \
  --time-col Survival.time --event-col deadstatus.event \
  -o "$USER_DATA/Lung1-out/analysis_ready.csv"

# 6. Univariate Cox PH on every radiomic feature → ranked CSV
qr analyze survival \
  -i "$USER_DATA/Lung1-out/analysis_ready.csv" \
  --outcome OS_months --event OS_event \
  -o "$USER_DATA/Lung1-out/cox_results.csv"
```

Expected outcome on Lung1 (≈ 420 patients): `original_ngtdm_Busyness`
ranks at the top (HR ≈ 1.23, p < 1e-4) — replicating the headline
finding from the Aerts 2014 *Nature Communications* radiomics paper.

A 422-patient run takes ≈ 1 h on a 16-core workstation (≈ 30 min
download, 20 min DICOM→NRRD, 15 min extraction, seconds for the rest).
For a 5-patient smoke run on the same chain, see
`scripts/smoke.py` (synthetic NRRD; no TCIA download required).

The exact same sequence is bundled per cohort under `pipelines/lung1/`,
`pipelines/lidc_idri/`, `pipelines/nsclc_cetuximab/`, and
`pipelines/acrin_heart/` with Nextflow / Prefect / inline executors —
see the **Deployable Pipelines** section below.

### Alternative starting point — manifest you already have

If your data is already in NRRD form with a manifest CSV (canonical
lowercase columns: `patient_id, modality, image_path, mask_path`), skip
steps 0-3 and start at step 4 above.

Browse the bundled patterns with `qr pattern list` and `qr pattern search <kw>`.

## Deployable Pipelines

For each TCIA-public cohort, [`pipelines/`](pipelines/) ships a ready-
to-run bundle: `plan.json` + `main.nf` + `prefect_flow.py` +
`nextflow.config` + `deploy.sh`. Run any cohort end-to-end with:

```bash
cd pipelines/lung1/
cp /path/to/your/clinical.csv clinical/clinical.csv
./deploy.sh                       # nextflow (per-patient parallel, default)
EXECUTOR=prefect ./deploy.sh      # via Prefect 2.x
EXECUTOR=inline ./deploy.sh       # sequential subprocess (smoke tests)
```

Available bundles: `lung1/`, `nsclc_cetuximab/`, `lidc_idri/`, `acrin_heart/`.

## Workflow Assembly — agents compose, qr executes

The canonical four-stage data flow is encoded in the **template
library** that `qr workflow plan` draws from:

| Template | Stages covered | When to use |
|---|---|---|
| `nrrd_survival` | data → features → modeling | cohort already in NRRD form |
| `dicom_survival` | data → image → features → modeling | cohort ships as DICOM + RTSTRUCT |
| `dicom_to_ml` | data → image → features → modeling (ML) | full end-to-end DICOM → trained model + CV metrics + held-out evaluation |

```bash
# 1. Generate a plan
qr workflow plan -t dicom_to_ml \
    -d /data/cohort -c clinical.csv \
    --roi GTV --pattern nsclc-survival \
    -o plan.json

# 2. (Optional) scaffold a Nextflow file for inspection / editing
qr workflow scaffold -p plan.json -e nextflow -o pipeline.nf

# 3. Run — default executor is Nextflow (per-patient parallel + cache)
qr workflow run plan.json
qr workflow run plan.json --executor inline      # small interactive
qr workflow run plan.json --executor prefect     # Prefect-orchestrated
```

The plan is plain JSON/YAML — agents can read, mutate (add a stage,
swap a pattern, change the executor), and re-run without re-templating.
Per-patient steps are marked in the plan and fanned out automatically
by the Nextflow / Prefect scaffolders.

## Validated Cohorts

The pipeline has been validated end-to-end on three TCIA public cohorts:

| Cohort | Format on TCIA | Conversion path |
|---|---|---|
| [NSCLC-Radiomics (LUNG1)](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) | DICOM CT + RTSTRUCT (or pre-converted NRRD via the published companion pack) | `qr convert dicom-series` + `qr convert rtstruct --roi GTV-1`, or feed NRRD directly |
| [NSCLC-Cetuximab](https://www.cancerimagingarchive.net/collection/nsclc-cetuximab/) | DICOM CT + RTSTRUCT | `qr convert dicom-series` + `qr convert rtstruct --roi PTV` |
| [ACRIN-NSCLC-FDG-PET](https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/) | DICOM CT/PET + RTSTRUCT | `qr convert dicom-series` + `qr convert rtstruct --roi Heart` (case-insensitive) |

All three flow cleanly through `convert → extract → results merge → analyze`.
Ready-to-run shell scripts for each cohort (plus LIDC-IDRI and the IBSI
phantom) live in [`examples/`](examples/README.md).

## Verified End-to-End Pipelines

Each row is a published study whose protocol has been reproduced end-to-end
in this repo. Pick the row matching your study, run the listed command, and
the result will be within reported tolerance of the paper.

| Pipeline | Paper anchor | Entry point | Verified outcome |
|---|---|---|---|
| **LIDC AHSN nodule detection** | Choi & Choi, *Comput Methods Programs Biomed* 2014;113(1):37–54 ([doi](https://doi.org/10.1016/j.cmpb.2013.08.015)) | `pipelines/lidc_idri/ahsn_proxy.py` · `pipelines/lidc_idri/ahsn_hardneg.py` | 180-D AHSN descriptor on LIDC nodule vs non-nodule |
| **LIDC + LUNGx malignancy radiomics (RM)** | Choi et al., *Med Phys* 2018;45(4):1537–1549 ([doi](https://doi.org/10.1002/mp.12820)) | `pipelines/lidc_idri/reproduce_papers.py` · `pipelines/lidc_idri/methods_compare.py` | LIDC RM AUC ≈ 0.816 ± 0.006 · LUNGx ext+cal ≈ 0.756 (paper: 0.80–0.85 / 0.76) |
| **LUNGx interpretable spiculation (PM, RM+spic)** | Choi et al., *Comput Methods Programs Biomed* 2021;200:105839 ([doi](https://doi.org/10.1016/j.cmpb.2020.105839)) | `pipelines/lidc_idri/methods_compare.py` (PM) · `qradiomics.shape.spiculation_from_voxel` | radiomics+spic PM ≈ 0.868 ± 0.039 (paper: 0.85) |
| **HeartToxicity FDG uptake 3-class (TJU prePT → ACRIN/HeartCB postPT)** | Choi et al., *JCO Clin Cancer Inform* 2024 Appendix C ([doi](https://doi.org/10.1200/CCI.23.00241)) | qradiomics-dev `pipelines/heart_local/` (private cohort extension; protocol reproduced with `qradiomics.feature.rtools` postPT-auto + pyradiomics two-source concat) | TJU prePT → ACRIN/HeartCB postPT external acc ≈ 0.80 / 0.78 (Step 1) — matches paper anchor |

Each entry point writes a JSON/CSV report next to the script; no manual
gluing required.

## Command Reference

| Command                     | Stage      | Purpose                                                       |
|-----------------------------|------------|---------------------------------------------------------------|
| `qr tcia download`          | data       | Bulk-download a TCIA collection (multi-process + progress)    |
| `qr anonymize`              | data       | Strip PHI from a DICOM tree (DICOM PS3.15 Annex E)            |
| `qr convert dicom-series`   | data/image | DICOM CT/MR → NRRD; PT auto-routes through SUV conversion     |
| `qr convert rtstruct`       | data/image | DICOM RTSTRUCT contour → label NRRD (case-insensitive ROI)    |
| `qr convert manifest-from-dir` | data    | Glob image+mask pairs into a manifest CSV                     |
| `qr preprocess`             | image      | bbox crop + isotropic resample per (image, mask) row          |
| `qr register`               | image      | Rigid Mattes-MI/LBFGSB moving→fixed (Scenario C mask transfer)|
| `qr hu-correct`             | image      | Histogram-match CBCT to a reference CT                        |
| `qr extract`                | features   | PyRadiomics → features.csv (manifest + pattern)               |
| `qr shape extract`          | features   | AHSN + spiculation shape descriptors                          |
| `qr delta`                  | features   | DeltaPair (A - B) + trend slope per patient across timepoints |
| `qr results merge`          | features   | features.csv + clinical.csv → analysis_ready.csv              |
| `qr analyze survival`       | modeling   | Univariate Cox proportional hazards                           |
| `qr analyze classify`       | modeling   | Univariate logistic regression                                |
| `qr analyze importance`     | modeling   | Random-forest + permutation (+ optional SHAP)                 |
| `qr ml train`               | modeling   | CV Cox / logistic + leakage-safe corr/univariate selection    |
| `qr ml predict`             | modeling   | Apply a trained model to new features                         |
| `qr ml evaluate`            | modeling   | Hold-out evaluation report (c-index / AUC)                    |
| `qr workflow plan`          | assembly   | Generate a multi-step plan from a template                    |
| `qr workflow show`          | assembly   | Inspect a plan's steps and variables                          |
| `qr workflow scaffold`      | assembly   | Render a plan as shell / nextflow / prefect                   |
| `qr workflow run`           | assembly   | Execute a plan (default executor: nextflow)                   |
| `qr pattern list / search`  | meta       | Browse bundled pattern templates                              |
| `qr config get / set`       | meta       | User preferences in `~/.qradiomics/config.yaml`               |

## Python API — atomic core

Every CLI command is a thin wrapper around a re-usable Python API.
External libraries (e.g. longitudinal CBCT orchestrators) consume the
atomic layer directly instead of shelling out.

```python
from qradiomics.atomic import (
    load_image_and_mask, preprocess_pair,
    build_extractor, run_extractor, extract_features,
    register_pair, resample_to_fixed, histogram_match_hu,
)
from qradiomics.data_model import (
    Cohort, Patient, TreatmentCourse, Study,
    ImageSeries, RTStructureSet, ROI,
    AtomicUnit, Modality, StudyType,
    save_cohort, load_cohort,
)
from qradiomics.manifest import flatten_cohort, read_manifest, write_manifest
from qradiomics.delta import DeltaPair, compute_delta, compute_trend
from qradiomics.io.dicom import read_pet_suv

# Single atomic unit: one image, one mask → ≈1409 features
image, mask = load_image_and_mask("planCT.nrrd", "Heart-label.nrrd")
cropped_img, cropped_msk = preprocess_pair(image, mask, pad_mm=20, resample_mm=1.0)
extractor = build_extractor(image_types=["Original", "LoG", "Wavelet"])
features = run_extractor(extractor, cropped_img, cropped_msk)
```

### Hierarchical cohort model

`qradiomics.data_model` mirrors the canonical 5–6 level hierarchy used
across the Choi-Lab ecosystem:

```
Cohort → Patient → TreatmentCourse → Study → ImageSeries / RTStructureSet → ROI
                       (optional)
```

Diagnostic-only cohorts omit `TreatmentCourse` and attach `Study`
directly to `Patient`. `flatten_cohort()` walks the tree and produces a
list of `AtomicUnit`s — one per (image, mask) pair — which becomes the
manifest CSV consumed by `qr extract`.

```python
cohort = Cohort(cohort_id="lng-cbct")
patient = Patient(patient_id="P001")
course = TreatmentCourse(course_id="rt1", fractions=30, prescription_dose_gy=60.0)
study = Study(study_id="S-week4", timepoint="week4", relative_day=28)
study.series["CBCT"] = ImageSeries(series_id="CBCT-w4",
    image_path="/data/CBCT_w4.nrrd", modality=Modality.CBCT, image_tag="CBCT-w4")
rs = RTStructureSet(rtstruct_id="rs", referenced_series_uid="...")
rs.rois["GTV"] = ROI(roi_id="GTV", mask_path="/data/GTV-label.nrrd",
                     mask_tag="manual", mask_image_tag="CBCT-w4")
study.structure_sets["rs"] = rs
course.studies[study.study_id] = study
patient.treatment_courses[course.course_id] = course
cohort.patients[patient.patient_id] = patient

units = flatten_cohort(cohort)            # list[AtomicUnit]
write_manifest(units, "manifest.csv")     # canonical 10-column schema
save_cohort(cohort, "cohort.yaml")        # full graph persistence
```

The manifest is the bridge: anyone (the CLI, Nextflow, JeffLungRadiomics,
external scripts) can consume it without needing the Python model.

## Shape Analysis — `qradiomics.shape`

Python re-implementations of two published Choi-Lab pipelines, used as
a library (no CLI yet — call as functions):

**2014 CMPB — AHSN pulmonary nodule detection**

```python
from qradiomics.shape import (
    surface_elements,          # Hessian eigendecomp + per-voxel normals (§2.2.1)
    detect_candidates,         # Multi-scale Sato/Li dot enhancement (§2.2.2)
    ahsn, AHSNConfig,          # Angular Histogram of Surface Normals (§2.3.1)
    wall_eliminate,            # Iterative wall detection / elimination (§2.3.2)
    make, make_all,            # Synthetic 3D lung phantoms for testing
)
```

**2021 CMPB — Spiculation quantification** (companion to
[CIR](https://github.com/choilab-jefferson/CIR))

```python
from qradiomics.shape import (
    voxel_to_mesh,                  # marching cubes → triangular mesh
    spherical_parameterization,     # cotangent-Laplacian → unit sphere
    area_distortion,                # per-vertex log-area distortion
    detect_peaks,                   # negative-distortion peaks = spike candidates
    spiculation_features,           # Na / Nl / Na_att / s1 / s2 features
    spiculation_from_voxel,         # one-shot mask → SpiculationFeatures
)
```

See `tests/shape/` for end-to-end usage on analytic shapes
(sphere / spiked-sphere / phantoms).

## Repository Layout

```
qradiomics/
├── __init__.py              # exposes PatternLoader, RadiomicsExtractor, __version__
├── cli/                     # Click CLI (qr / qradiomics / qrdx)
│   ├── main.py
│   ├── config_io.py
│   ├── commands/            # extract, results, analyze, config_cmd
│   └── pattern/             # list, match
├── pattern_loader.py        # YAML pattern templates → Pydantic models
├── extractor.py             # PyRadiomics wrapper
├── shape/                   # Published shape pipelines (re-implementation)
│   ├── hessian.py           # 2014 §2.2.1 — Hessian + surface elements
│   ├── detection.py         # 2014 §2.2.2 — multi-scale Sato/Li dot filter
│   ├── ahsn.py              # 2014 §2.3.1 — AHSN descriptor
│   ├── wall_elim.py         # 2014 §2.3.2 — iterative wall elimination
│   ├── mesh_utils.py        # 2021 — voxel → mesh + geometry primitives
│   ├── spiculation.py       # 2021 — spherical param + Na/Nl/Na_att/s1/s2
│   └── phantoms.py          # Synthetic 3D lung phantoms for testing
└── data/
    ├── templates/           # pattern YAMLs (ct_default, nsclc_survival, ...)
    ├── pyradiomics/         # per-pattern PyRadiomics extractor configs
    └── schema/              # pattern-template JSON schema

tests/                       # pytest: analyze + results.merge (19 tests)
LICENSE                      # MIT
pyproject.toml
```

## Bundled Pattern Templates

| `pattern_id`         | Description                                           |
|----------------------|-------------------------------------------------------|
| `ct-default`         | Plain CT, single timepoint, multi image-type baseline |
| `standard-radiomics` | Multi-modality generic radiomics                      |
| `survival-analysis`  | Cox + RSF + KM, time-to-event task                    |
| `nsclc-survival`     | NSCLC CT GTV, LoG+Wavelet+Square/Sqrt/Log image types |

Drop a new `*.yaml` into `qradiomics/data/templates/` to add a study;
`qr pattern list` picks it up automatically.

## Citing

If you use this CLI in published work, please cite the relevant
upstream papers. PyRadiomics and the NSCLC-Radiomics cohort are the
two essential citations for any qradiomics-derived feature analysis:

- **PyRadiomics** — van Griethuysen JJM, Fedorov A, Parmar C, et al.
  *Computational Radiomics System to Decode the Radiographic
  Phenotype.* Cancer Research 2017; 77(21):e104-e107.
  doi:[10.1158/0008-5472.CAN-17-0339](https://doi.org/10.1158/0008-5472.CAN-17-0339)
- **NSCLC-Radiomics (TCIA LUNG1)** — Aerts HJWL, Velazquez ER, Leijenaar
  RTH, et al. *Decoding tumour phenotype by noninvasive imaging using a
  quantitative radiomics approach.* Nature Communications 2014; 5:4006.
  doi:[10.1038/ncomms5006](https://doi.org/10.1038/ncomms5006)

If you build on the lung-screening lineage that this CLI grew out of,
please additionally cite:

- Choi W, Oh JH, Riyahi S, Liu C-J, Jiang F, Chen W, White C, Rimner A,
  Mechalakos JG, Deasy JO, Lu W. *Radiomics analysis of pulmonary
  nodules in low-dose CT for early detection of lung cancer.* Medical
  Physics 2018; 45(4):1537-1549.
  doi:[10.1002/mp.12820](https://doi.org/10.1002/mp.12820)
- Choi W, Nadeem S, Riyahi S, Deasy JO, Tannenbaum A, Lu W.
  *Reproducible and Interpretable Spiculation Quantification for Lung
  Cancer Screening.* Computer Methods and Programs in Biomedicine 2021;
  200:105839.
  doi:[10.1016/j.cmpb.2020.105839](https://doi.org/10.1016/j.cmpb.2020.105839)
- Choi WJ, Choi TS. *Automated pulmonary nodule detection based on
  three-dimensional shape-based feature descriptor.* Computer Methods
  and Programs in Biomedicine 2014; 113(1):37-54.
  doi:[10.1016/j.cmpb.2013.08.015](https://doi.org/10.1016/j.cmpb.2013.08.015)
- Choi W, Werner-Wasik M, Siglin J, et al. *Heart Radiomics for the
  Early Detection of Cardiac Toxicity in Non–Small-Cell Lung Cancer.*
  JCO Clinical Cancer Informatics 2024; 8:e2300241.
  doi:[10.1200/CCI.23.00241](https://doi.org/10.1200/CCI.23.00241)
  — Appendix C 3-class FDG uptake-pattern protocol (TJU prePT →
  ACRIN/HeartCB postPT external validation) is reproduced in the
  qradiomics-dev private extension on top of the public
  `qradiomics.feature.rtools` postPT-auto extractor.

## Authors and Acknowledgements

- [**Wookjin Choi**](https://github.com/taznux) — overall architecture,
  CLI design, pattern templates
- [**Pradeep Bhetwal**](https://github.com/Pradeepbhetwal) — survival
  analysis on the LUNG1 cohort
- Choi Lab, Department of Radiation Oncology, Sidney Kimmel Medical
  College at Thomas Jefferson University

## License

MIT — see [LICENSE](LICENSE).
