# `pipelines/` — production end-to-end TCIA pipelines

Self-contained deployable bundles for public TCIA cohorts. Each
subdirectory is a one-shot **TCIA → trained ML model** pipeline:

```
tcia download → DICOM→NRRD → manifest → extract → merge → ml train + evaluate
```

| Pipeline | Cohort (TCIA) | Default task | Default ROI |
|---|---|---|---|
| `lung1/` | NSCLC-Radiomics (LUNG1, Aerts 2014) | survival (Cox PH, OS) | GTV-1 |
| `nsclc_cetuximab/` | NSCLC-Cetuximab (RTOG-0617) | survival | PTV |
| `lidc_idri/` | LIDC-IDRI | nodule features (no clinical CSV by default) | nodule label |
| `acrin_heart/` | ACRIN-NSCLC-FDG-PET | classify (heart-ROI) | Heart |

## Contents of each pipeline directory

```
<cohort>/
├── README.md           — what this pipeline does + how to run it
├── plan.json           — qr workflow plan (10 atomic steps)
├── main.nf             — Nextflow workflow (scaffolded from plan.json)
├── nextflow.config     — executor profiles (local / docker / slurm)
├── prefect_flow.py     — Prefect 2.x flow (scaffolded from plan.json)
├── deploy.sh           — one-command deploy: install deps, fetch data, run
└── clinical/           — clinical CSV stub (replace with your file)
```

## Quick start

```bash
# Install qradiomics
pip install qradiomics[rtstruct]
# Optional: Prefect for orchestration
pip install qradiomics[prefect]

# Pick a pipeline
cd lung1/
./deploy.sh
```

`deploy.sh` runs (in order):
1. Install / upgrade `qradiomics`
2. Verify `nextflow` is on PATH (downloads it locally otherwise)
3. Run the pipeline via `nextflow run main.nf -profile local`

## Customising

The `plan.json` is the single source of truth. To change ROI, pattern,
analysis task, or output directory: edit `plan.json` (or regenerate it
via `qr workflow plan -t tcia_to_ml --collection ... -o plan.json`),
then re-scaffold the Nextflow + Prefect files:

```bash
qr workflow scaffold -p plan.json -e nextflow -o main.nf
qr workflow scaffold -p plan.json -e prefect  -o prefect_flow.py
```

## Full-cohort runner — `run_cohort.sh`

For unattended bulk processing (one whole cohort end-to-end with
parallelism + caching), use the **hierarchical runner**:

```bash
# Run a single cohort end-to-end (idempotent)
OUT=runs/lung1 \
CLINICAL=lung1/clinical/clinical.csv \
N_PARALLEL=8 \
./run_cohort.sh NSCLC-Radiomics GTV-1 nsclc-survival survival OS_event
```

### Hierarchical stages (each independently re-runnable, outputs cached)

```
0. catalog   tcia series       → <OUT>/{ct,rt}_series.csv
1. fetch     tcia download (CT + RT for SAME patients)
                                → <OUT>/dicom/<patient>/<study>/<series>/
2. series    DICOM → NRRD per-series  (parallel, idempotent per series UID)
                                → <OUT>/series/<patient>/<patient>__<sUID>__{CT,<ROI>-label}.nrrd
3. patient   pick best CT + matching label
                                → <OUT>/patient/<patient>.json
                                + <OUT>/manifest.csv
4. features  qr extract           → <OUT>/features.csv
5. merge     qr results merge     → <OUT>/analysis_ready.csv
6. modeling  qr ml train + evaluate
                                → <OUT>/{model.pkl, cv_metrics.json, evaluation.json}
```

Re-running skips any stage whose outputs already exist on disk. To
force a single stage to rebuild, delete its outputs (e.g. `rm
<OUT>/features.csv` to re-extract without re-downloading).

### Run every cohort

```bash
# Smoke (default 10 patients per cohort), writes runs/report.md
./test_all.sh

# Production (full collection per cohort)
./run_all.sh

# Subset
./test_all.sh lung1 acrin_heart
```

`test_all.sh` and `run_all.sh` share the same cohort table and
generate a Markdown summary report at `runs/report.md` with CV +
held-out metrics per cohort.

## Local-data mode

For internal validation or air-gapped runs, point `STAGES=2,3,4,5,6`
to skip the catalog + fetch stages and bring your own DICOM tree:

```bash
# Pre-stage the DICOM tree
mkdir -p runs/my_cohort/dicom
rsync -av /local/path/dicom/ runs/my_cohort/dicom/

OUT=runs/my_cohort \
CLINICAL=clinical.csv \
STAGES=2,3,4,5,6 \
./run_cohort.sh my-collection GTV nsclc-survival
```
