# `qr` CLI reference

Every command group, its key options, and copy-paste recipes. Options shown are the
load-bearing ones; run `qr <group> <cmd> --help` for the complete, authoritative list
(the help strings in `qradiomics/cli/commands/` are the source of truth). Aliases:
`qr`, `qradiomics`, `qrdx` all point to the same CLI. `qr info` prints the version.

## Table of contents

- [convert — DICOM → NRRD, RTSTRUCT → label, build a manifest](#convert)
- [extract — PyRadiomics features from a manifest](#extract)
- [shape — AHSN + spiculation shape descriptors](#shape)
- [delta — longitudinal delta / trend features](#delta)
- [results merge — join features with clinical outcomes](#results-merge)
- [analyze — survival / classify / importance](#analyze)
- [ml — train / predict / evaluate](#ml)
- [pattern — feature-extraction patterns](#pattern)
- [workflow — plan / scaffold / run pipelines](#workflow)
- [tcia — public dataset download](#tcia)
- [lidc — LIDC XML → NRRD](#lidc)
- [pacs — DICOM networking](#pacs)
- [other — preprocess, register, hu-correct, anonymize, config](#other)

---

## convert

`qr convert dicom-series` — one DICOM series directory → one NRRD.
```bash
qr convert dicom-series -i <series_dir> -o CT.nrrd [--modality auto|CT|PT|MR]
# PT: --raw <path> also writes the pre-SUV volume. Modality 'auto' reads the first slice.
```

`qr convert rtstruct` — an RTSTRUCT contour → a binary label NRRD (needs `rt-utils`,
i.e. `pip install qradiomics[rtstruct]`).
```bash
qr convert rtstruct -c <ct_dir> -s <RTSTRUCT.dcm> -r "GTV-1" -o GTV-label.nrrd
# -r/--roi omitted → first ROI in the structure set. -s may be a dir containing the RTSTRUCT.
```

`qr convert manifest-from-dir` — scan a per-patient tree and emit the manifest CSV
that `qr extract`/`qr shape` consume.
```bash
qr convert manifest-from-dir -i <root> -o manifest.csv \
  [--image-glob '*_CT.nrrd'] [--mask-glob '*-label.nrrd'] [--modality CT]
```

`qr convert fix-preamble` — repair a DICOM/RTSTRUCT with a missing/corrupt 128-byte
preamble so it can be read.

**Manifest schema** (the spine of the whole pipeline):
```csv
patient_id,modality,image_path,mask_path
LIDC-IDRI-0001,CT,/data/nrrd/LIDC-IDRI-0001_CT.nrrd,/data/labels/LIDC-IDRI-0001_nodule-label.nrrd
```

## extract

Read a manifest, run PyRadiomics per patient, stream results to `features.csv`
(~1409 features/row). Output is observable in real time (tail / `pipelines/monitor.py`).
```bash
qr extract -m manifest.csv -p ct-default -o features.csv [--jobs N]
```
- `-p/--pattern` picks the extraction settings (see [pattern](#pattern)). `ct-default`
  = full 1409-feature CT set; `pet-default`, `nsclc-ct`, `ct-original-only` also ship.
- `--jobs N` runs N worker processes, one patient each. A crashing worker is isolated
  (see `tests/test_extract_worker_crash.py`) — the run continues and logs the failure.

## shape

Shape-based descriptors from the Choi Lab papers, separate from PyRadiomics.
```bash
qr shape extract -m manifest.csv -o shape_features.csv \
  [--ahsn/--no-ahsn] [--spiculation/--no-spiculation] [--jobs N]
```
- `--ahsn` (default on): 180-dim Angular Histogram of Surface Normals (2014 CMPB).
- `--spiculation` (default on): Na / Nl / Na_att / s1 / s2 spiculation features (2021 CMPB).

## delta

Longitudinal (multi-timepoint) delta and optional trend features; one row per patient.
```bash
qr delta -f features.csv --pair 'delta=post-pre' -o delta.csv \
  [--pairs-file pairs.json] [--with-trend --time-col relative_day]
```
- `--pair 'name=minuend-subtrahend'` is repeatable; or list them in `--pairs-file`.
- `--with-trend` adds a per-feature linear slope vs `--time-col`.
- Input `features.csv` must carry a `patient_id` plus a timepoint column.

## results merge

Join `features.csv` with a clinical CSV into the `analysis_ready.csv` the modeling
commands expect.
```bash
qr results merge -f features.csv -c clinical.csv -o analysis_ready.csv \
  [--id-col patient_id] [--time-col OS_months] [--event-col OS_event]
```
- Survival time is auto-detected as days vs months (median > 100 → days). Omit
  `--event-col` for classification tasks.

## analyze

Statistical reporting; each writes a results CSV and prints a console summary.
```bash
qr analyze survival   -i analysis_ready.csv -o cox.csv [--outcome OS_months --event OS_event --top-n 20]
qr analyze classify   -i analysis_ready.csv --outcome <binary_col> -o clf.csv [--top-n 20]
qr analyze importance -i analysis_ready.csv --outcome <col> [--event <col>] [--method all] -o imp.csv
```
- `survival` fits Cox PH (lifelines). `importance` works for either task — pass
  `--event` to treat `--outcome` as survival time, omit it for classification.

## ml

Cross-validated model wrappers (leakage-safe: feature selection happens inside CV).
```bash
qr ml train -i analysis_ready.csv --task {survival|classify} --outcome <col> \
  --model model.pkl --metrics cv.json \
  [--time-col OS_months] [--folds 5] [--top-features 50] [--corr-threshold 0.95]
qr ml predict  -i features.csv --model model.pkl --task {survival|classify} -o pred.csv
qr ml evaluate -i analysis_ready.csv --model model.pkl --task {survival|classify} \
  --outcome <col> [--time-col OS_months] --report eval.json
```
- `--corr-threshold` drops one of each highly-correlated feature pair; `--top-features`
  caps the retained set. Both feed the leakage-safe selection.

## pattern

Feature-extraction patterns are named PyRadiomics setting bundles (YAML under
`qradiomics/data/pyradiomics/` + templates in `qradiomics/data/templates/`).
```bash
qr pattern list             # all bundled patterns with id / name / tags
qr pattern search <query>   # search by keyword
```
Common ids: `ct-default` (1409 CT features), `pet-default`, `nsclc-ct`,
`ct-original-only`. Pass the id to `qr extract -p`.

## workflow

Turn the atomic tasks into a runnable multi-step pipeline.
```bash
qr workflow templates                 # list templates (nrrd_survival, dicom_survival, dicom_to_ml, tcia_to_ml, ...)
qr workflow plan -t <template> [-d <cohort_dir>] [-c clinical.csv] [-C <TCIA collection>] \
  [--roi GTV] [--pattern nsclc-survival] [--task survival] [--outcome OS_event] \
  [--max-series N] [--outdir runs/cohort] -o plan.json
qr workflow show plan.json            # inspect the plan
qr workflow scaffold -p plan.json -e {shell|nextflow} -o pipeline.nf
qr workflow run plan.json [--executor nextflow|prefect|inline] [--dry-run]
```
- Executor default is **nextflow** (per-patient parallel + cache + HPC). **prefect** is
  secondary; **inline** is the small-cohort fallback. `--dry-run` prints commands only.
- `-d` is required for `nrrd_survival`/`dicom_survival`/`dicom_to_ml`; `-C` for `tcia_to_ml`;
  `-c` for any survival/classify analysis.

## tcia

Download and inspect public TCIA collections (only documented collections — don't invent URLs).
```bash
qr tcia collections
qr tcia series -c "LIDC-IDRI" [-m CT -m RTSTRUCT] [-p <PatientID>] [-o series.csv]
qr tcia download -c "LIDC-IDRI" [--modality CT] [-o <raw_dir>]
qr tcia clinical -c <collection> -o clinical.csv
```

## lidc

LIDC-IDRI ships nodule annotations as per-reader XML, not RTSTRUCT. These voxelize it.
```bash
qr lidc convert -d <ct_series_dir> [--xml annotation.xml] -o <out_dir> --pid <patient_id>
qr lidc convert-cohort --src <LIDC_root> --out <out_dir> [--limit N] [--jobs 4]
```
Output per patient: CT NRRD + mask NRRDs + `nodules.csv`. For a single consensus mask
across the four readers, use STAPLE (`qradiomics.io.lidc.staple_patient`).

## pacs

DICOM-network (C-ECHO/C-FIND/C-MOVE/C-STORE) operations against a PACS. Configure
profiles in a YAML like `qradiomics-pacs.example.yaml`.
```bash
qr pacs profiles                          # list configured PACS nodes
qr pacs ping <profile>                    # C-ECHO connectivity test
qr pacs query studies|series|instances <profile> [filters]
qr pacs fetch|retrieve <profile> ...      # C-MOVE / C-GET a study or series
qr pacs send <profile> <path>             # C-STORE
qr pacs watch <profile>                   # watch for incoming studies
```

## other

- `qr preprocess` — resample / normalize an image+mask pair before extraction.
- `qr register` — register a moving image+mask onto a fixed image (`register_pair`).
- `qr hu-correct` — histogram-match HU across scanners (`histogram_match_hu`).
- `qr anonymize` — strip identifying DICOM tags. Confirm no PHI leaks downstream.
- `qr config` — show / manage CLI config (`qradiomics.yaml`, PACS profiles, etc.).
