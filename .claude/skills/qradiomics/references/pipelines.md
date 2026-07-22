# Pipelines, patterns, and workflow templates

Two families of pipelines live in this repo:

1. **Production TCIA pipelines** (`pipelines/lung1`, `nsclc_cetuximab`, `lidc_idri`,
   `acrin_heart`) — one-shot **TCIA → trained ML model** bundles, each scaffolded from a
   `qr workflow plan`.
2. **The LIDC-IDRI / LUNGx reproducibility harness** (`pipelines/lidc_idri/`) — 13
   scripts that quantitatively reproduce four Choi Lab papers and provide a drop-in
   benchmark for new methods.

`reports/reproducibility.md` is the canonical protocol + results document; read it
before touching reproducibility numbers. `reports/PUBLIC_DATASET_FRAMEWORK.md` defines
the public-dataset loader contract — use only the TCIA/Zenodo/GitHub sources it lists.

## Feature-extraction patterns

Patterns are named PyRadiomics setting bundles (`qr extract -p <id>`). They live in
`qradiomics/data/pyradiomics/*.yaml` with higher-level templates in
`qradiomics/data/templates/*.yaml`. List them with `qr pattern list`.

| Pattern id | Use |
|---|---|
| `ct-default` | Full 1409-feature CT extraction (the usual default) |
| `ct-original-only` | Original image only, no wavelet/LoG filters (faster) |
| `nsclc-ct` | NSCLC CT survival tuning |
| `pet-default` | PET/SUV extraction |

## Workflow templates

`qr workflow templates` lists them; `qr workflow plan -t <name>` instantiates. The
library (`qradiomics/workflow.py` → `LIBRARY`):

| Template | Starts from | Needs |
|---|---|---|
| `nrrd_survival` | pre-converted NRRD cohort | `-d <cohort>` `-c clinical.csv` |
| `dicom_survival` | DICOM cohort (converts first) | `-d <cohort>` `-c clinical.csv` `--roi` |
| `dicom_to_ml` | DICOM cohort → full ML | `-d <cohort>` `-c clinical.csv` |
| `tcia_to_ml` | TCIA collection → full ML | `-C <collection>` `-c clinical.csv` |

A plan is a JSON/YAML list of ~10 atomic `qr` steps. `qr workflow scaffold` renders it
to a runnable file per executor: **nextflow** (`main.nf`, default — per-patient parallel,
cache, HPC via `nextflow.config` local/docker/slurm profiles), **prefect**
(`prefect_flow.py`), or **shell**. `qr workflow run --executor inline` is the
small-cohort fallback that runs steps in-process.

## The LIDC-IDRI / LUNGx reproducibility harness

One script per task under `pipelines/lidc_idri/`:

| Script | Purpose |
|---|---|
| `extract_features.py` | Per-(reader, nodule) PyRadiomics 1409 + spiculation on LIDC XML masks |
| `extract_lungx.py` | LUNGx with intensity-based region-grow masks |
| `extract_cir.py` | LIDC / LUNGx with CIRDataset paper-grade masks |
| `ahsn_proxy.py` | Choi 2014 AHSN — annotated centroids + random lung-tissue negatives |
| `ahsn_hardneg.py` | Choi 2014 AHSN — annotated nodule + XML `nonNodule` hard negatives |
| `reproduce_papers.py` | Med Phys 2018 + CMPB 2021 leakage-safe RF CV |
| `reproduce_cir.py` | CIR LIDC-PM internal + LIDC→LUNGx external (10-pat calibration) |
| `methods_compare.py` | Methods harness (RM / PM / LUNGx-cal / LUNGx-test, CMPB 2021 protocol) |
| `validate_lcsr.py` | Spearman ρ qradiomics vs LCSR bundled reference |
| `mesh_to_voxel_compare.py` | Voxelize mesh peaks; Dice vs LCSR; export OBJ + NRRD |
| `run.sh` | Convenience wrapper for the full pipeline |
| `lidc_pm_ids.txt` | 72 pathology-confirmed LIDC patient IDs (pinned from CIR) |

### Headline numbers (from `AGENTS.md §3`, targets to preserve)

```
spic6 RM AUC        ≈ 0.816 ± 0.006   (paper CMPB 2021: 0.80–0.85)
radiomics+spic PM   ≈ 0.868 ± 0.039   (paper CMPB 2021: 0.85)
LUNGx ext + cal     ≈ 0.756           (paper CMPB 2021: 0.76)
```

A 10-patient LIDC smoke run is ≈ 10 min on a 16-core box; the full 1,018-patient
reproduction is ≈ 6 h (5 h extraction + 1 h modeling).

### Driving data + commands

- Imaging: `qr tcia download --collection LIDC-IDRI --modality CT` and
  `--collection "SPIE-AAPM Lung CT Challenge"` (LUNGx).
- Auxiliary: LIDC XML (`LIDC-XML-only.zip`), LUNGx calibration + test xlsx,
  `CIRDataset_LCSR` (Zenodo 6762573).
- Convert + extract + compare: `qr lidc convert-cohort` → `extract_cir.py` (LIDC, LUNGx)
  → `methods_compare.py --lidc-pm-ids pipelines/lidc_idri/lidc_pm_ids.txt`.

### Adding a method to the benchmark

Register a column-selector in `methods_compare.py` — nothing else changes; the same
RM / PM / LUNGx-cal / LUNGx-test splits and the leakage-safe RF CV apply automatically:

```python
# pipelines/lidc_idri/methods_compare.py
METHODS["my_method"] = lambda df: [c for c in df.columns if c.startswith("my_prefix_")]
```

Add a `tests/test_<module>.py` with a synthetic-data unit test alongside any new
pipeline code (`AGENTS.md §7`).

## `examples/` vs `pipelines/`

`examples/*.sh` are minimal, readable, single-cohort walkthroughs of the plain `qr`
chain (LUNG1, NSCLC-Cetuximab, ACRIN heart, LIDC nodule, IBSI phantom validation) —
start here to learn the flow. `pipelines/*` are the deployable, parallel,
executor-backed bundles for the same cohorts.
