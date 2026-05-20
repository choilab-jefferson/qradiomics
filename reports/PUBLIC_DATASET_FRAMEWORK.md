---
title: "Public Dataset Reproducibility Framework"
version: "1.0"
last_updated: "2026-05-19"
status: "active"
---

# Public Dataset Reproducibility Framework

> **Context**: [[architecture/_context]]
> **Related**: [[architecture/MATLAB_PORT_MAP]] · [[reports/reproducibility_full_2026-05-19]]

qradiomics-public is positioned as a **portable, plug-in benchmark harness** for any lung-nodule (or, more generally, radiomics) study that needs to evaluate a new feature-extraction or classification method against published reference numbers. The same pipeline can ingest data from many public sources without re-architecting.

## Source registry

| Source | Subject of interest | Loader / converter |
| :--- | :--- | :--- |
| **TCIA nbia-toolkit** (LIDC-IDRI, NSCLC-Radiomics, NSCLC-Cetuximab, LUNGx / SPIE-AAPM, Lung-PET-CT-Dx, …) | DICOM CT series + RTSTRUCT, by SeriesInstanceUID | `qr tcia download --collection ... -o ...` |
| **TCIA bundled XML / xlsx** (LIDC-XML-only.zip, LUNGx truth xlsx) | per-series annotations | `pipelines/lidc_idri/extract_features.py` (XML), `pipelines/lidc_idri/extract_lungx.py` (xlsx) |
| **Zenodo** (CIRDataset 6762573) | paper-grade nodule meshes + masks (NRRD per nodule) | `pipelines/lidc_idri/extract_cir.py` |
| **GitHub** (nadeemlab/CIR, choilab-jefferson/CIR, pylidc, lidc2dicom) | reference implementations / annotation lists | one-off ports — see MATLAB_PORT_MAP for the LIDC XML port |
| **Local DICOM trees** (institutional cohorts, e.g. NSCLC-Cetuximab/RTOG-0617 from `/data/users/$USER/...`) | DICOM + RTSTRUCT | `pipelines/<cohort>_local/` |

## Pipeline contract

All loaders / converters emit a **wide features CSV** with the same schema:

```
pid, reader, nodule_id, n_voxels, volume_mm3, malignancy, ...,
status_radiomics, status_spic, [1409 PyRadiomics features], [6 spiculation features]
```

Downstream tools that depend on this schema:

- `reproduce_papers.py` — Med Phys 2018 + CMPB 2021 internal CV
- `reproduce_cir.py` — CIR internal + external validation
- `methods_compare.py` — multi-method comparison on RM / PM / LUNGx-cal / LUNGx-test splits

These tools don't care which source the features came from; the same CSV works.

## Validation-split contract (CMPB 2021 protocol)

The harness pins the four standard splits to fixed identifiers so any future source can reuse them:

| Split | Identifier | Used by |
| :--- | :--- | :--- |
| RM (Radiomic Model training) | LIDC-IDRI 1,018 patients, malignancy ≥ 4 vs ≤ 2 (drop 3) | all RM AUC numbers in the harness |
| PM (Pathology Model calibration) | LIDC-PM 72 patient IDs (pinned at `pipelines/lidc_idri/lidc_pm_ids.txt`, extracted from CIR `dataset/lidc.py:selected`) | all PM AUC numbers |
| LUNGx CalibrationSet | TCIA `CT-Training-*` (10 size-matched nodules) | LUNGx domain-adapted external |
| LUNGx TestSet | TCIA `LUNGx-CT*` (60 nodules / 73 with multi-nodule cases) | external test AUC |

## Provenance & integrity

Each loader emits the source URL / DOI in the features CSV header as a comment, so a downstream consumer can trace any AUC back to the original public data record (TCIA collection URL, Zenodo DOI, GitHub commit hash). Mask provenance matters — the `extract_lungx.py` region-grow masks produce a different AUC than the `extract_cir.py` paper-grade masks, and the harness should report which mask source was used.

## How to add a new dataset

1. Add a `pipelines/<cohort>_loader.py` (or extend an existing one) that yields `(pid, nodule_id, ct_image, mask)` from the new source's native format.
2. Feed those tuples through `qradiomics.atomic.extract_features` + `qradiomics.shape.spiculation_from_voxel` (the same two calls the existing loaders use).
3. Emit a CSV with the standard schema.
4. Plug the CSV into `methods_compare.py` and `reproduce_cir.py` — no harness changes needed.

This keeps the source-specific work confined to the loader; the downstream modelling is identical across datasets.

## References

- [[architecture/MATLAB_PORT_MAP]] — legacy MATLAB → Python port map
- [[reports/reproducibility_full_2026-05-19]] — full LIDC + LUNGx reproducibility results
- [[reports/reproducibility_2026-05-18]] — plan + smoke (lung1 + NSCLC-Cetuximab)
- [Zenodo 6762573 — CIRDataset](https://zenodo.org/records/6762573)
- [github.com/choilab-jefferson/CIR](https://github.com/choilab-jefferson/CIR)
