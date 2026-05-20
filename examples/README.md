# `examples/` — End-to-end workflows on TCIA public cohorts

Each script demonstrates the canonical qradiomics CLI pipeline on a
TCIA-public cohort using only `qr` commands. Replace `<DATASET_ROOT>`
and `<OUT>` with your paths.

| Script | Cohort | Modality | Mask source | Pattern |
|---|---|---|---|---|
| `lung1_survival.sh` | [NSCLC-Radiomics (LUNG1)](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) | CT | DICOM RTSTRUCT GTV-1 | `nsclc-survival` |
| `nsclc_cetuximab_survival.sh` | [NSCLC-Cetuximab](https://www.cancerimagingarchive.net/collection/nsclc-cetuximab/) | CT | DICOM RTSTRUCT PTV | `nsclc-survival` |
| `acrin_heart_classification.sh` | [ACRIN-NSCLC-FDG-PET](https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/) | CT | DICOM RTSTRUCT Heart | `ct-default` |
| `lidc_idri_nodule.sh` | [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) | CT | XML annotation → label NRRD | `ct-default` |
| `ibsi_validation.sh` | [IBSI digital phantom](https://github.com/theibsi/data_sets) | — | bundled | `ct-default` |

Each script is idempotent: re-running skips already-converted files.

## Pre-requisites

```bash
pip install qradiomics[rtstruct]
# rt-utils is required by qr convert rtstruct
```

Download cohorts from TCIA via the [TCIA Data Retriever](https://www.cancerimagingarchive.net/get-tcia-data/)
or `nbia-data-retriever`. Each script assumes the standard TCIA layout
under `<DATASET_ROOT>/<patient>/<study>/`.

## What each script does

```
DICOM CT series          ─┐
                           ├─→ qr convert dicom-series  → CT NRRD
RTSTRUCT contour         ─┤
                           └─→ qr convert rtstruct      → label NRRD
                                                          │
                                          manifest.csv ←──┘
                                          │
                                          ├─→ qr extract          → features.csv
                                          ├─→ qr results merge     → analysis_ready.csv
                                          └─→ qr analyze {survival|classify|importance}
```
