# LUNG1 (NSCLC-Radiomics) — TCIA → survival prediction

End-to-end pipeline:

```
qr tcia download           NSCLC-Radiomics DICOM CT + RTSTRUCT
    ↓
qr convert dicom-series    CT DICOM → NRRD
qr convert rtstruct        RTSTRUCT GTV-1 → label NRRD
    ↓
qr extract -p nsclc-survival   ~1130 features per patient
    ↓
qr results merge           with clinical/clinical.csv on patient_id
    ↓
qr ml train + evaluate     Cox PH with 5-fold CV → model.pkl + metrics.json
```

## Run

```bash
# Drop your clinical CSV in place (patient_id, OS_days, OS_event columns)
cp /path/to/lung1_clinical.csv clinical/clinical.csv

./deploy.sh                  # nextflow (default)
EXECUTOR=prefect ./deploy.sh # via Prefect 2.x
EXECUTOR=inline ./deploy.sh  # sequential subprocess (smoke tests)
```

## Cohort details

- TCIA collection: https://www.cancerimagingarchive.net/collection/nsclc-radiomics/
- Citation: Aerts HJWL, et al. *Decoding tumour phenotype by noninvasive imaging
  using a quantitative radiomics approach.* Nature Communications 2014; 5:4006.
- N: 422 patients (CT + manual RTSTRUCT GTV-1)
- Pattern: `nsclc-survival` (Original + LoG + Wavelet + Square + SquareRoot + Logarithm)

## Verified end-to-end on a live TCIA download

```
qr tcia series   --collection NSCLC-Radiomics --patient LUNG1-001 -o series.csv
qr tcia download --manifest series.csv -o dicom/
qr convert dicom-series -i dicom/LUNG1-001/<study>/<ct-uid>/ -o LUNG1-001_CT.nrrd
qr convert rtstruct     -d dicom/LUNG1-001/<study>/<ct-uid>/ \
                        -r dicom/LUNG1-001/<study>/<rs-uid>/ \
                        --roi GTV-1 -o LUNG1-001_GTV-1-label.nrrd
qr extract -m manifest.csv -p nsclc-survival -o features.csv
# → 1130 features for LUNG1-001 (the canonical Aerts feature set)
```

## Outputs (under `runs/lung1/`)

| File | Purpose |
|---|---|
| `series.csv` | TCIA series-level manifest |
| `dicom/` | downloaded DICOM tree |
| `nrrd/` | per-patient `*_CT.nrrd` + `*_GTV-1-label.nrrd` |
| `manifest.csv` | qr-style manifest CSV |
| `features.csv` | ~1130 radiomics features per patient |
| `analysis_ready.csv` | features merged with clinical OS_months / OS_event |
| `model.pkl` | trained Cox PH model |
| `cv_metrics.json` | 5-fold CV c-index mean ± std |
| `evaluation.json` | hold-out c-index on the full cohort |

## Skip download (local data path)

If you already have the DICOM tree on disk, regenerate the plan with
`dicom_to_ml` instead of `tcia_to_ml`:

```bash
qr workflow plan -t dicom_to_ml -d /path/to/lung1/dicom \
    -c clinical/clinical.csv --roi GTV-1 --pattern nsclc-survival \
    -o plan.json
qr workflow scaffold -p plan.json -e nextflow -o main.nf
./deploy.sh
```
