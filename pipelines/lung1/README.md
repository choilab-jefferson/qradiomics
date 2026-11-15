# LUNG1 (NSCLC-Radiomics) — TCIA → survival prediction

End-to-end pipeline:

```
qr tcia download           NSCLC-Radiomics DICOM CT + RTSTRUCT
    ↓
qr convert dicom-series    CT DICOM → NRRD
qr convert rtstruct        RTSTRUCT GTV-1 → label NRRD
    ↓
qr extract -p nsclc-survival   ~97 features (rtools) / ~1130 (--engine pyradiomics)
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
- Pattern: `nsclc-survival` (Original + LoG + Wavelet + Square + SquareRoot + Logarithm, applied by the pyradiomics engine only)

> **Engine.** `plan.json` sets `"engine": "rtools"` (native, ~97 Original-image
> features per patient; no extra install). The `nsclc-survival` settings (filtered image
> types, binWidth, ...) apply only to the pyradiomics engine: the Aerts/Lung1 set of ~1130 features (Original + LoG + Wavelet + Square + SquareRoot + Logarithm) is reproduced only with pyradiomics. To
> get them, set `"engine": "pyradiomics"` in the plan's `vars` (or regenerate with
> `qr workflow plan -t tcia_to_ml --pattern nsclc-survival --engine pyradiomics`) after
> `pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"`.
> The `rtools` and `pyradiomics` feature names and values differ; do not mix them.

## Verified end-to-end on a live TCIA download

```
qr tcia series   --collection NSCLC-Radiomics --patient LUNG1-001 -o series.csv
qr tcia download --manifest series.csv -o dicom/
qr convert dicom-series -i dicom/LUNG1-001/<study>/<ct-uid>/ -o LUNG1-001_CT.nrrd
qr convert rtstruct     -d dicom/LUNG1-001/<study>/<ct-uid>/ \
                        -r dicom/LUNG1-001/<study>/<rs-uid>/ \
                        --roi GTV-1 -o LUNG1-001_GTV-1-label.nrrd
qr extract -m manifest.csv -p nsclc-survival --engine pyradiomics -o features.csv
# → 1130 features for LUNG1-001 (the canonical Aerts feature set; needs pyradiomics.
#   Without --engine pyradiomics the native rtools engine gives ~97 features)
```

## Outputs (under `runs/lung1/`)

| File | Purpose |
|---|---|
| `series.csv` | TCIA series-level manifest |
| `dicom/` | downloaded DICOM tree |
| `nrrd/` | per-patient `*_CT.nrrd` + `*_GTV-1-label.nrrd` |
| `manifest.csv` | qr-style manifest CSV |
| `features.csv` | ~97 (rtools) or ~1130 (pyradiomics) radiomics features per patient |
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
./deploy.sh
```
