# NSCLC-Cetuximab — TCIA → survival prediction

End-to-end pipeline for the RTOG-0617 cetuximab arm. Same shape as
`lung1/` but with PTV instead of GTV-1 as the contour (PTV is the
most consistent ROI name across the cohort's RTSTRUCT files).

## Run

```bash
cp /path/to/cetuximab_clinical.csv clinical/clinical.csv
./deploy.sh
```

## Cohort details

- TCIA collection: https://www.cancerimagingarchive.net/collection/nsclc-cetuximab/
- N: ~490 patients (DICOM CT + RTSTRUCT PTV)
- Pattern: `nsclc-survival`

> **Engine.** `plan.json` sets `"engine": "rtools"` (native, ~97 Original-image
> features per patient). The pattern's LoG/Wavelet/Square/SquareRoot/Logarithm image
> types (the ~1130-feature Aerts set, as for `lung1/`) apply only with
> `"engine": "pyradiomics"` in the plan's `vars` (needs
> `pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"`).
- Clinical columns expected: `patient_id`, `OS_days`, `OS_event` (or your equivalents — adjust `plan.json`)

See `../lung1/README.md` for the canonical TCIA→ML pattern; this
pipeline is identical with `--collection NSCLC-Cetuximab --roi PTV`.
