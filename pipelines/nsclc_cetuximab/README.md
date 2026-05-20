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
- Clinical columns expected: `patient_id`, `OS_days`, `OS_event` (or your equivalents — adjust `plan.json`)

See `../lung1/README.md` for the canonical TCIA→ML pattern; this
pipeline is identical with `--collection NSCLC-Cetuximab --roi PTV`.
