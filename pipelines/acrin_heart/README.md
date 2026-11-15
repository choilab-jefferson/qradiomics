# ACRIN-NSCLC-FDG-PET — TCIA → heart-ROI classification

End-to-end pipeline for the ACRIN 6668 lung cancer FDG-PET cohort,
focused on heart-ROI radiomics + binary FDG-uptake classification.

## Run

```bash
cp /path/to/acrin_clinical.csv clinical/clinical.csv
./deploy.sh
```

## Cohort details

- TCIA collection: https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/
- N: ~250 patients (DICOM CT + DICOM PT + RTSTRUCT)
- ROI: `Heart` (case-insensitive — pipeline auto-matches `heart` / `HEART`)
- Pattern: `ct-default` (1409 features with `--engine pyradiomics`; ~97 with the default rtools engine)

> **Engine.** `plan.json` sets `"engine": "rtools"` (native, ~97 Original-image
> features per patient; no extra install). The `ct-default` settings (filtered image
> types, binWidth, ...) apply only to the pyradiomics engine: the 1409-feature set (9 image types x all classes) is pyradiomics-only. To
> get them, set `"engine": "pyradiomics"` in the plan's `vars` (or regenerate with
> `qr workflow plan -t tcia_to_ml --pattern ct-default --engine pyradiomics`) after
> `pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"`.
> The `rtools` and `pyradiomics` feature names and values differ; do not mix them.
- Default outcome: `fdg_uptake_binary` (your derived binary FDG-uptake label)

## Clinical CSV format

```csv
patient_id,fdg_uptake_binary,age,sex
ACRIN-NSCLC-FDG-PET-002,0,67,M
ACRIN-NSCLC-FDG-PET-009,1,73,M
...
```

See `../lung1/README.md` for the canonical TCIA→ML structure; the
ACRIN pipeline uses `--task classify --outcome fdg_uptake_binary`.
