# LIDC-IDRI — TCIA → nodule classification

End-to-end pipeline for the public LIDC-IDRI lung nodule cohort.

## Caveat: nodule labels

LIDC-IDRI ships its nodule annotations as **per-reader XML** (not
RTSTRUCT). The `qr convert rtstruct` step in the generated plan will
fail unless you first voxelise the XML annotations into binary label
NRRDs. Two practical paths:

1. **pylidc** (https://pylidc.github.io/) — produces per-nodule label
   volumes that can be combined into a single binary mask per patient.
2. **STAPLE / 50% consensus** of the four readers — see the LIDC
   documentation for the formal definition.

Place the resulting `<patient_id>_Nodule-label.nrrd` files next to the
downloaded CT NRRDs and re-run the pipeline with `dicom_to_ml`
(skipping `tcia_to_ml`'s automatic RTSTRUCT step):

```bash
qr workflow plan -t dicom_to_ml -d runs/lidc_idri/nrrd \
    -c clinical/clinical.csv --roi Nodule --pattern ct-default \
    --task classify --outcome malignancy \
    -o plan.json
./deploy.sh
```

## Engine

`plan.json` sets `"engine": "rtools"` (native, ~97 features per nodule). The
paper-reproduction scripts (`reproduce_papers.py`, `methods_compare.py`) expect
pyradiomics column names (`original_shape_*`, `original_firstorder_*`, ...), so
extract with `python pipelines/lidc_idri/extract_features.py --engine pyradiomics`
(and set `"engine": "pyradiomics"` in the plan) to reproduce them. Needs
`pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"`.

## Cohort details

- TCIA collection: https://www.cancerimagingarchive.net/collection/lidc-idri/
- N: 1,018 patients (DICOM CT + XML annotations)
- Default outcome: `malignancy` (1–5 reader scale; threshold at ≥4 for
  binary `malignant` per the canonical literature)

## Outputs

Same shape as `lung1/runs/` but `cv_metrics.json` reports AUC instead
of c-index.
