# Testing & validating radiomics work

Three tiers of test, cheapest first. Use the cheapest that gives you confidence, and
reach for real TCIA data when you need a true end-to-end / reproducibility check.

## Tier 1 — offline, no data (seconds)

The shape and atomic tests run on **synthetic phantoms** (`qradiomics.shape.phantoms`),
so they need no patient data and no network. This is the first thing to run after any
change:

```bash
python -m pytest tests/ -q                     # full suite
python -m pytest tests/shape/ tests/test_data_model.py -q   # phantom-only subset, ~4s
python -m pytest tests/test_lidc.py -q         # the 13 LIDC reproducibility tests
```

To smoke-test the feature pipeline itself without a real scan, build a synthetic
image+mask pair and run `qr extract` on it — this exercises the exact PyRadiomics path
`qr extract` uses:

```python
import numpy as np, SimpleITK as sitk
img = np.random.RandomState(0).randint(-200, 200, (20, 40, 40)).astype("int16")
mask = np.zeros((20, 40, 40), "uint8")
zz, yy, xx = np.ogrid[:20, :40, :40]
mask[((zz-10)**2 + (yy-20)**2 + (xx-20)**2) < 36] = 1   # a sphere
for arr, name in [(img, "CT.nrrd"), (mask, "label.nrrd")]:
    im = sitk.GetImageFromArray(arr); im.SetSpacing((1, 1, 1))
    sitk.WriteImage(im, name)
```
```bash
printf 'patient_id,modality,image_path,mask_path\nphantom,CT,CT.nrrd,label.nrrd\n' > manifest.csv
qr extract -m manifest.csv -p ct-default -o features.csv   # → "phantom ok (1409 features)"
qr shape extract -m manifest.csv -o shape.csv              # → AHSN (ahsn_000..) + spiculation
```

Also validate any workflow you author without running it — see the `--dry-run` path in
`references/workflows.md`. `qr workflow run plan.json --executor inline --dry-run`
prints every resolved command so you can confirm flags/paths against zero real files.

## Tier 2 — the IBSI digital phantom (offline, reference-checked)

For **numerical correctness** of feature values, the IBSI digital phantom has published
closed-form reference values. `examples/ibsi_validation.sh` clones
[`theibsi/data_sets`](https://github.com/theibsi/data_sets), builds a one-row manifest
against the bundled NRRD phantom, runs `qr extract -p ct-default`, and points you at the
IBSI reference table to diff each feature class. Use this when you changed extraction
settings and need to prove parity.

## Tier 3 — real TCIA data (end-to-end / reproducibility)

TCIA public collections are the real integration test: they exercise download →
DICOM→NRRD → manifest → extract → merge → analyze exactly as production does. Keep it
small first — cap the series count so a smoke run is minutes, not hours.

**Smoke recipe (one small cohort, a few series):**

```bash
# 1. data — list then download a capped slice of a collection
qr tcia series   -c "NSCLC-Radiomics" -m CT -m RTSTRUCT -o series.csv
qr tcia download -m series.csv -o ./raw --max-series 4 -j 8      # --max-series keeps it fast

# 2. image — DICOM → NRRD (CT + RTSTRUCT ROI)
qr convert dicom-series -i ./raw/<patient>/<study>/<ct_series> -o nrrd/<pid>_CT.nrrd
qr convert rtstruct -c ./raw/<patient>/<study>/<ct_series> -s <RTSTRUCT.dcm> -r GTV-1 \
  -o nrrd/<pid>_GTV-label.nrrd

# 3. features
qr convert manifest-from-dir -i nrrd -o manifest.csv
qr extract -m manifest.csv -p ct-default -o features.csv

# 4. modeling (needs a clinical CSV; qr tcia clinical fetches collection metadata)
qr tcia clinical -c "NSCLC-Radiomics" -o clinical.csv
qr results merge -f features.csv -c clinical.csv -o analysis_ready.csv
qr analyze survival -i analysis_ready.csv -o cox.csv
```

Or drive the whole thing from one template and dry-run it first:

```bash
qr workflow plan -t tcia_to_ml -C "NSCLC-Radiomics" -c clinical.csv \
  --roi GTV-1 --pattern nsclc-survival --task survival --outcome OS_event \
  --max-series 4 -o plan.json
qr workflow run plan.json --executor inline --dry-run   # confirm commands
qr workflow run plan.json --executor nextflow           # then run for real
```

**The ready-made deployable pipelines** already wire a cohort end-to-end — the fastest
"does my environment work" check is `cd pipelines/lung1 && ./deploy.sh` (edit the
clinical stub first). Cohorts and default ROIs: `lung1` (NSCLC-Radiomics, GTV-1),
`nsclc_cetuximab` (PTV), `acrin_heart` (Heart, classify), `lidc_idri` (nodule labels).

**Reproducibility harness.** To check you still hit the published AUCs (spic6 RM ≈ 0.816,
radiomics+spic PM ≈ 0.868, LUNGx ext ≈ 0.756), follow the canonical protocol in
`reports/reproducibility.md`: `qr lidc convert-cohort` → `pipelines/lidc_idri/extract_cir.py`
(LIDC + LUNGx) → `methods_compare.py --lidc-pm-ids pipelines/lidc_idri/lidc_pm_ids.txt`.
A 10-patient LIDC smoke is ≈ 10 min; the full 1,018-patient run is ≈ 6 h.

## Notes on data & the environment

- **TCIA needs network and disk.** Full collections are large — always cap with
  `--max-series` for a smoke run, and clean up `raw/` afterward (large `*.nrrd`/DICOM
  are `.gitignore`d but eat the session disk allowance).
- **`pyradiomics` install is the usual failure point**, not qradiomics. Install it from
  upstream git *first* (`pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"`);
  its PyPI metadata is broken. If a transitive build (docopt/pykwalify) or a NumPy 2.x
  ABI mismatch blocks it, that's a pyradiomics packaging issue — pin `numpy<2` and
  install pyradiomics's deps (`pywavelets`, `pykwalify`, `six`) before it.
- **Never let PHI into logs or commits** while testing. `pid`/`patient_id` stay
  anonymized; stage explicit files, never `git add -A`.
