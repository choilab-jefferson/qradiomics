# IBSI-1 Digital-Phantom Feature Parity — PySERA Engine

**QA check:** C03 — IBSI-1 compliance (PySERA sibling; was excluded from #12's promote pending real verification)
**Date:** 2026-07-23
**Extractor:** `pysera.process_batch` (`pysera` 2.1.5)
**Phantom:** IBSI-1 chapter-2 digital phantom (`mirp/test/data/ibsi_1_digital_phantom/nifti`, 5x4x4 grid, 74 ROI voxels, grey levels {1,3,4,6})
**Reference:** `mirp/test/data/ibsi_1_reference_values/ibsi_1_dig_phantom.csv` (482 valued tags)
**Config:** IBSI "A" analogue — `bin_size=1`, `apply_preprocessing=False` (see `pipelines/ibsi1/run_ibsi1_pysera_parity.py`)

## Verdict

**PASS.** Of the 119 IBSI tags with a clean 1:1 pysera correspondence,
**117 match the published reference within tolerance (98.3%)** — slightly
higher than the PyRadiomics engine's 98.0% (97/99), and with a larger mapped
set (pysera implements genuine NGLDM and additional GLCM conventions that
PyRadiomics does not). The two failures are a single documented pysera
normalisation-constant deviation (IDN/IDMN), not a configuration error.

| Metric | Value |
|---|---|
| Reference tags (valued) | 482 |
| Mapped (1:1 correspondence) | 119 |
| Within tolerance | 117 |
| **Pass rate** | **98.3%** |

## Config (matches `run_ibsi1_parity.py`'s config "A" intent)

- No interpolation / resampling (`apply_preprocessing=False`).
- No resegmentation (pysera default `radiomics_isReSegRng=0`).
- Fixed-bin-size discretisation, `bin_size=1` — the 6 integer grey levels
  map to their own bins. **Important:** pysera's own default (`bin_size=25`,
  i.e. `radiomics_BinSize=25`) silently collapses this phantom's 6 grey
  levels into a single bin, producing degenerate/NaN `ih_*` values. This is
  the one non-obvious parameter fix required to make the identity-mapping
  placeholder in the original module produce anything meaningful at all.
- Full 3D aggregation (`_3D` / `_3D_avg` tag families only, matching the
  PyRadiomics sibling's scope).

## Name mapping & aggregation

pysera's own column names are close to, but not identical to, the IBSI tags
verbatim (the placeholder's original assumption). Differences found and
corrected:

- `cm_*` (IBSI) -> pysera `glcm_*`; `rlm_*` -> `glrlm_*`; `ngt_*` -> `ngtdm_*`.
  `szm_*` and `stat_*`/`ih_*` prefixes are unchanged.
- A handful of features are simply renamed: `rlm_rl_entr_3D_avg` ->
  `glrlm_rl_entropy_3D_avg`; `szm_szlge/szhge/lzlge/lzhge_3D` ->
  `szm_sdlge/sdhge/ldlge/ldhge_3D`; `ngl_dc_perc_3D` -> `ngl_dcperc_3D`.
- pysera's `stat_kurt` / `ih_kurt` are **already excess kurtosis** (matches
  IBSI directly) — unlike PyRadiomics, no `-3` correction is applied.
- **Bonus over the PyRadiomics engine:** pysera's `glcm_clust_shade_3D_avg`,
  `glcm_clust_prom_3D_avg`, and `glcm_info_corr1_3D_avg` all match the IBSI
  reference exactly. The PyRadiomics sibling map intentionally leaves these
  three unmapped there (documented sign/normalisation mismatch specific to
  PyRadiomics' `glcm` implementation) — pysera does not have that mismatch,
  so they are included here.
- **Bonus over the PyRadiomics engine:** pysera implements a genuine IBSI
  NGLDM (`ngl_*`, 17 `_3D` tags, all passing). PyRadiomics' `gldm` is a
  different construct entirely and is not mapped to `ngl_*` in the
  PyRadiomics sibling.

### Coverage by family (within-tolerance / mapped)

| Family | Pass / Mapped |
|---|---|
| morphology (`morph_*`) | 11 / 11 |
| statistics (`stat_*`) | 15 / 15 |
| intensity histogram (`ih_*`) | 15 / 15 |
| GLCM (`cm_*`) | 22 / 24 |
| GLRLM (`rlm_*`) | 16 / 16 |
| GLSZM (`szm_*`) | 16 / 16 |
| NGTDM (`ngt_*`) | 5 / 5 |
| NGLDM (`ngl_*`) | 17 / 17 |

Notably, the two morphology PCA-axis features that fail for PyRadiomics
(`morph_pca_min_axis`, `morph_pca_least_axis` — a known PyRadiomics-vs-IBSI
axis-length convention gap) **pass exactly** with pysera
(`morph_minor_axis_length` = 9.31 vs ref 9.31; `morph_least_axis_length` =
8.54 vs ref 8.54).

## Failures

| feature | ours | reference | tolerance | abs diff |
|---|---|---|---|---|
| cm_inv_diff_norm_3D_avg | 0.833 | 0.851 | 0 | 0.018 |
| cm_inv_diff_mom_norm_3D_avg | 0.870 | 0.898 | 0 | 0.028 |

Both are Inverse-Difference-(Moment)-Normalized GLCM features. Traced to
pysera's source
(`pysera/engine/visera_oop/core/extractors/gray_level_cooccurrence_matrix_features_extractor.py`,
`_calc_inv_diff_norm` / `_calc_inv_diff_mom_norm`): pysera normalises the
`|i-j|` distance by `(Ng - 1)` in the denominator, where `Ng` is the number
of grey levels. The IBSI reference formula normalises by `Ng` directly. This
is a genuine pysera implementation convention difference — not a
configuration mistake on our side, not force-mapped away, reported as-is
(same discipline as the PyRadiomics sibling's PCA-axis note).

## Honestly out of scope (not mapped)

Not computed by pysera at all under the `categories`/`dimensions` used here,
so left unmapped (same discipline as the PyRadiomics sibling):

- **GLDZM** (`dzm_*`) — pysera has no `gldzm` category in this version.
- **IVH** (`ivh_*`, intensity-volume histogram) and **local intensity peak**
  (`loc_peak_*`) — not emitted by `process_batch` under `handcrafted_feature`
  extraction mode.
- **2D / 2.5D aggregations** (`*_2D_*`, `*_2.5D_*`) — pysera does emit these
  columns (it was run with `dimensions="1st,2d,2_5d,3d"`), but consistent
  with the PyRadiomics sibling's config-"A" scope we only map the `_3D` /
  `_3D_avg` forms here; a 2D/2.5D parity pass would be a separate check.
- Morphology density/shape descriptors with no clean reference counterpart
  in this mapping pass (`morph_comp_*`, `morph_*_dens_*`, etc.) — same
  omissions as the PyRadiomics sibling report.

## Reproduce

```bash
pip install -e ".[pysera]"
python pipelines/ibsi1/run_ibsi1_pysera_parity.py          # prints table + pass rate
python -m pytest tests/test_ibsi1_pysera_parity.py         # regression gate (floor 95%)
```

The phantom lives outside the repo; override its location with
`IBSI1_MIRP_DATA` (default `~/gitRepos/mirp/test/data`, a sparse checkout of
`oncoray/mirp`'s `test/data` directory — the reference fixtures are not
published on PyPI). The test skips cleanly when the data or pysera is
absent.

## Artifacts

- Engine + mapping: `pipelines/ibsi1/run_ibsi1_pysera_parity.py`
- Regression test: `tests/test_ibsi1_pysera_parity.py` (asserts >=95% of mapped within tolerance)
- PyRadiomics sibling: `reports/radiomics_qa/rigor/ibsi1_phantom_parity.md`
