# Changelog

## v0.9.9 (2026-11-14)

- **Behaviour change: `rtools` is now the default extraction engine everywhere** (#28).
  pyradiomics cannot be installed from PyPI on Python 3.10+, so a plain
  `pip install qradiomics` failed on the first `qr extract`. `qr extract` without
  `--engine` now runs the native `rtools` engine, and so do all shipped defaults:
  - `qradiomics.extractor_registry.DEFAULT_ENGINE`, `FeatureExtractionConfig.extractor`
    and `pattern_schema.yaml` default to `rtools`.
  - The 5 shipped patterns (`ct-default`, `pet-default`, `nsclc-survival`,
    `standard-radiomics`, `survival-analysis`) now set `extractor: rtools`.
  - `pipelines/{acrin_heart,lidc_idri,lung1,nsclc_cetuximab}/plan.json` and
    `nf-modules/cohorts/*.json` set `"engine": "rtools"`; `qr workflow plan --engine`
    and the template `engine` parameters default to the pattern's extractor (rtools).
  - `pipelines/common/engine_extract.extract_features_multi` (LIDC scripts) routes
    `engine=None` through the registry (rtools); only `engine="pyradiomics"` calls
    `qradiomics.atomic.extract_features`. `pipelines/run_cohort_nf.sh` and the
    `examples/*.sh` scripts take an `ENGINE` variable (default `rtools`).
  - `qradiomics.atomic.extract_features` is unchanged: an explicit pyradiomics
    primitive that no CLI or workflow default path routes to.
- **What rtools does not reproduce.** rtools computes Original-image 2D/3D intensity,
  shape and GLCM/GLRLM features (97 per patient, radiomics-tools feature names). It
  does not compute filtered images (LoG, Wavelet, Square, SquareRoot, Logarithm,
  Exponential, Gradient, LBP) or the GLSZM/GLDM/NGTDM classes, does not apply
  pyradiomics settings (`binWidth`, resampling, `normalize`, `preCrop`, a params YAML)
  or the pattern's `image_types` / `feature_classes`, and does not emit `original_*` /
  `wavelet-*` feature names. Feature values differ from pyradiomics even for
  overlapping classes, so do not mix features from the two engines in one analysis.
  `qr extract` prints a note on stderr when rtools ignores pattern settings.
- **Keeping the old pyradiomics behaviour** is one flag: `qr extract ... --engine
  pyradiomics` (install it first:
  `pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"`).
  The pyradiomics-only settings stay in the shipped patterns and
  `qradiomics/data/pyradiomics/*.yaml` and are applied by that engine. Examples:
  Aerts / Lung1 (`qr extract -p nsclc-survival --engine pyradiomics`, ~1130
  features), the 1409-feature `ct-default` set, and the LIDC paper reproductions
  (`extract_features.py --engine pyradiomics`). For the cohort plans set
  `"engine": "pyradiomics"` in the plan `vars` or regenerate with
  `qr workflow plan --engine pyradiomics`; for the examples and
  `run_cohort_nf.sh` set `ENGINE=pyradiomics`.
- If pyradiomics is requested (by `--engine` or by a pattern) but not installed,
  `qr extract` now stops up front with the git install command and a suggestion to
  use `--engine rtools`.
- **Colab notebooks** (#28): `examples/ibsi1_phantom_colab.ipynb` (IBSI-1 digital phantom)
  and `examples/lung1_survival_colab.ipynb` (TCIA NSCLC-Radiomics subset) run in Google
  Colab with `pip install qradiomics` only; `examples/README.md` links them and the
  agent-driven nexus.lite demo notebook.
- **Agent skill refresh**: `.claude/skills/qradiomics` and `.agents/skills/qradiomics`
  describe the rtools-default workflow (PyRadiomics optional).

## v0.9.8 (2026-11-14)

- **Generic Nextflow backend for `qr workflow run`** (#24, #25):
  `qr workflow run <plan> --executor nextflow` runs any plan through one
  packaged pipeline, `qradiomics/nf/main.nf` (+ `nextflow.config`; needs
  `nextflow` and Java 17+).
  - Plan steps carry a `fanout` (`null` / `"case"` / `"pair"`); processes are
    `QR_PRE` -> `QR_CASE` -> `QR_PAIR` -> `QR_POST`, each task being the plan's
    own `qr ...` command lines, so `-resume` caches on the exact commands.
  - Run reports (trace, report, timeline, dag) are written to
    `<outdir>/pipeline_info/`; per-pair provenance to `<outdir>/prov/`.
  - Failure policy and containers: `--on-error`, `--max-forks`,
    `--container <digest-pinned image>`, `--profile local|slurm`.
  - `qr extract --case` per-case fan-out; `qr workflow plan -t rt_pair_radiomics`.
  - The generated per-cohort `pipelines/*/main.nf` files and
    `nf-modules/workflows/*.nf` were removed; every cohort runs through the
    single entry point above. `qr workflow scaffold -e nextflow|prefect` is
    deprecated. The default executor stays `inline`.
  - Note: the plan entry point (#25) is only partly done. Plans for the
    HeartToxicity, LADC and MSKLung cohorts are follow-up work.
- **Pair manifest contract** (`qradiomics.pairs`): one row per (image, label)
  pair, `.parquet` (preferred) or `.csv`, with a shared reader used by
  `convert`, `preprocess`, `extract` (`--only`, `--only-case`) and
  `results merge` (`--pairs`, `--coverage` coverage table).
  New command `qr select validate` checks a pair manifest.
- **Dependencies**: `pandas` (`>=1.5,<3.0`) and `pyarrow` (`>=14.0`) are now
  core dependencies, since parquet pair manifests are read by core `qr`
  commands. A clear error is raised when no parquet engine is available.
- **Behaviour change: spiculation default output** (`qradiomics.shape.spiculation`):
  `spherical_parameterization` now implements the Nadeem 2017 / Choi 2021
  conformalized-MCF (cMCF) parameterization by default, replacing the previous
  1-ring neighbour-averaging relaxation (200 iterations), and
  `spiculation_from_voxel` defaults to `n_param_iter=12`
  (was 200). Spiculation features therefore differ from v0.9.7 for the same
  input. Example, default arguments on `phantoms.soft_lung_cancer_spiculated()`:
  `Np` 49 -> 3, `Na` 6 -> 3, `Nl` 43 -> 0, `s1` -0.610 -> -3.210,
  `s2` 1.212 -> 3.605. `SpiculationFeatures` also gains `spic_a`, `spic_b` and
  `roundness`. Opt-in extras (`mobius_center`, `yamabe_refine`, `yamabe_flip`,
  `sphere_kwargs`) leave the default path unchanged. Also carried from dev:
  `pre_crop` / `pad_distance` in `atomic.features` and a single-thread worker
  guard for `pipelines/common/parallel.py`. Features computed with v0.9.7
  should not be mixed with v0.9.8 spiculation features.
- Docs: `reports/rtools_port_speed_benchmark.md` (native rtools port speed).
- Tests: `tests/test_pairs.py`, `tests/test_pair_commands.py`,
  `tests/test_workflow_nextflow.py`.


## v0.9.7 (2026-11-04)

- **Native Python `rtools` engine** (`qradiomics.feature.rtools`): a port of the
  legacy radiomics-tools C++ `FeatureExtraction` binary (numpy + scipy;
  SimpleITK for image I/O only). Output matches the binary's printed values;
  neither ITK Python nor the compiled binary is required.
  - `qr extract --engine rtools` now uses the native port by default.
  - `rtools_backend="cxx"` keeps the legacy binary for parity checks.
  - `rtools_ibsi=True` selects an opt-in IBSI-1 profile
    (`rtools_ibsi_bin_width` / `rtools_ibsi_bin_number`), verified against the
    IBSI-1 digital phantom (`tests/fixtures/ibsi1_digital_phantom.json`).
- Tests: `tests/test_rtools_ibsi.py`, `tests/test_rtools_parity.py`.
