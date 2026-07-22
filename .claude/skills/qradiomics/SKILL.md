---
name: qradiomics
description: >-
  Work inside the qradiomics repository — a lung-radiomics research CLI (`qr`) and
  Python toolkit built on PyRadiomics, scikit-learn, and lifelines. Use this skill
  whenever the task touches this repo: running or wiring up the `qr` CLI (convert,
  extract, results merge, analyze, ml, shape, delta, workflow, tcia, pacs, lidc),
  writing radiomics feature-extraction or modeling code, calling the atomic Python
  core (`qradiomics.atomic`, `qradiomics.io`, `qradiomics.shape`), building or
  editing the LIDC-IDRI / LUNGx reproducibility pipelines, choosing a pattern or
  workflow template, or handling DICOM/NRRD medical imaging here. Trigger even when
  the user only names a piece ("extract features", "convert this RTSTRUCT", "fit a
  Cox model on the features", "spiculation descriptor") without naming qradiomics.
---

# qradiomics

qradiomics is a lung-radiomics research toolkit: a `qr` Click CLI plus a reusable
Python core. It is the modern Python successor to three earlier Choi Lab codebases
(AHSN 2014, radiomics 2018, spiculation 2021, CIRDataset 2022) and reproduces those
papers quantitatively. Work here almost always sits on one canonical data flow:

```
data  →  image  →  features  →  modeling
DICOM/TCIA  NRRD     features.csv   Cox / classifier / importance
```

Read this file first. Pull in a reference file only when the task lands in it:

- `references/cli.md` — every `qr` command group, its options, and copy-paste recipes.
- `references/python-api.md` — the atomic core + `qradiomics.shape` Python API.
- `references/pipelines.md` — LIDC-IDRI/LUNGx reproducibility scripts, patterns, workflow templates.

Before doing anything substantial, skim `AGENTS.md` and `README.md` at the repo
root — they are the source of truth and may have moved past this skill. `qr info`
prints the installed version; `reports/reproducibility.md` holds the paper protocol.

## Non-negotiables (read before you touch anything)

These come from `AGENTS.md §8` and exist because this is clinical-imaging code:

- **Never log or emit PHI** — no patient names, MRNs, dates of birth. Every script
  treats `pid` / `patient_id` as an *already-anonymized* identifier; keep it that way
  and never expand it back to anything identifying.
- **Never `git add -A`.** Stage explicit file lists. `.gitignore` skips large
  `*.nrrd`/`*.csv` outputs but doesn't catch every stray file, and committing a
  patient scan would be a real breach.
- **Never push to a remote without explicit confirmation** — `origin` is the public
  GitHub mirror.
- **Don't invent public-dataset URLs.** Use only the TCIA/Zenodo/GitHub sources
  documented in `AGENTS.md` and `reports/PUBLIC_DATASET_FRAMEWORK.md`.

## Install

pyradiomics has broken PyPI metadata, so order matters — install it from upstream
git **first**, then the package:

```bash
pip install "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"
pip install -e .            # in a clone; installs the qr / qradiomics / qrdx entry points
```

The one-shot backend for a fresh clone + venv + smoke test is `scripts/kickoff.sh`.
Verify with `qr info` and `python -m pytest tests/test_lidc.py -q` (→ 13 passed).
Python 3.11+, no GPU.

## The `qr` CLI at a glance

`qr` does two things: **atomic tasks** (one command, files in / files out) and
**workflow assembly** (chain those tasks into a runnable pipeline). The atomic chain
follows the data flow above:

```bash
# data → image
qr tcia download --collection LIDC-IDRI --modality CT -o <raw>   # or bring your own DICOM
qr convert dicom-series -i <series_dir> -o CT.nrrd
qr convert rtstruct -c <ct_dir> -s RTSTRUCT.dcm -r GTV-1 -o GTV-label.nrrd
qr convert manifest-from-dir -i <root> -o manifest.csv           # pairs image ↔ mask

# image → features
qr extract -m manifest.csv -p ct-default -o features.csv         # ~1409 PyRadiomics features
qr shape extract -m manifest.csv -o shape_features.csv           # AHSN 2014 + spiculation 2021
qr delta -f features.csv --pair 'delta=post-pre' -o delta.csv    # longitudinal delta/trend

# features → modeling
qr results merge -f features.csv -c clinical.csv -o analysis_ready.csv
qr analyze survival    -i analysis_ready.csv -o cox.csv          # Cox PH
qr analyze classify    -i analysis_ready.csv --outcome label -o clf.csv
qr analyze importance  -i analysis_ready.csv --outcome label -o imp.csv
qr ml train -i analysis_ready.csv --task survival --outcome OS_event --model m.pkl --metrics cv.json
qr ml predict  -i features.csv --model m.pkl --task survival -o pred.csv
qr ml evaluate -i analysis_ready.csv --model m.pkl --task survival --outcome OS_event --report ev.json
```

Workflow assembly instantiates that whole chain from one template:

```bash
qr workflow templates                                            # list templates
qr workflow plan -t dicom_to_ml -d <cohort> -c clinical.csv -o plan.json
qr workflow scaffold -p plan.json -e nextflow -o pipeline.nf
qr workflow run plan.json --executor nextflow                    # default; inline = small-cohort fallback
```

Other groups: `qr pattern list|search` (feature-extraction patterns), `qr tcia`
(public downloads), `qr pacs` (DICOM networking), `qr lidc convert|convert-cohort`
(LIDC XML → NRRD), `qr preprocess`, `qr register`, `qr hu-correct`, `qr anonymize`,
`qr config`. Full options and recipes are in `references/cli.md`.

**The manifest CSV is the spine.** Almost every extraction command reads a manifest
with columns `patient_id, modality, image_path, mask_path`. When something upstream
of `qr extract` needs wiring, you are almost always producing or fixing a manifest.

## Python core

For programmatic work (or a new pipeline step), the atomic core is the entry point.
The most common shape:

```python
from qradiomics.atomic import extract_features, load_image_and_mask

image, mask = load_image_and_mask("CT.nrrd", "GTV-label.nrrd")
features = extract_features(image, mask, params_file=None, label=1)  # dict of ~1409 features
```

Shape descriptors (Choi 2014/2021) live in `qradiomics.shape`; DICOM/LIDC I/O in
`qradiomics.io`. Always do image array I/O through `qradiomics.io.dicom.load_dicom_series`
and SimpleITK `ReadImage`/`WriteImage` — never pull pixel arrays out of pydicom. See
`references/python-api.md` for the full surface.

## Conventions when writing code here

- Style is PEP 8 with type hints; lint with `ruff` (not enforced in CI).
- Import as `from qradiomics.x import Y`, not `import qradiomics.x` aliases.
- Raise concrete exceptions (`ValueError`, `FileNotFoundError`), never bare `Exception`.
- Every new pipeline module gets a `tests/test_<module>.py` with at least one
  synthetic-data unit test — the shape and atomic tests under `tests/` use generated
  phantoms (`qradiomics.shape.phantoms`) so they need no real patient data. Follow that
  pattern; run `python -m pytest tests/ -q` before you claim something works.
- To add a method to the benchmark harness, register a column-selector lambda in
  `pipelines/lidc_idri/methods_compare.py` (`METHODS["my_method"] = lambda df: [...]`);
  the leakage-safe RF CV and all splits apply automatically.

## If you cite the toolkit

Work built on qradiomics cites the Choi Lab papers (AHSN 2014 CMPB; radiomics 2018
Med Phys; spiculation 2021 CMPB; CIRDataset MICCAI 2022). Full details are in
`AGENTS.md §9` and `reports/reproducibility.md §10`.
