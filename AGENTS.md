# AGENTS.md — qradiomics-public for AI Coding Agents

> **For Claude Code, Cursor, Gemini CLI, Codex CLI, opencode, aider, and any other coding-agent tooling.** This file is the single self-contained brief that lets an AI agent install, configure, and run end-to-end reproducibility experiments on this repository without further instruction.

`qradiomics-public` is a Python radiomics toolkit with:

- A reusable atomic feature-extraction core (`qradiomics.atomic`, `qradiomics.io`, `qradiomics.shape`).
- A `qr` CLI for DICOM → NRRD conversion, ROI extraction, PyRadiomics-backed feature extraction, ML wrappers, and TCIA downloads.
- An LIDC-IDRI / LUNGx reproducibility pipeline that quantitatively reproduces Choi 2018 (Med Phys), Choi 2021 (CMPB), and CIR 2022 (MICCAI), with a drop-in benchmark harness for any new method.

## 1. What an agent should read first

If your tooling supports skills (Claude Code and compatible agents), the bundled skill
at `.claude/skills/qradiomics/` auto-triggers in this repo and is the fastest orientation
— it distills this file, the CLI, the Python API, workflow/pipeline authoring, and the
TCIA testing paths into progressive-disclosure references. It is a navigation aid, not a
replacement: the files below remain the source of truth, and this AGENTS.md wins on any
conflict.

In strict order:

1. `README.md` — repository overview, install commands, end-to-end example.
2. `reports/reproducibility.md` — single canonical reproducibility document (results, methodology, paper citations).
3. `reports/PUBLIC_DATASET_FRAMEWORK.md` — public-dataset loader contract (TCIA / Zenodo / GitHub).
4. `pipelines/lidc_idri/` — the 13 reproducibility scripts (one per task).
5. `qradiomics/` — atomic core (`atomic/`, `io/`, `shape/`, `cli/`).
6. `tests/` — pytest suite (13 lidc tests, atomic-core tests).

## 2. Install — operational notes

> The actual install/smoke recipe for agents lives in `scripts/kickoff.sh`
> and the kick-off prompt block at the top of `README.md`. This section
> only documents what the install touches at the system level.

- Python 3.11+. `pip install -e .` installs the `qradiomics` package and
  the `qr` / `qradiomics` / `qrdx` entry points.
- System prerequisites pulled by pip: `numpy`, `pandas`, `SimpleITK`,
  `pyradiomics`, `pydicom`, `scikit-learn`, `lifelines`, `scipy`,
  `scikit-image`. GPU is not required.
- Verify with `qr info` and `python3 -m pytest tests/test_lidc.py -q`
  (→ 13 passed).

## 3. End-to-end reproducibility — headline numbers

The LIDC-IDRI / LUNGx reproducibility harness reproduces three
published methods on paper-grade CIR masks:

```
spic6 RM AUC        ≈ 0.816 ± 0.006   (paper: CMPB 2021, 0.80-0.85)
radiomics+spic PM   ≈ 0.868 ± 0.039   (paper: CMPB 2021, 0.85)
LUNGx ext + cal     ≈ 0.756           (paper: CMPB 2021, 0.76)
```

A LIDC-only smoke run on 10 patients takes ≈ 10 minutes on a 16-core
workstation. The full 1,018-patient reproduction takes ≈ 6 hours
(5 hours feature extraction + 1 hour modelling).

Driving data + commands:
- Imaging: `qr tcia download --collection LIDC-IDRI --modality CT`
  and `--collection "SPIE-AAPM Lung CT Challenge"`.
- Auxiliary: LIDC XML (LIDC-XML-only.zip), LUNGx calibration +
  test xlsx, CIRDataset_LCSR (Zenodo 6762573).
- Conversion + extraction + comparison: `qr lidc convert-cohort` +
  `pipelines/lidc_idri/extract_cir.py` (LIDC, LUNGx) +
  `pipelines/lidc_idri/methods_compare.py` with
  `--lidc-pm-ids pipelines/lidc_idri/lidc_pm_ids.txt`.

See `reports/reproducibility.md` for the full RM / PM /
LUNGx-cal / LUNGx-test protocol and the canonical command sequence.

## 4. Pipeline catalogue — one file per task

| Script | Purpose | Inputs | Outputs |
| :--- | :--- | :--- | :--- |
| `pipelines/lidc_idri/extract_features.py` | Per-(reader, nodule) PyRadiomics 1409 + spiculation on LIDC XML masks | converted LIDC-IDRI-out | wide features CSV |
| `pipelines/lidc_idri/extract_lungx.py` | LUNGx with intensity-based region-grow masks | LUNGx TCIA + xlsx | wide features CSV |
| `pipelines/lidc_idri/extract_cir.py` | LIDC / LUNGx with **CIRDataset paper-grade masks** | CIRDataset_LCSR archive | wide features CSV |
| `pipelines/lidc_idri/ahsn_proxy.py` | Choi 2014 AHSN — annotated nodule centroids + random lung-tissue negatives | LIDC-IDRI-out | candidates CSV (180-D AHSN per row) |
| `pipelines/lidc_idri/ahsn_hardneg.py` | Choi 2014 AHSN — annotated nodule + LIDC XML `nonNodule` hard negatives | converted LIDC + XML | candidates CSV |
| `pipelines/lidc_idri/reproduce_papers.py` | Med Phys 2018 + CMPB 2021 leakage-safe RF CV | features CSV | markdown report |
| `pipelines/lidc_idri/reproduce_cir.py` | CIR LIDC-PM internal + LIDC→LUNGx external (10-pat calibration) | LIDC + LUNGx CSVs + LIDC-PM ID list | markdown report |
| `pipelines/lidc_idri/methods_compare.py` | **Methods harness** (RM / PM / LUNGx-cal / LUNGx-test on CMPB 2021 protocol) | LIDC + LUNGx CSVs | markdown report |
| `pipelines/lidc_idri/validate_lcsr.py` | Spearman ρ qradiomics vs LCSR bundled reference | CIRDataset | comparison CSV |
| `pipelines/lidc_idri/mesh_to_voxel_compare.py` | Voxelise our mesh peaks; Dice vs LCSR; export OBJ + NRRD | CIRDataset | comparison CSV + meshes |
| `pipelines/lidc_idri/run.sh` | Convenience wrapper for the full pipeline | env vars | features CSV |
| `pipelines/lidc_idri/lidc_pm_ids.txt` | 72 pathology-confirmed LIDC patient IDs (pinned from CIR) | n/a | text list |

## 5. Atomic core API quick reference

```python
from qradiomics.atomic import extract_features, load_image_and_mask
from qradiomics.shape import spiculation_from_voxel, ahsn, detect_candidates
from qradiomics.io.lidc import (parse_lidc_xml, convert_patient, scan_lidc_dir,
                                 staple_consensus, staple_patient)
from qradiomics.io.dicom import load_dicom_series

# Most common workflow:
image, mask = load_image_and_mask("CT.nrrd", "PTV-label.nrrd")
features = extract_features(image, mask, params_file=None, label=1)
#  → dict of ~1,409 PyRadiomics features

# Spiculation features on a binary voxel mask (Choi 2021):
sf, peaks, distortion, mesh = spiculation_from_voxel(mask_np_array, spacing=(0.7, 0.7, 0.7))
#  → SpiculationFeatures(Np, Na, Nl, Na_att, s1, s2)
```

## 6. Adding a new feature-extraction method to the benchmark harness

```python
# In pipelines/lidc_idri/methods_compare.py
METHODS["my_method"] = lambda df: [c for c in df.columns if c.startswith("my_prefix_")]
```

That's it. The same RM / PM / LUNGx-cal / LUNGx-test splits and the leakage-safe RF CV are applied automatically. The output is appended to the comparison table.

## 7. Coding conventions

- Style: PEP 8, type hints encouraged. Lint via `ruff` (not enforced via pre-commit).
- Imports: prefer `from qradiomics.x import Y` over absolute `import qradiomics.x` aliases.
- Errors: raise concrete exception types (`ValueError`, `FileNotFoundError`) rather than bare `Exception`.
- DICOM / NRRD I/O: always go through `qradiomics.io.dicom.load_dicom_series` and `SimpleITK.ReadImage` / `WriteImage`. Avoid pydicom for image arrays.
- Tests: pytest, all new pipeline code should have a `tests/test_<module>.py` with at least one synthetic-data unit test.

## 8. Don't do these

- Never log Protected Health Information (PHI) — patient names, MRNs, dates of birth, etc. All scripts treat `pid` as an anonymised identifier; downstream rendering must not expand it.
- Never `git add -A` — stage explicit file lists. The repo's `.gitignore` excludes large `*.nrrd` / `*.csv` outputs but does not catch every accidental case.
- Never push to remotes without explicit user confirmation. The `origin` remote is the public GitHub mirror.

## 9. Paper citations

If your AI-generated work uses qradiomics-public, cite (in order of relevance):

- Choi W, Choi T-S. *Comput Methods Programs Biomed* 2014;113(1):37–54 — AHSN.
- Choi W, Oh JH, Riyahi S, et al. *Med Phys* 2018;45(4):1537–1549 — radiomics for early lung-cancer detection (Editor's Pick).
- Choi W, Nadeem S, Riyahi S, et al. *Comput Methods Programs Biomed* 2021;200:105839 — interpretable spiculation quantification.
- Choi W, Dahiya N, Nadeem S. *MICCAI 2022* — CIRDataset. arXiv:2206.14903.

Full bibliographic details + supporting cohort references are in `reports/reproducibility.md` §10.

## 10. Kick-off — see README.md

The user-facing one-line install and the drop-in agent prompt both live
in `README.md` under "Kick-off". This AGENTS.md stays operations-only;
when an agent finishes the kick-off, it should come here for
conventions, layout, don'ts, and citations.

Backend that the kick-off prompt actually runs: `scripts/kickoff.sh`.
