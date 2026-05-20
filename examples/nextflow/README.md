# `examples/nextflow/` — qr CLI driven by Nextflow

The standalone shell scripts in `examples/` cover the canonical
single-process path. For large cohorts (hundreds to thousands of patients)
or distributed execution on HPC/AWS, drive the same `qr` commands from
a Nextflow workflow:

```bash
pip install qradiomics[rtstruct]
nextflow run qradiomics.nf \
    --dataset_root /path/to/cohort \
    --clinical    /path/to/clinical.csv \
    --roi         GTV-1 \
    --pattern     nsclc-survival \
    --analysis    survival \
    --outdir      results \
    -profile      local
```

## What you get from Nextflow + qr

| Capability | shell loop | Nextflow + qr |
|---|---|---|
| per-patient parallelism | sequential | up to N cores / N nodes |
| caching | none (re-runs redo everything) | per-process work directory; re-runs skip OK tasks |
| resume after failure | restart from scratch | `nextflow run -resume` |
| container reproducibility | host install | `-profile docker` swaps in a versioned image |
| HPC scheduler | shell only | `-profile slurm` (or `awsbatch`, `k8s`, ...) |
| timeline + DAG report | none | `report.html`, `timeline.html`, `dag.html` |

## Workflow shape

```
per-patient                                          collect
┌──────────┐   ┌──────────┐   ┌───────────────┐   ┌──────────┐   ┌─────────┐
│ convert  ├──▶│ convert  ├──▶│ extract       ├──▶│ gather   ├──▶│ analyze │
│ dicom-ct │   │ rtstruct │   │ (1 patient)   │   │ features │   │         │
└──────────┘   └──────────┘   └───────────────┘   └──────────┘   └─────────┘
```

Each process runs a single `qr` invocation:

| Process | qr command |
|---|---|
| `convert_ct` | `qr convert dicom-series` |
| `convert_rt` | `qr convert rtstruct` |
| `extract_one` | `qr extract` (single-patient manifest) |
| `gather_features` | (concat per-patient CSVs) |
| `merge_clinical` | `qr results merge` |
| `analyze` | `qr analyze {survival,classify,importance}` |

## Choosing shell vs Nextflow

- **shell (`examples/*.sh`)** — interactive / small (≤ ~50 patients),
  one-shot, no cluster, no resume needed.
- **Nextflow (this directory)** — production runs, large cohorts,
  HPC, caching, container reproducibility.

The two forms share the same `qr` CLI internally, so workflows are
trivial to port either direction.

## Profiles

| Profile | Use |
|---|---|
| `local` | qradiomics installed on the same machine |
| `docker` | `qradiomics:0.9.0` container (set image tag in `nextflow.config`) |
| `slurm` | submit each process as a SLURM job |
