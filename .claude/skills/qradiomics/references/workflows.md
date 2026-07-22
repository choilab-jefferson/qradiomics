# Building radiomics workflows & pipelines

This is the heart of what the skill unlocks: turning the atomic `qr` tasks into
reproducible, multi-step, parallel pipelines. There are three levels, pick the
lowest one that does the job:

1. **A shell chain** — 4–6 `qr` commands in a `.sh`. Best for learning and one-offs.
   See `examples/*.sh`.
2. **A workflow plan** — a JSON/YAML `WorkflowPlan` you generate, edit, and run or
   scaffold to Nextflow/Prefect/shell. Best for a cohort you'll rerun. This page.
3. **A production pipeline directory** — a self-contained bundle under `pipelines/`
   (`plan.json` + `main.nf` + `nextflow.config` + `deploy.sh` + clinical stub). Best
   for a cohort you'll deploy/hand off. This page, last section.

The engine lives in `qradiomics/workflow.py` and is deliberately small: a plan is
**pure data**, so you (or an agent) can inspect, mutate, and re-emit it without any
state machine. Everything below is validated against that module.

## The plan data model

A `WorkflowPlan` is `{version, name, description, vars, steps[]}`. Each step is a
`WorkflowStep`:

```json
{
  "id": "extract",                     // stable, unique — used as the process name
  "cmd": "extract",                    // the qr verb chain, e.g. "convert dicom-series"
  "stage": "features",                 // one of: data, image, features, modeling
  "args": {"manifest": "{outdir}/manifest.csv",
           "pattern": "{pattern}",
           "output": "{outdir}/features.csv"},   // 1:1 with CLI flags (--manifest ...)
  "inputs": ["{outdir}/manifest.csv"], // logical file refs for dependency tracking
  "outputs": ["{outdir}/features.csv"],
  "per_patient": true                  // fan out one task per patient when scaffolded
}
```

Rules that make plans work:

- **`args` keys map 1:1 to CLI long flags.** `"dataset-root": "x"` becomes
  `--dataset-root x`. Get the flag names from `references/cli.md` or `qr <cmd> --help`.
- **`{placeholder}` values resolve against `vars`** at run/scaffold time. Define every
  moving path (`outdir`, `cohort_root`, `pattern`, `roi`, outcome columns) once in
  `vars` and reference it everywhere. `{patient_id}` is the special per-patient token.
- **`stage`** orders the plan along `data → image → features → modeling` and groups it
  in scaffolded output. **`per_patient: true`** is what turns a step into a fan-out
  process (convert + extract are per-patient; manifest/merge/analyze are cohort-wide).
- **`inputs`/`outputs`** are logical declarations for dependency tracking and for the
  runner to `mkdir -p` output parents. Keep them honest — they document the DAG.

## Start from a template, then edit

Don't hand-write a plan from scratch — generate the closest template and mutate it.

```bash
qr workflow templates          # nrrd_survival | dicom_survival | dicom_to_ml | tcia_to_ml
qr workflow plan -t nrrd_survival -d ./cohort -c clinical.csv -o plan.json
qr workflow show plan.json      # inspect
```

`qr workflow plan -t nrrd_survival` emits exactly four steps (verified):

| id | cmd | stage | per_patient |
|---|---|---|---|
| manifest | convert manifest-from-dir | data | no |
| extract | extract | features | yes |
| merge | results merge | features | no |
| analyze | analyze survival | modeling | no |

`dicom_survival` / `dicom_to_ml` prepend per-patient `convert dicom-series` (+ `rtstruct`)
image steps; `tcia_to_ml` prepends `tcia series` + `tcia download` data steps.

**To customize**, edit `plan.json` directly — it's just data. Common edits:

- **Swap the pattern**: set `vars.pattern` to `ct-default`, `nsclc-survival`, etc.
- **Add a shape step**: insert `{"id":"shape","cmd":"shape extract","stage":"features",
  "args":{"manifest":"{outdir}/manifest.csv","output":"{outdir}/shape.csv"},
  "per_patient":true, ...}` after `extract`.
- **Switch to classification**: change the `analyze` step's `cmd` to `analyze classify`,
  drop the `--event` arg, point `--outcome` at your label column.
- **Add an ML train/evaluate tail**: append `ml train` + `ml evaluate` modeling steps.

You can also build a plan in Python when the edit is structural:

```python
from qradiomics.workflow import WorkflowPlan, WorkflowStep, save_plan, load_plan
plan = load_plan("plan.json")
plan.steps.insert(2, WorkflowStep(
    id="shape", cmd="shape extract", stage="features", per_patient=True,
    args={"manifest": "{outdir}/manifest.csv", "output": "{outdir}/shape.csv"},
    inputs=["{outdir}/manifest.csv"], outputs=["{outdir}/shape.csv"]))
save_plan(plan, "plan.json")
```

## Run it

```bash
qr workflow run plan.json --executor inline   --dry-run   # print the exact commands, run nothing
qr workflow run plan.json --executor inline               # small cohort: run steps in-process
qr workflow run plan.json --executor nextflow             # DEFAULT: parallel + cache + HPC
qr workflow run plan.json --executor prefect              # secondary orchestrator
```

**Always `--dry-run` first.** It prints each resolved command as JSON so you can
confirm flags and paths before touching data. The runner invokes steps as
`python -m qradiomics.cli.main <cmd> ...` (robust to PATH), stops on the first
non-zero exit, and `mkdir -p`s output parents. `nextflow`/`prefect` executors need
that tool on PATH (`pip install nextflow` / `curl -fsSL https://get.nextflow.io | bash`;
`pip install qradiomics[prefect]`).

## Scaffold to a file you can commit

`run` executes ephemerally; `scaffold` writes a standalone file you can version, edit,
and hand off:

```bash
qr workflow scaffold -p plan.json -e nextflow -o main.nf         # per-patient DSL2 processes
qr workflow scaffold -p plan.json -e shell    -o pipeline.sh     # plain bash, vars as ${UPPER}
# prefect: qr workflow run --executor prefect scaffolds a flow internally; or scaffold_prefect() in Python
```

The Nextflow scaffold turns each step into a `process p_<id>`, lifts `vars` to
`params.*`, tags per-patient processes with `$pid`, and chains processes by their
declared order. Pair it with a `nextflow.config` that defines `local` / `docker` /
`slurm` profiles (copy `pipelines/lung1/nextflow.config`), then
`nextflow run main.nf -profile local -resume`.

## Build a production pipeline directory

To ship a cohort as a deployable bundle, mirror the layout of `pipelines/lung1/`:

```
pipelines/<cohort>/
├── README.md          — what it does + how to run
├── plan.json          — qr workflow plan (source of truth)
├── main.nf            — scaffolded: qr workflow scaffold -p plan.json -e nextflow
├── nextflow.config    — local / docker / slurm profiles
├── prefect_flow.py    — scaffolded Prefect flow (optional)
├── deploy.sh          — pip install + fetch data + run (one command)
└── clinical/          — clinical CSV stub (user replaces with real outcomes)
```

Recipe:

1. `qr workflow plan -t tcia_to_ml -C "<TCIA collection>" -c clinical/clinical.csv
   --roi <ROI> --pattern <id> --task <survival|classify> --outcome <col> -o plan.json`.
2. Edit `plan.json` `vars` for the cohort (collection, ROI names, glob patterns, column
   names). Confirm with `qr workflow run plan.json --executor inline --dry-run`.
3. `qr workflow scaffold -p plan.json -e nextflow -o main.nf` and copy a `nextflow.config`.
4. Write `deploy.sh` (see `pipelines/lung1/deploy.sh`): install `qradiomics[rtstruct,prefect]`,
   fetch Nextflow if missing, `nextflow run main.nf -profile local -resume`.
5. Add a smoke path — see `references/testing.md`. Register any new pipeline module with a
   `tests/test_<module>.py` synthetic-data test (`AGENTS.md §7`).

Keep the four don'ts in mind throughout: no PHI in logs, no `git add -A`, no push
without confirmation, no invented dataset URLs.
