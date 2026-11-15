"""qr workflow — agent-friendly workflow assembly + execution."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from qradiomics.workflow import (
    LIBRARY,
    SCAFFOLDERS,
    NextflowNotFoundError,
    WorkflowRunner,
    load_plan,
    save_plan,
)


@click.group()
def workflow():
    """Assemble, scaffold, and run qr-based workflows.

    \b
    A workflow plan is a JSON/YAML file listing qr atomic-command steps.
    The qr CLI is both the atomic-task runner AND the workflow runner —
    agents compose plans from the template library, mutate them, and
    either execute inline or scaffold them out to Nextflow / shell.
    """


@workflow.command("templates")
def list_templates():
    """List available workflow templates."""
    for name, fn in LIBRARY.items():
        doc = (fn.__doc__ or "").splitlines()[0] if fn.__doc__ else ""
        click.echo(f"  {name:20s}  {doc}")


@workflow.command("plan")
@click.option(
    "--template",
    "-t",
    type=click.Choice(sorted(LIBRARY.keys())),
    required=True,
    help="Template name (see 'qr workflow templates')",
)
@click.option(
    "--cohort-root",
    "-d",
    type=click.Path(file_okay=False),
    help="Cohort root directory (required for nrrd_survival / dicom_survival / dicom_to_ml)",
)
@click.option(
    "--collection",
    help="TCIA collection name (required for tcia_to_ml)",
)
@click.option(
    "--clinical",
    "-c",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    help="Clinical CSV (required for survival / classify / ml analyses)",
)
@click.option(
    "--pair-manifest",
    type=click.Path(exists=True, dir_okay=False),
    help="Pair manifest (required for rt_pair_radiomics; see 'qr select validate')",
)
@click.option("--roi", default="GTV", help="RTSTRUCT ROI name (DICOM templates)")
@click.option("--pattern", default="nsclc-survival", help="qr pattern id")
@click.option("--task", default="survival", help="ml task (survival | classify)")
@click.option("--outcome", default="OS_event", help="Outcome column name")
@click.option(
    "--max-series",
    default=0,
    type=int,
    help="Limit TCIA download to N series (tcia_to_ml only; 0 = no limit)",
)
@click.option(
    "--engine",
    default=None,
    help="Extraction engine(s) for the 'extract' step: rtools, pysera, "
    "pyradiomics, a comma-list, or 'all'. Default: the pattern's configured "
    "extractor (rtools if it names none).",
)
@click.option("--outdir", default="runs/cohort", help="Workflow output directory")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Where to write the plan (JSON or YAML by extension)",
)
def plan_cmd(
    template,
    cohort_root,
    collection,
    clinical,
    pair_manifest,
    roi,
    pattern,
    task,
    outcome,
    max_series,
    engine,
    outdir,
    output,
):
    """Generate a workflow plan from a template."""
    fn = LIBRARY[template]
    sig = fn.__code__.co_varnames[: fn.__code__.co_argcount]

    kwargs: dict = {"outdir": outdir, "pattern": pattern}
    if "cohort_root" in sig:
        if not cohort_root:
            raise click.UsageError(f"Template '{template}' requires --cohort-root")
        # Resolve to absolute so plans stay valid when consumed from a different
        # working directory (Prefect workers, Nextflow runs, …).
        kwargs["cohort_root"] = str(Path(cohort_root).resolve())
    if "collection" in sig:
        if not collection:
            raise click.UsageError(f"Template '{template}' requires --collection")
        kwargs["collection"] = collection
    if "clinical_csv" in sig:
        if not clinical:
            raise click.UsageError(f"Template '{template}' requires --clinical")
        # Same: absolutize so a Prefect worker / Nextflow run resolves the
        # path the same way the human did when running `qr workflow plan`.
        kwargs["clinical_csv"] = str(Path(clinical).resolve())
    if "pair_manifest" in sig:
        if not pair_manifest:
            raise click.UsageError(f"Template '{template}' requires --pair-manifest")
        kwargs["pair_manifest"] = str(Path(pair_manifest).resolve())
    if "roi" in sig:
        kwargs["roi"] = roi
    if "task" in sig:
        kwargs["task"] = task
    if "outcome" in sig:
        kwargs["outcome"] = outcome
    if "max_series" in sig:
        kwargs["max_series"] = max_series
    if "engine" in sig and engine:
        kwargs["engine"] = engine

    plan = fn(**kwargs)
    save_plan(plan, Path(output))
    click.echo(f"Wrote plan ({len(plan.steps)} steps) -> {output}")


@workflow.command("scaffold")
@click.option(
    "--plan",
    "-p",
    "plan_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Plan JSON / YAML produced by 'qr workflow plan'",
)
@click.option(
    "--executor",
    "-e",
    type=click.Choice(sorted(SCAFFOLDERS.keys())),
    default="shell",
    help="Executor to scaffold (shell or nextflow)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Where to write the scaffolded file",
)
def scaffold_cmd(plan_path, executor, output):
    """Render a workflow plan as a shell script or Nextflow workflow."""
    plan = load_plan(Path(plan_path))
    text = SCAFFOLDERS[executor](plan)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    if executor == "shell":
        out.chmod(0o755)
    click.echo(f"Scaffolded {executor} -> {output}")


@workflow.command("run")
@click.argument("plan_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--executor",
    "-e",
    type=click.Choice(["inline", "nextflow", "prefect"]),
    default="inline",
    show_default=True,
    help="Execution backend (inline for local runs; nextflow for bulk runs "
    "with per-case fan-out and -resume; prefect for scheduled / observed runs)",
)
@click.option("--dry-run", is_flag=True, help="Print commands without executing")
@click.option(
    "--workers",
    "-j",
    type=int,
    default=None,
    help="Per-patient parallelism for --executor inline (default: $QR_INLINE_WORKERS"
    " or 4). Ignored by the nextflow / prefect executors (they set their own"
    " concurrency).",
)
@click.option(
    "--manifest",
    "-m",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Cohort manifest (.csv/.parquet; case_id or patient_id column, or a pair"
    " manifest) giving the case list for inline and nextflow alike. Required for"
    " fanout=pair steps. Default: the plan's case_manifest var (e.g. tcia manifest"
    " output) when present, else patient sub-directories of the plan's cohort_root.",
)
@click.option(
    "--profile",
    "nf_profile",
    default="local",
    show_default=True,
    help="[nextflow] Nextflow profile (local, slurm, docker, singularity; comma-combine).",
)
@click.option(
    "--run-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="[nextflow] Run directory holding params.json, work/ and .nextflow/"
    " (default: <plan outdir>/nf_run). Reuse it to resume.",
)
@click.option(
    "--on-error",
    type=click.Choice(["finish", "ignore", "terminate", "retry"]),
    default="finish",
    show_default=True,
    help="What a failed case/pair does. 'ignore' drops it and merges the rest"
    " (the coverage table lists what is missing); 'finish' lets running tasks end"
    " then stops. inline honours finish|ignore only.",
)
@click.option(
    "--max-forks",
    type=int,
    default=0,
    show_default=True,
    help="[nextflow] Max parallel tasks per process (0 = nextflow default).",
)
@click.option(
    "--container",
    default=None,
    help="[nextflow] Image pinned by digest (repo/img@sha256:<64 hex>); runs every"
    " task in it (docker unless --profile names singularity/apptainer).",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="[nextflow] Reuse cached tasks of a previous run in --run-dir. Resume"
    " keys on the commands + manifest content + qradiomics version, NOT on"
    " image/mask file contents: after re-segmentation delete the run dir or"
    " pass --no-resume. Auto-disabled (with a warning) when a declared cohort"
    " output is missing.",
)
def run_cmd(
    plan_path,
    executor,
    dry_run,
    workers,
    manifest,
    nf_profile,
    run_dir,
    on_error,
    max_forks,
    container,
    resume,
):
    """Execute a workflow plan.

    \b
    The default executor is inline: a local subprocess loop, per-patient
    steps in a thread pool. Use --executor nextflow for bulk runs: one
    Nextflow task per case, cached by -resume, local or slurm profile.
    Use --executor prefect for orchestration with a Prefect server.
    """
    if workers is not None:
        os.environ["QR_INLINE_WORKERS"] = str(workers)
    plan = load_plan(Path(plan_path))
    runner = WorkflowRunner(
        plan,
        executor=executor,
        dry_run=dry_run,
        manifest=manifest,
        run_dir=run_dir,
        profile=nf_profile,
        resume=resume,
        on_error=on_error,
        max_forks=max_forks,
        container=container,
    )
    try:
        results = runner.run()
    except (NextflowNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))
    for r in results:
        click.echo(json.dumps(r))
    failed = [r for r in results if r.get("returncode", 0) != 0]
    if on_error == "ignore" and executor == "inline":
        # same contract as nextflow errorStrategy=ignore: failed cases/pairs are
        # dropped and reported, the run still succeeds. A failed cohort-level step
        # (no patient_id) is never ignorable.
        ignored = [r for r in failed if "patient_id" in r]
        for r in ignored:
            click.echo(
                f"ignored failure: {r['id']} {r.get('pair_id') or r['patient_id']}", err=True
            )
        failed = [r for r in failed if "patient_id" not in r]
    if failed:
        raise SystemExit(failed[0]["returncode"])


@workflow.command("show")
@click.argument("plan_path", type=click.Path(exists=True, dir_okay=False))
def show_cmd(plan_path):
    """Print a workflow plan's steps in a compact form."""
    plan = load_plan(Path(plan_path))
    click.echo(f"# {plan.name}: {plan.description}")
    click.echo(f"  vars: {plan.vars}")
    for i, step in enumerate(plan.steps, 1):
        marker = f"[{step.fanout}]" if step.fanout else ""
        click.echo(f"  {i}. {step.id} — qr {step.cmd}  {marker}")
        for k, v in step.args.items():
            click.echo(f"       --{k} {v}")


@workflow.command("prov")
@click.option("--pair-id", required=True)
@click.option("--file", "files", multiple=True, help="Declared output file (repeatable).")
@click.option("--container", default=None, help="Pinned container ref to record.")
@click.option("--output", "-o", required=True, type=click.Path(), help="<pair>.prov.json path.")
def prov_cmd(pair_id, files, container, output):
    """Write a pair provenance record (output sha256 + tool versions). Called by nf/main.nf."""
    from qradiomics.workflow import write_provenance

    try:
        write_provenance(pair_id, files, output, container=container)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    click.echo(f"provenance -> {output}")
