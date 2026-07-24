"""qr workflow — agent-friendly workflow assembly + execution."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

from qradiomics.workflow import (
    LIBRARY,
    SCAFFOLDERS,
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
@click.option("--roi", default="GTV", help="RTSTRUCT ROI name (DICOM templates)")
@click.option("--pattern", default="nsclc-survival", help="qr pattern id")
@click.option("--task", default="survival", help="ml task (survival | classify)")
@click.option("--outcome", default="OS_event", help="Outcome column name")
@click.option("--max-series", default=0, type=int,
              help="Limit TCIA download to N series (tcia_to_ml only; 0 = no limit)")
@click.option("--outdir", default="runs/cohort", help="Workflow output directory")
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Where to write the plan (JSON or YAML by extension)",
)
def plan_cmd(template, cohort_root, collection, clinical, roi, pattern, task, outcome,
             max_series, outdir, output):
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
    if "roi" in sig:
        kwargs["roi"] = roi
    if "task" in sig:
        kwargs["task"] = task
    if "outcome" in sig:
        kwargs["outcome"] = outcome
    if "max_series" in sig:
        kwargs["max_series"] = max_series

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
    default="nextflow",
    help="Execution backend (default nextflow for production; inline for "
         "small interactive runs; prefect for scheduled / observed runs)",
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
def run_cmd(plan_path, executor, dry_run, workers):
    """Execute a workflow plan.

    \b
    The default executor is Nextflow — it parallelises per-patient steps,
    caches successful processes, and is the right choice for large
    cohorts. Use --executor inline for small interactive runs (still
    parallel across patients via a thread pool when nextflow isn't
    available), or --executor prefect for orchestration with a Prefect
    server.
    """
    if workers is not None:
        os.environ["QR_INLINE_WORKERS"] = str(workers)
    plan = load_plan(Path(plan_path))
    runner = WorkflowRunner(plan, executor=executor, dry_run=dry_run)
    results = runner.run()
    for r in results:
        click.echo(json.dumps(r))
    failed = [r for r in results if r.get("returncode", 0) != 0]
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
        marker = "[per-patient]" if step.per_patient else ""
        click.echo(f"  {i}. {step.id} — qr {step.cmd}  {marker}")
        for k, v in step.args.items():
            click.echo(f"       --{k} {v}")
