"""Workflow assembly for qradiomics.

A `WorkflowPlan` is a JSON/YAML-able list of qr atomic-command steps with
declared inputs and outputs. Agents and humans both produce plans
(`qr workflow plan`), optionally edit them, then execute them
(`qr workflow run`) or scaffold them out as a Nextflow / shell file
(`qr workflow scaffold`).

Design notes:
- Atomic-step set is the same as the qr CLI: `convert dicom-series`,
  `convert rtstruct`, `convert manifest-from-dir`, `extract`,
  `results merge`, `analyze survival|classify|importance`.
- Each step has a stable `id`, a `cmd` (the qr verb chain), an `args`
  dict (mapped 1:1 to CLI flags), and explicit `inputs`/`outputs`
  files for dependency tracking. Templates use `{...}` placeholders
  resolved against the workflow-level `vars`.
- Plans are pure data; the runtime is in `WorkflowRunner` (this file)
  and `WorkflowScaffolder` (one method per executor: shell, nextflow).
- No global state — calling code (CLI or library) just passes a
  `WorkflowPlan` dict around.

This module deliberately stays small: an agent should be able to
inspect, mutate, and re-emit a plan without going through any state
machine. The hard part is the templates (`templates.LIBRARY`), which
are the canonical workflow shapes the CLI can scaffold.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List


# ─── Data model ──────────────────────────────────────────────────────────────


#: Canonical radiomics data flow stages.
#:
#:   data      — cohort discovery, manifest, clinical CSV registration
#:   image     — DICOM → NRRD (CT + mask), preprocessing
#:   features  — PyRadiomics feature extraction
#:   modeling  — ML model training, validation, evaluation
STAGES = ("data", "image", "features", "modeling")


@dataclass
class WorkflowStep:
    """One atomic qr-CLI invocation within a workflow plan."""

    id: str
    cmd: str                                    # e.g. "convert dicom-series"
    stage: str = "features"                     # one of STAGES
    args: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)   # logical file refs
    outputs: List[str] = field(default_factory=list)
    per_patient: bool = False                   # split across patients when scaffolded


@dataclass
class WorkflowPlan:
    """A list of WorkflowSteps + cohort/var context."""

    name: str
    description: str = ""
    vars: Dict[str, Any] = field(default_factory=dict)
    steps: List[WorkflowStep] = field(default_factory=list)
    version: str = "0.9"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "vars": self.vars,
            "steps": [asdict(s) for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkflowPlan":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            vars=d.get("vars", {}),
            steps=[WorkflowStep(**s) for s in d.get("steps", [])],
            version=d.get("version", "0.9"),
        )


# ─── Template library ────────────────────────────────────────────────────────
# These produce a default WorkflowPlan for common shapes. Agents start from
# one of these and can add/remove/reorder steps before running.


def template_nrrd_survival(
    cohort_root: str,
    clinical_csv: str,
    image_glob: str = "*_CT.nrrd",
    mask_glob: str = "*-label.nrrd",
    pattern: str = "nsclc-survival",
    time_col: str = "OS_days",
    event_col: str = "OS_event",
    outdir: str = "runs/cohort",
) -> WorkflowPlan:
    """Cohort already in NRRD form. Manifest -> extract -> merge -> survival."""
    return WorkflowPlan(
        name="nrrd_survival",
        description="Build manifest from NRRD tree, extract, merge clinical, run Cox PH.",
        vars={
            "cohort_root": cohort_root,
            "clinical_csv": clinical_csv,
            "outdir": outdir,
            "image_glob": image_glob,
            "mask_glob": mask_glob,
            "pattern": pattern,
            "time_col": time_col,
            "event_col": event_col,
        },
        steps=[
            WorkflowStep(
                id="manifest",
                stage="data",
                cmd="convert manifest-from-dir",
                args={
                    "dataset-root": "{cohort_root}",
                    "image-glob": "{image_glob}",
                    "mask-glob": "{mask_glob}",
                    "output": "{outdir}/manifest.csv",
                },
                outputs=["{outdir}/manifest.csv"],
            ),
            WorkflowStep(
                id="extract",
                stage="features",
                cmd="extract",
                args={
                    "manifest": "{outdir}/manifest.csv",
                    "pattern": "{pattern}",
                    "output": "{outdir}/features.csv",
                },
                inputs=["{outdir}/manifest.csv"],
                outputs=["{outdir}/features.csv"],
                per_patient=True,
            ),
            WorkflowStep(
                id="merge",
                stage="features",
                cmd="results merge",
                args={
                    "features": "{outdir}/features.csv",
                    "clinical": "{clinical_csv}",
                    "clinical-id-col": "patient_id",
                    "time-col": "{time_col}",
                    "event-col": "{event_col}",
                    "output": "{outdir}/analysis_ready.csv",
                },
                inputs=["{outdir}/features.csv", "{clinical_csv}"],
                outputs=["{outdir}/analysis_ready.csv"],
            ),
            WorkflowStep(
                id="analyze",
                stage="modeling",
                cmd="analyze survival",
                args={
                    "input": "{outdir}/analysis_ready.csv",
                    "outcome": "OS_months",
                    "event": "OS_event",
                    "output": "{outdir}/cox_results.csv",
                    "top-n": 20,
                },
                inputs=["{outdir}/analysis_ready.csv"],
                outputs=["{outdir}/cox_results.csv"],
            ),
        ],
    )


def template_dicom_to_ml(
    cohort_root: str,
    clinical_csv: str,
    roi: str = "GTV",
    pattern: str = "nsclc-survival",
    task: str = "survival",          # survival | classify
    outcome: str = "OS_event",
    time_col: str = "OS_days",
    event_col: str = "OS_event",
    outdir: str = "runs/cohort",
) -> WorkflowPlan:
    """Full DICOM → outcome-prediction ML model workflow (4 canonical stages).

    Stages:
      1. data      — cohort discovery, manifest, clinical CSV check
      2. image     — DICOM CT → NRRD + RTSTRUCT → label NRRD (per-patient)
      3. features  — qr extract (per-patient or batch via manifest)
      4. modeling  — qr ml train (CV) + qr ml evaluate

    The plan is structured so the Nextflow scaffolder fans out the
    image and feature-extraction stages per patient automatically.
    """
    return WorkflowPlan(
        name=f"dicom_to_ml_{task}",
        description="DICOM cohort → outcome-prediction ML model (4 stages).",
        vars={
            "cohort_root": cohort_root,
            "clinical_csv": clinical_csv,
            "outdir": outdir,
            "nrrd_dir": f"{outdir}/nrrd",
            "roi": roi,
            "pattern": pattern,
            "task": task,
            "outcome": outcome,
            "time_col": time_col,
            "event_col": event_col,
        },
        steps=[
            # ── Stage 1: data ────────────────────────────────────────────────
            WorkflowStep(
                id="manifest_dicom",
                stage="data",
                cmd="convert manifest-from-dir",
                args={
                    "dataset-root": "{cohort_root}",
                    "image-glob": "**/CT",
                    "mask-glob": "**/RS.*.dcm",
                    "output": "{outdir}/dicom_manifest.csv",
                },
                outputs=["{outdir}/dicom_manifest.csv"],
            ),
            # ── Stage 2: image (DICOM → NRRD, per patient) ──────────────────
            WorkflowStep(
                id="convert_ct",
                stage="image",
                cmd="convert dicom-series",
                args={
                    "input": "{cohort_root}/{patient_id}/CT",
                    "output": "{nrrd_dir}/{patient_id}_CT.nrrd",
                },
                outputs=["{nrrd_dir}/{patient_id}_CT.nrrd"],
                per_patient=True,
            ),
            WorkflowStep(
                id="convert_rt",
                stage="image",
                cmd="convert rtstruct",
                args={
                    "dicom-dir": "{cohort_root}/{patient_id}/CT",
                    "rtstruct": "{cohort_root}/{patient_id}/RTSeries/RS.dcm",
                    "roi": "{roi}",
                    "output": "{nrrd_dir}/{patient_id}_{roi}-label.nrrd",
                },
                inputs=["{nrrd_dir}/{patient_id}_CT.nrrd"],
                outputs=["{nrrd_dir}/{patient_id}_{roi}-label.nrrd"],
                per_patient=True,
            ),
            WorkflowStep(
                id="manifest_nrrd",
                stage="data",
                cmd="convert manifest-from-dir",
                args={
                    "dataset-root": "{nrrd_dir}",
                    "image-glob": "*_CT.nrrd",
                    "mask-glob": "*_{roi}-label.nrrd",
                    "output": "{outdir}/manifest.csv",
                },
                inputs=["{nrrd_dir}"],
                outputs=["{outdir}/manifest.csv"],
            ),
            # ── Stage 3: features ────────────────────────────────────────────
            WorkflowStep(
                id="extract",
                stage="features",
                cmd="extract",
                args={
                    "manifest": "{outdir}/manifest.csv",
                    "pattern": "{pattern}",
                    "output": "{outdir}/features.csv",
                },
                inputs=["{outdir}/manifest.csv"],
                outputs=["{outdir}/features.csv"],
                per_patient=True,
            ),
            WorkflowStep(
                id="merge",
                stage="features",
                cmd="results merge",
                args={
                    "features": "{outdir}/features.csv",
                    "clinical": "{clinical_csv}",
                    "clinical-id-col": "patient_id",
                    "time-col": "{time_col}",
                    "event-col": "{event_col}",
                    "output": "{outdir}/analysis_ready.csv",
                },
                inputs=["{outdir}/features.csv", "{clinical_csv}"],
                outputs=["{outdir}/analysis_ready.csv"],
            ),
            # ── Stage 4: modeling ────────────────────────────────────────────
            WorkflowStep(
                id="train",
                stage="modeling",
                cmd="ml train",
                args={
                    "input": "{outdir}/analysis_ready.csv",
                    "task": "{task}",
                    "outcome": "{outcome}",
                    "model": "{outdir}/model.pkl",
                    "metrics": "{outdir}/cv_metrics.json",
                },
                inputs=["{outdir}/analysis_ready.csv"],
                outputs=["{outdir}/model.pkl", "{outdir}/cv_metrics.json"],
            ),
            WorkflowStep(
                id="evaluate",
                stage="modeling",
                cmd="ml evaluate",
                args={
                    "input": "{outdir}/analysis_ready.csv",
                    "model": "{outdir}/model.pkl",
                    "task": "{task}",
                    "outcome": "{outcome}",
                    "report": "{outdir}/evaluation.json",
                },
                inputs=["{outdir}/analysis_ready.csv", "{outdir}/model.pkl"],
                outputs=["{outdir}/evaluation.json"],
            ),
        ],
    )


def template_dicom_survival(
    cohort_root: str,
    clinical_csv: str,
    roi: str = "GTV",
    pattern: str = "nsclc-survival",
    outdir: str = "runs/cohort",
) -> WorkflowPlan:
    """Cohort ships as DICOM + RTSTRUCT. Adds per-patient convert steps.

    The convert steps are marked per_patient=True so the Nextflow scaffolder
    fans them out across all patient directories.
    """
    plan = template_nrrd_survival(
        cohort_root=cohort_root,
        clinical_csv=clinical_csv,
        image_glob=f"*_CT.nrrd",
        mask_glob=f"*_{roi}-label.nrrd",
        pattern=pattern,
        outdir=outdir,
    )
    plan.name = "dicom_survival"
    plan.description = "DICOM -> NRRD (CT + RTSTRUCT) -> extract -> merge -> survival."
    plan.vars["roi"] = roi
    plan.vars["nrrd_dir"] = "{outdir}/nrrd"
    # Insert DICOM conversion steps before the manifest step.
    convert_steps = [
        WorkflowStep(
            id="convert_ct",
            cmd="convert dicom-series",
            args={
                "input": "{cohort_root}/{patient_id}/CT",
                "output": "{nrrd_dir}/{patient_id}_CT.nrrd",
            },
            outputs=["{nrrd_dir}/{patient_id}_CT.nrrd"],
            per_patient=True,
        ),
        WorkflowStep(
            id="convert_rt",
            cmd="convert rtstruct",
            args={
                "dicom-dir": "{cohort_root}/{patient_id}/CT",
                "rtstruct": "{cohort_root}/{patient_id}/RTSeries/RS.dcm",
                "roi": "{roi}",
                "output": "{nrrd_dir}/{patient_id}_{roi}-label.nrrd",
            },
            inputs=["{nrrd_dir}/{patient_id}_CT.nrrd"],
            outputs=["{nrrd_dir}/{patient_id}_{roi}-label.nrrd"],
            per_patient=True,
        ),
    ]
    # Make the manifest scan the new NRRD dir, not the DICOM tree.
    plan.steps[0].args["dataset-root"] = "{nrrd_dir}"
    plan.steps = convert_steps + plan.steps
    return plan


def template_tcia_to_ml(
    collection: str,
    clinical_csv: str,
    modalities: str = "CT,RTSTRUCT",
    roi: str = "GTV",
    pattern: str = "nsclc-survival",
    task: str = "survival",
    outcome: str = "OS_event",
    time_col: str = "OS_days",
    event_col: str = "OS_event",
    outdir: str = "runs/cohort",
    max_series: int = 0,
) -> WorkflowPlan:
    """Full TCIA → outcome-prediction ML model workflow.

    Adds a `tcia` stage in front of `dicom_to_ml`:
      0. tcia       — qr tcia series + qr tcia download (downloads DICOM)
      1. data       — qr convert manifest-from-dir
      2. image      — qr convert dicom-series + qr convert rtstruct (per-patient)
      3. features   — qr extract + qr results merge
      4. modeling   — qr ml train + qr ml evaluate
    """
    plan = template_dicom_to_ml(
        cohort_root=f"{outdir}/dicom",   # populated by the tcia step
        clinical_csv=clinical_csv,
        roi=roi,
        pattern=pattern,
        task=task,
        outcome=outcome,
        time_col=time_col,
        event_col=event_col,
        outdir=outdir,
    )
    plan.name = f"tcia_to_ml_{task}"
    plan.description = "TCIA download → DICOM → outcome-prediction ML model (5 stages)."
    plan.vars["collection"] = collection
    plan.vars["modalities"] = modalities
    plan.vars["max_series"] = max_series

    tcia_steps = [
        WorkflowStep(
            id="tcia_series",
            stage="data",
            cmd="tcia series",
            args={
                "collection": "{collection}",
                "output": "{outdir}/series.csv",
            },
            outputs=["{outdir}/series.csv"],
        ),
        WorkflowStep(
            id="tcia_download",
            stage="data",
            cmd="tcia download",
            args={
                "manifest": "{outdir}/series.csv",
                "output": "{outdir}/dicom",
                "max-series": "{max_series}",
            },
            inputs=["{outdir}/series.csv"],
            outputs=["{outdir}/dicom"],
        ),
    ]
    plan.steps = tcia_steps + plan.steps
    return plan


LIBRARY = {
    "nrrd_survival": template_nrrd_survival,
    "dicom_survival": template_dicom_survival,
    "dicom_to_ml": template_dicom_to_ml,
    "tcia_to_ml": template_tcia_to_ml,
}


# ─── Resolver ────────────────────────────────────────────────────────────────


def _resolve(value: Any, vars: Dict[str, Any]) -> Any:
    """Substitute {var} placeholders in strings, recursively."""
    if isinstance(value, str):
        out = value
        for k, v in vars.items():
            out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(value, list):
        return [_resolve(v, vars) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, vars) for k, v in value.items()}
    return value


# ─── Runner ──────────────────────────────────────────────────────────────────


class WorkflowRunner:
    """Execute a WorkflowPlan.

    Executor modes:
      inline    — sequential subprocess loop (this Python process)
      nextflow  — scaffold to a temporary .nf + nextflow.config and run
                  `nextflow run ...` (DEFAULT for production / large cohorts)
      prefect   — scaffold to a temporary .py flow and run it via Python
                  (secondary; integrates with Prefect orchestration if a
                  Prefect server is configured in the environment)
    """

    def __init__(self, plan: WorkflowPlan, executor: str = "inline", dry_run: bool = False):
        self.plan = plan
        self.executor = executor
        self.dry_run = dry_run

    def run(self) -> List[Dict[str, Any]]:
        if self.executor == "inline":
            return self._run_inline()
        if self.executor == "nextflow":
            return self._run_nextflow()
        if self.executor == "prefect":
            return self._run_prefect()
        raise ValueError(f"Unknown executor '{self.executor}'")

    def _run_inline(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for step in self.plan.steps:
            cmd = self._build_command(step)
            if self.dry_run:
                results.append({"id": step.id, "cmd": cmd, "status": "dry-run"})
                continue
            for out in step.outputs:
                Path(_resolve(out, self.plan.vars)).parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(cmd, check=False)
            results.append({"id": step.id, "cmd": cmd, "returncode": proc.returncode})
            if proc.returncode != 0:
                break
        return results

    def _run_nextflow(self) -> List[Dict[str, Any]]:
        if shutil.which("nextflow") is None:
            return [{"status": "error", "message": "nextflow not on PATH; pip install nextflow or download from https://nextflow.io"}]
        with tempfile.TemporaryDirectory() as tmp:
            nf = Path(tmp) / "qradiomics.nf"
            nf.write_text(scaffold_nextflow(self.plan))
            cmd = ["nextflow", "run", str(nf)]
            if self.dry_run:
                return [{"status": "dry-run", "cmd": cmd, "scaffold": str(nf)}]
            proc = subprocess.run(cmd, check=False)
            return [{"executor": "nextflow", "cmd": cmd, "returncode": proc.returncode}]

    def _run_prefect(self) -> List[Dict[str, Any]]:
        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp) / "qradiomics_flow.py"
            py.write_text(scaffold_prefect(self.plan))
            cmd = ["python", str(py)]
            if self.dry_run:
                return [{"status": "dry-run", "cmd": cmd, "scaffold": str(py)}]
            proc = subprocess.run(cmd, check=False)
            return [{"executor": "prefect", "cmd": cmd, "returncode": proc.returncode}]

    def _build_command(self, step: WorkflowStep) -> List[str]:
        # Use `python -m qradiomics.cli.main` rather than the bare `qr` shim
        # so the runner is robust to PATH and pre-v0.9 shim leftovers.
        parts: List[str] = [sys.executable, "-m", "qradiomics.cli.main"] + step.cmd.split()
        for k, v in step.args.items():
            parts.append(f"--{k}")
            parts.append(str(_resolve(v, self.plan.vars)))
        return parts


# ─── Scaffolders ─────────────────────────────────────────────────────────────


def scaffold_shell(plan: WorkflowPlan) -> str:
    """Render the plan as a bash script that calls qr in sequence."""
    lines = [
        "#!/usr/bin/env bash",
        f"# {plan.name} — generated by `qr workflow scaffold`",
        f"# {plan.description}",
        "set -euo pipefail",
        "",
    ]
    for k, v in plan.vars.items():
        lines.append(f'{k.upper()}="${{{k.upper()}:-{v}}}"')
    lines.append("")
    for step in plan.steps:
        cmd = ["qr"] + step.cmd.split()
        for k, v in step.args.items():
            cmd.append(f"--{k}")
            # turn {cohort_root} into ${COHORT_ROOT}
            v_str = str(v)
            for var in plan.vars:
                v_str = v_str.replace("{" + var + "}", "${" + var.upper() + "}")
            cmd.append(f'"{v_str}"')
        lines.append(f"# step: {step.id}")
        lines.append(" \\\n    ".join(cmd))
        lines.append("")
    return "\n".join(lines)


def scaffold_prefect(plan: WorkflowPlan) -> str:
    """Render the plan as a Prefect 2.x flow (secondary executor).

    Each WorkflowStep becomes a `@task`. Per-patient steps are mapped over
    a patient list discovered from `params.cohort_root`. The generated
    flow can be invoked via `python <file>.py` or registered with a
    Prefect deployment for scheduled runs.
    """
    lines = [
        f"# {plan.name} — generated by `qr workflow scaffold --executor prefect`",
        f"# {plan.description}",
        "from pathlib import Path",
        "from subprocess import run",
        "from prefect import flow, task",
        "",
        "PARAMS = {",
    ]
    for k, v in plan.vars.items():
        if isinstance(v, str):
            lines.append(f'    "{k}": "{v}",')
        else:
            lines.append(f'    "{k}": {v!r},')
    lines.append("}")
    lines.append("")
    lines.append("def _format(value):")
    lines.append("    \"\"\"Substitute {var} placeholders in a string.\"\"\"")
    lines.append("    if isinstance(value, str):")
    lines.append("        for k, v in PARAMS.items():")
    lines.append("            value = value.replace('{' + k + '}', str(v))")
    lines.append("    return value")
    lines.append("")

    task_names: List[str] = []
    for step in plan.steps:
        tname = "t_" + step.id
        task_names.append(tname)
        sig = "(patient_id: str)" if step.per_patient else "()"
        lines.append("@task")
        lines.append(f"def {tname}{sig}:")
        cmd_parts = ['"qr"'] + [f'"{p}"' for p in step.cmd.split()]
        for k, v in step.args.items():
            cmd_parts.append(f'"--{k}"')
            v_str = str(v)
            if step.per_patient and "{patient_id}" in v_str:
                v_str = v_str.replace("{patient_id}", "{pid}")
                # Use f-string so we can inject pid; PARAMS substitution handled by _format
                cmd_parts.append(f'_format(f"{v_str}".replace("{{pid}}", patient_id))')
            else:
                cmd_parts.append(f'_format("{v_str}")')
        cmd = "cmd = [" + ", ".join(cmd_parts) + "]"
        lines.append("    " + cmd)
        lines.append("    Path(_format('{outdir}')).mkdir(parents=True, exist_ok=True)")
        lines.append("    run(cmd, check=True)")
        lines.append("")

    lines.append("@flow")
    lines.append("def workflow():")
    if any(s.per_patient for s in plan.steps):
        lines.append("    cohort = Path(_format('{cohort_root}'))")
        lines.append("    patients = sorted(p.name for p in cohort.iterdir() if p.is_dir())")
    for step in plan.steps:
        tname = "t_" + step.id
        if step.per_patient:
            lines.append(f"    {tname}.map(patients)")
        else:
            lines.append(f"    {tname}()")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    workflow()")
    return "\n".join(lines)


def scaffold_nextflow(plan: WorkflowPlan) -> str:
    """Render the plan as a Nextflow DSL2 workflow.

    Generates `done_*` placeholder channels so each step depends on the
    previous step's completion. Per-patient steps are fanned out across
    a `patients` channel that is populated once the upstream stage emits
    a `done` signal.
    """
    lines = [
        "// " + plan.name + " — generated by `qr workflow scaffold`",
        "// " + plan.description,
        "nextflow.enable.dsl = 2",
        "",
    ]
    for k, v in plan.vars.items():
        if isinstance(v, str):
            lines.append(f'params.{k} = "{v}"')
        else:
            lines.append(f"params.{k} = {v}")
    lines.append("")

    def _render_args(step: WorkflowStep) -> List[str]:
        out: List[str] = []
        for k, v in step.args.items():
            v_str = str(v)
            for var in plan.vars:
                v_str = v_str.replace("{" + var + "}", "${params." + var + "}")
            if step.per_patient and "{patient_id}" in v_str:
                v_str = v_str.replace("{patient_id}", "${pid}")
            out.append(f"--{k} {v_str}")
        return out

    for step in plan.steps:
        pname = "p_" + step.id
        lines.append(f"process {pname} {{")
        if step.per_patient:
            lines.append('    tag "$pid"')
            lines.append("    input:")
            lines.append("    tuple val(pid), val(_ready)")
            lines.append("    output:")
            lines.append("    tuple val(pid), val(true), emit: done")
        else:
            lines.append("    input:")
            lines.append("    val(_ready)")
            lines.append("    output:")
            lines.append("    val(true), emit: done")
        lines.append("")
        lines.append("    script:")
        lines.append('    """')
        cmd_line = "qr " + step.cmd + " \\\n        " + " \\\n        ".join(_render_args(step))
        lines.append("    " + cmd_line)
        lines.append('    """')
        lines.append("}")
        lines.append("")

    lines.append("workflow {")
    lines.append("    // Channel of patient IDs; emits once upstream has populated cohort_root")
    lines.append(
        '    patients = Channel.value(1).map {'
        ' new File("${params.cohort_root}").listFiles().findAll { d -> d.isDirectory() }.collect { d -> d.name }'
        ' }.flatten()'
    )
    lines.append("    ready = Channel.value(true)")
    lines.append("")
    prev_done = "ready"
    for step in plan.steps:
        pname = "p_" + step.id
        if step.per_patient:
            lines.append(f"    {pname}_in = patients.combine({prev_done})")
            lines.append(f"    {pname}({pname}_in)")
            lines.append(f"    {pname}_done = {pname}.out.done.collect().map {{ true }}")
            prev_done = f"{pname}_done"
        else:
            lines.append(f"    {pname}({prev_done})")
            prev_done = f"{pname}.out.done"
    lines.append("}")
    return "\n".join(lines)


SCAFFOLDERS = {
    "shell": scaffold_shell,
    "nextflow": scaffold_nextflow,    # default for production / large cohorts
    "prefect": scaffold_prefect,      # secondary, scheduling / observability
}


# ─── Helpers ─────────────────────────────────────────────────────────────────


def load_plan(path: Path) -> WorkflowPlan:
    text = path.read_text()
    if path.suffix in (".yml", ".yaml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return WorkflowPlan.from_dict(data)


def save_plan(plan: WorkflowPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in (".yml", ".yaml"):
        import yaml

        path.write_text(yaml.safe_dump(plan.to_dict(), sort_keys=False))
    else:
        path.write_text(json.dumps(plan.to_dict(), indent=2))
