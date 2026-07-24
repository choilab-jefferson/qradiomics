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
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


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
                stage="image",
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
    convert_jobs: int = 4,
    extract_jobs: int = 4,
) -> WorkflowPlan:
    """Full TCIA → outcome-prediction ML model workflow (8 steps).

    The 8 steps map cleanly onto the canonical 4-stage architecture:

    \b
      data:      tcia_series  →  series.csv
                 tcia_download →  DICOM tree under {outdir}/dicom
                 tcia_manifest →  (pid, modality, CT-dir, RTSTRUCT-file) CSV
      image:     convert_from_manifest → per-patient NRRDs + nrrd manifest CSV
      features:  extract       →  features.csv
                 merge         →  analysis_ready.csv
      modeling:  train         →  model.pkl + cv_metrics.json
                 evaluate      →  evaluation.json

    Why not reuse template_dicom_to_ml: TCIA's UID-based DICOM tree
    (`<patient>/<study-uid>/<series-uid>/<n>.dcm`) makes glob-based
    classification impossible (`**/CT` matches nothing). We use the
    Modality column from `qr tcia series`'s output instead, then convert
    every (image_dir, rtstruct) row in one batch — the two `qr tcia
    manifest` + `qr convert from-manifest` primitives.
    """
    return WorkflowPlan(
        name=f"tcia_to_ml_{task}",
        description="TCIA → outcome-prediction ML model (8 steps, 4 canonical stages).",
        vars={
            "outdir": outdir,
            "nrrd_dir": f"{outdir}/nrrd",
            "clinical_csv": clinical_csv,
            "collection": collection,
            "modalities": modalities,
            "max_series": max_series,
            "convert_jobs": convert_jobs,
            "extract_jobs": extract_jobs,
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
            WorkflowStep(
                id="tcia_manifest",
                stage="data",
                cmd="tcia manifest",
                args={
                    "series": "{outdir}/series.csv",
                    "dicom-root": "{outdir}/dicom",
                    "output": "{outdir}/dicom_manifest.csv",
                },
                inputs=["{outdir}/series.csv", "{outdir}/dicom"],
                outputs=["{outdir}/dicom_manifest.csv"],
            ),
            # ── Stage 2: image (DICOM → NRRD, batched over the manifest) ─────
            WorkflowStep(
                id="convert_from_manifest",
                stage="image",
                cmd="convert from-manifest",
                args={
                    "manifest": "{outdir}/dicom_manifest.csv",
                    "nrrd-dir": "{nrrd_dir}",
                    "roi": "{roi}",
                    "output": "{outdir}/manifest.csv",
                    "jobs": "{convert_jobs}",
                },
                inputs=["{outdir}/dicom_manifest.csv"],
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
                    "jobs": "{extract_jobs}",
                },
                inputs=["{outdir}/manifest.csv"],
                outputs=["{outdir}/features.csv"],
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


LIBRARY = {
    "nrrd_survival": template_nrrd_survival,
    "dicom_survival": template_dicom_survival,
    "dicom_to_ml": template_dicom_to_ml,
    "tcia_to_ml": template_tcia_to_ml,
}


# ─── Resolver ────────────────────────────────────────────────────────────────


def _resolve(value: Any, vars: Dict[str, Any],
             patient_id: Optional[str] = None) -> Any:
    """Substitute {var} placeholders in strings, recursively.

    When `patient_id` is given, `{patient_id}` is also substituted — used by
    the per-patient inline executor to expand a single step into one
    invocation per patient.
    """
    if isinstance(value, str):
        out = value
        if patient_id is not None:
            out = out.replace("{patient_id}", patient_id)
        for k, v in vars.items():
            out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(value, list):
        return [_resolve(v, vars, patient_id=patient_id) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, vars, patient_id=patient_id) for k, v in value.items()}
    return value


def _list_cohort_patients(cohort_root: Optional[str]) -> List[str]:
    """Discover patient subdirectories under a cohort root. Returns [] when
    the path is unset or doesn't exist, so callers can skip per-patient steps
    cleanly without exception handling.
    """
    if not cohort_root:
        return []
    root = Path(str(cohort_root))
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


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
        """Sequential by-step execution; per-patient steps fan out across a
        bounded thread pool (size = QR_INLINE_WORKERS, default 4).

        Stays usable when Nextflow / Prefect are absent — the trade-off is
        bounded parallelism: only per-patient steps within a single stage are
        concurrent, never across stages. For real production scale, run with
        --executor nextflow instead.
        """
        results: List[Dict[str, Any]] = []
        workers = max(1, int(os.environ.get("QR_INLINE_WORKERS", "4")))
        for step in self.plan.steps:
            if step.per_patient:
                patients = _list_cohort_patients(self.plan.vars.get("cohort_root"))
                if not patients:
                    results.append({"id": step.id, "skipped": "no patients found",
                                    "cohort_root": self.plan.vars.get("cohort_root")})
                    continue
                per_pid_results = self._run_per_patient(step, patients, workers)
                results.extend(per_pid_results)
                if any(r.get("returncode", 0) != 0 for r in per_pid_results):
                    break
            else:
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

    def _run_per_patient(self, step: WorkflowStep, patients: List[str],
                         workers: int) -> List[Dict[str, Any]]:
        """Fan a per-patient step across `workers` threads, one task per id."""
        out: List[Dict[str, Any]] = []
        if self.dry_run:
            for pid in patients:
                out.append({"id": step.id, "patient_id": pid,
                            "cmd": self._build_command(step, patient_id=pid),
                            "status": "dry-run"})
            return out

        def _run_one(pid: str) -> Dict[str, Any]:
            cmd = self._build_command(step, patient_id=pid)
            for o in step.outputs:
                Path(_resolve(o, self.plan.vars, patient_id=pid)).parent.mkdir(
                    parents=True, exist_ok=True
                )
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            return {
                "id": step.id, "patient_id": pid, "cmd": cmd,
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-500:] if proc.returncode else "",
            }

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_one, pid): pid for pid in patients}
            for fut in as_completed(futures):
                out.append(fut.result())
        return out

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

    def _build_command(self, step: WorkflowStep,
                       patient_id: Optional[str] = None) -> List[str]:
        # Use `python -m qradiomics.cli.main` rather than the bare `qr` shim
        # so the runner is robust to PATH and pre-v0.9 shim leftovers.
        parts: List[str] = [sys.executable, "-m", "qradiomics.cli.main"] + step.cmd.split()
        for k, v in step.args.items():
            parts.append(f"--{k}")
            parts.append(str(_resolve(v, self.plan.vars, patient_id=patient_id)))
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
    """Render the plan as a Prefect 2.x flow with a three-level hierarchy.

    The generated artefact is::

        @flow cohort_flow(...)              # top — one per cohort run
            ↓ calls
        @flow phase_a_features(...)         # subflow — Phase A (data/image/features)
            ↓ calls
            @task stage_data() | stage_image() | stage_features()

        @flow phase_b_modeling(...)         # subflow — Phase B (modeling)
            ↓ calls
            @task stage_modeling()

    Why this shape: the two-phase architecture (see
    `wiki/architecture/NEXTFLOW_REACTIVATION.md` §2) splits cohort runs at
    the `analysis_ready.csv` boundary. Surfacing that split as two Prefect
    subflows makes the UI show a clean parent → 2 subflows → ≤ 4 tasks tree
    instead of a flat task list. Each subflow can be re-run independently
    from the UI ("Phase B only" with a fresh model on the same features).

    Coarse remains coarse: there is still exactly one `@task` per pipeline
    stage, never per patient. Per-patient fan-out happens inside the task
    via a thread pool; Nextflow / sklearn handle the real parallelism.

    Canonical artifacts auto-registered as Prefect link artifacts when they
    exist after a stage completes::

        data      → series.csv, dicom_manifest.csv
        image     → manifest.csv
        features  → features.csv
        modeling  → analysis_ready.csv, model.pkl, cv_metrics.json,
                    evaluation.json
    """
    # Group steps by stage. We require stage labels to form contiguous runs
    # in declaration order — interleaved stages (e.g. data→image→data) would
    # silently reorder execution when we collapse them into stage tasks, which
    # would break dependencies. If you hit this, either re-label the offending
    # step or split it out into its own stage.
    stages_order: List[str] = []
    by_stage: Dict[str, List[WorkflowStep]] = {}
    seen_stages: set = set()
    last_stage: str | None = None
    for step in plan.steps:
        if step.stage != last_stage:
            if step.stage in seen_stages:
                raise ValueError(
                    f"scaffold_prefect: stage '{step.stage}' is interleaved with "
                    f"other stages — step '{step.id}' returns to stage '{step.stage}' "
                    f"after intervening stages. Re-label the step or split the plan."
                )
            stages_order.append(step.stage)
            by_stage[step.stage] = []
            seen_stages.add(step.stage)
            last_stage = step.stage
        by_stage[step.stage].append(step)

    # Stage → canonical artifacts to register if they exist after the stage
    # task completes. Paths are formatted with PARAMS at runtime.
    canonical_artifacts = {
        "data": ["{outdir}/series.csv", "{outdir}/dicom_manifest.csv"],
        "image": ["{outdir}/manifest.csv"],
        "features": ["{outdir}/features.csv"],
        "modeling": [
            "{outdir}/analysis_ready.csv",
            "{outdir}/model.pkl",
            "{outdir}/cv_metrics.json",
            "{outdir}/evaluation.json",
        ],
    }

    lines: List[str] = [
        f"# {plan.name} — generated by `qr workflow scaffold --executor prefect`",
        f"# {plan.description}",
        "#",
        "# Coarse stage-level Prefect flow. One @task per pipeline stage; per-",
        "# patient fan-out happens inside the task via a thread pool so the",
        "# Prefect UI stays readable. Prefect tracks WHEN/WHERE; qr does WHAT.",
        "import os",
        "from concurrent.futures import ThreadPoolExecutor, as_completed",
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
    lines.append("MAX_WORKERS = int(os.environ.get('QR_PREFECT_WORKERS', '4'))")
    lines.append("")
    lines.append("def _format(value, pid=None):")
    lines.append('    """Substitute {var} placeholders. {patient_id} → pid when given."""')
    lines.append("    if isinstance(value, str):")
    lines.append("        if pid is not None:")
    lines.append("            value = value.replace('{patient_id}', pid)")
    lines.append("        for k, v in PARAMS.items():")
    lines.append("            value = value.replace('{' + k + '}', str(v))")
    lines.append("    return value")
    lines.append("")
    lines.append("def _patients():")
    lines.append("    cohort = Path(_format('{cohort_root}'))")
    lines.append("    if not cohort.is_dir():")
    lines.append("        return []")
    lines.append("    return sorted(p.name for p in cohort.iterdir() if p.is_dir())")
    lines.append("")
    lines.append("def _register_artifacts(stage):")
    lines.append('    """Attach canonical artifacts to the active Prefect run."""')
    lines.append("    try:")
    lines.append("        from prefect.artifacts import create_link_artifact")
    lines.append("    except ImportError:")
    lines.append("        return")
    lines.append("    for path_tmpl in _STAGE_ARTIFACTS.get(stage, []):")
    lines.append("        p = Path(_format(path_tmpl))")
    lines.append("        if p.exists():")
    lines.append("            try:")
    lines.append("                # Path.as_uri() requires an absolute path; the plan vars use")
    lines.append("                # relative paths (e.g. `runs/<cohort>/...`). Resolve against")
    lines.append("                # the worker's CWD before turning into a file:// URI.")
    lines.append("                create_link_artifact(")
    lines.append("                    key=f'{stage}-{p.name.replace(\".\", \"-\")}',")
    lines.append("                    link=p.resolve().as_uri(),")
    lines.append("                    description=f'{stage} stage artifact: {p.name}',")
    lines.append("                )")
    lines.append("            except Exception:")
    lines.append("                # Never let an observability hiccup kill a real stage.")
    lines.append("                pass")
    lines.append("")
    lines.append("_STAGE_ARTIFACTS = {")
    for stage, arts in canonical_artifacts.items():
        if stage in by_stage:
            arts_repr = ", ".join(repr(a) for a in arts)
            lines.append(f'    "{stage}": [{arts_repr}],')
    lines.append("}")
    lines.append("")

    # Emit the bash command builder for a single step, used by stage tasks.
    lines.append("def _run_step(cmd_str, args, pid=None):")
    lines.append('    """Run a single qr step. cmd_str is the qr verb chain; args is a dict."""')
    lines.append("    cmd = ['qr'] + cmd_str.split()")
    lines.append("    for k, v in args.items():")
    lines.append("        cmd.append(f'--{k}')")
    lines.append("        cmd.append(_format(str(v), pid=pid))")
    lines.append("    Path(_format('{outdir}')).mkdir(parents=True, exist_ok=True)")
    lines.append("    run(cmd, check=True)")
    lines.append("")

    # Emit one @task per stage.
    stage_task_names: List[str] = []
    for stage in stages_order:
        steps_in_stage = by_stage[stage]
        tname = f"stage_{stage}"
        stage_task_names.append(tname)
        lines.append("@task")
        lines.append(f"def {tname}():")
        lines.append(f'    """Stage `{stage}` — {len(steps_in_stage)} step(s):'
                     f' {", ".join(s.id for s in steps_in_stage)}."""')
        for step in steps_in_stage:
            args_repr = "{" + ", ".join(f'"{k}": {v!r}' for k, v in step.args.items()) + "}"
            if step.per_patient:
                lines.append(f"    # Per-patient step `{step.id}`: parallel across cohort")
                lines.append("    pats = _patients()")
                lines.append("    if pats:")
                lines.append("        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:")
                lines.append(
                    f"            futures = [ex.submit(_run_step, {step.cmd!r}, {args_repr}, pid=p)"
                    f" for p in pats]"
                )
                lines.append("            for f in as_completed(futures):")
                lines.append("                f.result()  # propagate exceptions")
            else:
                lines.append(f"    # Step `{step.id}`")
                lines.append(f"    _run_step({step.cmd!r}, {args_repr})")
        if stage in canonical_artifacts:
            lines.append(f'    _register_artifacts("{stage}")')
        lines.append("")

    # ─── Phase classification ───────────────────────────────────────────────
    # Phase A = data + image + features (the Nextflow-suited side).
    # Phase B = modeling (the MLflow/sklearn side).
    # Any other stage label falls under Phase A by default — keep the
    # classification side-effect-free so callers can extend stages later.
    phase_b_stages = {"modeling"}
    phase_a_task_names = [n for n, s in zip(stage_task_names, stages_order)
                          if s not in phase_b_stages]
    phase_b_task_names = [n for n, s in zip(stage_task_names, stages_order)
                          if s in phase_b_stages]

    # ─── Subflow: Phase A (features) ────────────────────────────────────────
    if phase_a_task_names:
        lines.append("@flow(name='phase_a_features')")
        lines.append("def phase_a_features():")
        lines.append('    """Phase A — PACS/TCIA → features. Nextflow-suited, file-level fan-out."""')
        for tname in phase_a_task_names:
            lines.append(f"    {tname}()")
        lines.append("")

    # ─── Subflow: Phase B (modeling) ────────────────────────────────────────
    if phase_b_task_names:
        lines.append("@flow(name='phase_b_modeling')")
        lines.append("def phase_b_modeling():")
        lines.append('    """Phase B — analysis_ready.csv → trained model. MLflow-suited."""')
        for tname in phase_b_task_names:
            lines.append(f"    {tname}()")
        lines.append("")

    # ─── Top-level @flow that calls the subflows in order ──────────────────
    lines.append("@flow(name=" + repr(plan.name) + ")")
    lines.append("def workflow():")
    lines.append('    """Cohort entry point — wires Phase A → Phase B."""')
    if phase_a_task_names:
        lines.append("    phase_a_features()")
    if phase_b_task_names:
        lines.append("    phase_b_modeling()")
    if not phase_a_task_names and not phase_b_task_names:
        lines.append("    pass  # empty plan; nothing to run")
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

    # Helper closure: list patient subdirectories of cohort_root. Defined at
    # script scope so the workflow body can reference it. The closure is only
    # evaluated when invoked inside a per-patient flatMap, by which point the
    # upstream `done` signal has fired and cohort_root is populated.
    lines.append("def _list_patients() {")
    lines.append('    def d = new File("${params.cohort_root}")')
    lines.append("    return d.isDirectory()")
    lines.append("        ? d.listFiles().findAll { f -> f.isDirectory() }.collect { f -> f.name }")
    lines.append("        : []")
    lines.append("}")
    lines.append("")
    lines.append("workflow {")
    lines.append("    ready = Channel.value(true)")
    lines.append("")
    prev_done = "ready"
    for step in plan.steps:
        pname = "p_" + step.id
        if step.per_patient:
            # Lazy patients fan-out: list cohort_root *after* the upstream done
            # signal arrives. The earlier `Channel.value(1).map { File.listFiles() }`
            # pattern eagerly evaluated at workflow definition time, silently
            # producing an empty channel when cohort_root was populated by an
            # upstream NF process — every per-patient step then ran zero times
            # while NF reported SUCCESS.
            lines.append(
                f"    {pname}_in = {prev_done}.flatMap {{ it -> _list_patients() }}"
                f".map {{ pid -> tuple(pid, true) }}"
            )
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
