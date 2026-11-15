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
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

# ─── Data model ──────────────────────────────────────────────────────────────


#: Canonical radiomics data flow stages.
#:
#:   data      — cohort discovery, manifest, clinical CSV registration
#:   image     — DICOM → NRRD (CT + mask), preprocessing
#:   features  — PyRadiomics feature extraction
#:   modeling  — ML model training, validation, evaluation
STAGES = ("data", "image", "features", "modeling")

#: valid WorkflowStep.fanout values
FANOUTS = (None, "case", "pair")


@dataclass
class WorkflowStep:
    """One atomic qr-CLI invocation within a workflow plan."""

    id: str
    cmd: str  # e.g. "convert dicom-series"
    stage: str = "features"  # one of STAGES
    args: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)  # logical file refs
    outputs: List[str] = field(default_factory=list)
    #: fan-out unit: None = once per cohort, "case" = once per case id,
    #: "pair" = once per (image, label) pair of the pair manifest.
    fanout: Optional[str] = None
    #: legacy alias of fanout == "case"; kept in sync in both directions
    per_patient: Optional[bool] = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "fanout":
            if value not in FANOUTS:
                raise ValueError(f"step {self.id!r}: fanout {value!r} not in {FANOUTS}")
            object.__setattr__(self, "fanout", value)
            if "per_patient" in self.__dict__:
                object.__setattr__(self, "per_patient", value == "case")
        elif name == "per_patient":
            if value is None:  # constructor default: follow fanout
                value = self.fanout == "case"
            elif value and self.fanout not in (None, "case"):
                raise ValueError(
                    f"step {self.id!r}: per_patient=True conflicts with fanout={self.fanout!r}"
                )
            elif value:
                object.__setattr__(self, "fanout", "case")
            elif self.fanout == "case":
                object.__setattr__(self, "fanout", None)
            object.__setattr__(self, "per_patient", bool(value))
        else:
            object.__setattr__(self, name, value)


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


def _plan_engine(engine: Optional[str], pattern: str) -> str:
    """Engine spec for a plan's extract step.

    An explicit ``engine`` wins. Otherwise follow the pattern's configured
    extractor (shipped patterns name pyradiomics; a pattern that names none
    gets the native rtools default), falling back to ``DEFAULT_ENGINE`` when
    the pattern cannot be loaded.
    """
    from qradiomics.extractor_registry import DEFAULT_ENGINE

    if engine:
        return engine
    try:
        from qradiomics.pattern_loader import PatternLoader

        tmpl = PatternLoader().get_pattern(pattern)
        if tmpl is not None:
            return tmpl.feature_extraction.engine_spec
    except Exception:  # noqa: BLE001 - plan generation must not fail on pattern lookup
        log.debug("could not load pattern %r for engine default", pattern, exc_info=True)
    return DEFAULT_ENGINE


CASE_FEATURES = "{outdir}/cases/{patient_id}/features.csv"
CASE_FEATURES_GLOB = "{outdir}/cases/*/features.csv"


def _fan_out_extract(plan: WorkflowPlan, case_manifest: str) -> WorkflowPlan:
    """Make `extract` per-case (`--case {patient_id}`, one features.csv per
    case) and point `merge` (cohort-level, runs after every case) at the
    per-case outputs. Cases come from `case_manifest` (a plan var)."""
    steps = {st.id: st for st in plan.steps}
    ex, mg = steps["extract"], steps["merge"]
    ex.per_patient = True
    ex.args.pop("jobs", None)
    ex.args["case"] = "{patient_id}"
    ex.args["output"] = CASE_FEATURES
    ex.outputs = [CASE_FEATURES]
    mg.args["features"] = CASE_FEATURES_GLOB
    mg.inputs = [CASE_FEATURES_GLOB] + mg.inputs[1:]
    plan.vars["case_manifest"] = case_manifest
    return plan


def template_nrrd_survival(
    cohort_root: str,
    clinical_csv: str,
    image_glob: str = "*_CT.nrrd",
    mask_glob: str = "*-label.nrrd",
    pattern: str = "nsclc-survival",
    time_col: str = "OS_days",
    event_col: str = "OS_event",
    outdir: str = "runs/cohort",
    engine: Optional[str] = None,
) -> WorkflowPlan:
    """Cohort already in NRRD form. Manifest -> per-case extract -> merge -> survival."""
    plan = _nrrd_survival_base(**locals())
    return _fan_out_extract(plan, "{outdir}/manifest.csv")


def _nrrd_survival_base(
    cohort_root: str,
    clinical_csv: str,
    image_glob: str = "*_CT.nrrd",
    mask_glob: str = "*-label.nrrd",
    pattern: str = "nsclc-survival",
    time_col: str = "OS_days",
    event_col: str = "OS_event",
    outdir: str = "runs/cohort",
    engine: Optional[str] = None,
) -> WorkflowPlan:
    """Shared body of the NRRD templates; extract is cohort-level here."""
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
            "engine": _plan_engine(engine, pattern),
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
                    "engine": "{engine}",
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
    task: str = "survival",  # survival | classify
    outcome: str = "OS_event",
    time_col: str = "OS_days",
    event_col: str = "OS_event",
    outdir: str = "runs/cohort",
    engine: Optional[str] = None,
) -> WorkflowPlan:
    """Full DICOM → outcome-prediction ML model workflow (4 canonical stages).

    Stages:
      1. data      — cohort discovery, manifest, clinical CSV check
      2. image     — DICOM CT → NRRD + RTSTRUCT → label NRRD (per-patient)
      3. features  — qr extract (per-patient or batch via manifest)
      4. modeling  — qr ml train (CV) + qr ml evaluate

    The convert steps are per-patient; extract reads the whole-cohort
    manifest and is therefore a cohort-level step.
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
            "engine": _plan_engine(engine, pattern),
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
                    "engine": "{engine}",
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


def template_dicom_survival(
    cohort_root: str,
    clinical_csv: str,
    roi: str = "GTV",
    pattern: str = "nsclc-survival",
    outdir: str = "runs/cohort",
    engine: Optional[str] = None,
) -> WorkflowPlan:
    """Cohort ships as DICOM + RTSTRUCT. Adds per-patient convert steps.

    The convert steps are marked per_patient=True so the Nextflow scaffolder
    fans them out across all patient directories.
    """
    plan = _nrrd_survival_base(
        cohort_root=cohort_root,
        clinical_csv=clinical_csv,
        image_glob=f"*_CT.nrrd",
        mask_glob=f"*_{roi}-label.nrrd",
        pattern=pattern,
        outdir=outdir,
        engine=engine,
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


def build_workflow_plan(
    extraction: Any,
    per_engine: bool = False,
    template: str = "dicom_to_ml",
    **template_kwargs: Any,
) -> WorkflowPlan:
    """Build a plan whose features stage follows a FeatureExtractionConfig.

    ``extraction`` is a ``pattern_loader.FeatureExtractionConfig`` (anything
    exposing ``.engines``). By default the single ``extract`` step gets
    ``--engine a,b`` for every configured engine. With ``per_engine=True``
    and more than one engine, the step is split into one ``extract_<engine>``
    step per engine (each writing ``features_<engine>.csv``) followed by a
    ``fuse`` step (``qr results fuse``) that rebuilds ``features.csv``.
    """
    factories = {
        "dicom_to_ml": template_dicom_to_ml,
        "dicom_survival": template_dicom_survival,
        "nrrd_survival": template_nrrd_survival,
    }
    if template not in factories:
        raise ValueError(f"Unknown template '{template}'. Available: {sorted(factories)}")
    engines = tuple(extraction.engines)
    plan = factories[template](engine=",".join(engines), **template_kwargs)
    if not per_engine or len(engines) < 2:
        return plan

    idx = next(i for i, st in enumerate(plan.steps) if st.id == "extract")
    base = plan.steps[idx]
    split = []
    # A per-case extract keeps per-engine + fused outputs inside the case dir.
    out_dir = "{outdir}/cases/{patient_id}" if base.per_patient else "{outdir}"
    for eng in engines:
        out = f"{out_dir}/features_{eng}.csv"
        split.append(
            WorkflowStep(
                id=f"extract_{eng}",
                stage=base.stage,
                cmd=base.cmd,
                args={**base.args, "engine": eng, "output": out},
                inputs=list(base.inputs),
                outputs=[out],
                fanout=base.fanout,
            )
        )
    fuse = WorkflowStep(
        id="fuse",
        stage=base.stage,
        cmd="results fuse",
        args={
            "features": ",".join(f"{eng}={out_dir}/features_{eng}.csv" for eng in engines),
            "output": base.args["output"],
        },
        inputs=[f"{out_dir}/features_{eng}.csv" for eng in engines],
        outputs=list(base.outputs),
        per_patient=base.per_patient,
    )
    plan.steps[idx : idx + 1] = split + [fuse]
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
    engine: Optional[str] = None,
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
    Modality column from `qr tcia series`'s output instead (`qr tcia
    manifest`), then run `qr convert from-manifest --case` and `qr extract
    --case` once per case (per_patient), so Nextflow fans them out. Cases
    are the patient_id values of {outdir}/dicom_manifest.csv.
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
            "case_manifest": "{outdir}/dicom_manifest.csv",
            "roi": roi,
            "pattern": pattern,
            "task": task,
            "outcome": outcome,
            "time_col": time_col,
            "event_col": event_col,
            "engine": _plan_engine(engine, pattern),
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
            # ── Stage 2: image (DICOM → NRRD, one task per case) ─────────────
            WorkflowStep(
                id="convert_from_manifest",
                stage="image",
                cmd="convert from-manifest",
                args={
                    "manifest": "{outdir}/dicom_manifest.csv",
                    "nrrd-dir": "{nrrd_dir}",
                    "roi": "{roi}",
                    "case": "{patient_id}",
                    "output": "{outdir}/cases/{patient_id}/manifest.csv",
                },
                inputs=["{outdir}/dicom_manifest.csv"],
                outputs=["{outdir}/cases/{patient_id}/manifest.csv"],
                per_patient=True,
            ),
            # ── Stage 3: features (extract per case, merge once) ────────────
            WorkflowStep(
                id="extract",
                stage="features",
                cmd="extract",
                args={
                    "manifest": "{outdir}/cases/{patient_id}/manifest.csv",
                    "pattern": "{pattern}",
                    "case": "{patient_id}",
                    "output": CASE_FEATURES,
                    "engine": "{engine}",
                },
                inputs=["{outdir}/cases/{patient_id}/manifest.csv"],
                outputs=[CASE_FEATURES],
                per_patient=True,
            ),
            WorkflowStep(
                id="merge",
                stage="features",
                cmd="results merge",
                args={
                    "features": CASE_FEATURES_GLOB,
                    "clinical": "{clinical_csv}",
                    "clinical-id-col": "patient_id",
                    "time-col": "{time_col}",
                    "event-col": "{event_col}",
                    "output": "{outdir}/analysis_ready.csv",
                },
                inputs=[CASE_FEATURES_GLOB, "{clinical_csv}"],
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


def template_rt_pair_radiomics(
    pair_manifest: str,
    clinical_csv: str,
    pattern: str = "nsclc-survival",
    outdir: str = "runs/pairs",
    engine: Optional[str] = None,
    pad_mm: float = 20.0,
    time_col: str = "OS_days",
    event_col: str = "OS_event",
) -> WorkflowPlan:
    """RT pair radiomics: pair manifest -> per-pair features -> merged table + coverage.

    Reads a pair manifest (qr select validate; one row per image x label pair).
    cohort : validate the manifest.
    case   : DICOM series -> NRRD image per (case, image_key).
    pair   : RTSTRUCT -> label NRRD (ROI fallback over roi_candidates), crop +
             per-row resample, extract features -- one task per pair, so one bad
             pair never blocks the rest (--on-error ignore).
    cohort : merge per-pair features with clinical data, write the coverage table.
    """
    return WorkflowPlan(
        name="rt_pair_radiomics",
        description="Pair manifest -> per-pair radiomics -> merged features + coverage.",
        vars={
            "pair_manifest": pair_manifest,
            "clinical_csv": clinical_csv,
            "outdir": outdir,
            "nrrd_dir": f"{outdir}/nrrd",
            "pattern": pattern,
            "engine": _plan_engine(engine, pattern),
            "pad_mm": pad_mm,
            "time_col": time_col,
            "event_col": event_col,
        },
        steps=[
            WorkflowStep(
                id="validate",
                stage="data",
                cmd="select validate",
                args={"manifest": "{pair_manifest}"},
                inputs=["{pair_manifest}"],
            ),
            WorkflowStep(
                id="convert_image",
                stage="image",
                cmd="convert from-manifest",
                args={
                    "manifest": "{pair_manifest}",
                    "only-case": "{case_id}",
                    "part": "image",
                    "nrrd-dir": "{nrrd_dir}",
                    "output": "{outdir}/cases/{case_id}/image_manifest.parquet",
                },
                inputs=["{pair_manifest}"],
                outputs=["{outdir}/cases/{case_id}/image_manifest.parquet"],
                fanout="case",
            ),
            WorkflowStep(
                id="convert_label",
                stage="image",
                cmd="convert from-manifest",
                args={
                    "manifest": "{pair_manifest}",
                    "only": "{pair_id}",
                    "part": "label",
                    "nrrd-dir": "{nrrd_dir}",
                    "output": "{outdir}/pairs/{pair_id}/nrrd_manifest.parquet",
                },
                inputs=["{pair_manifest}"],
                outputs=["{outdir}/pairs/{pair_id}/nrrd_manifest.parquet"],
                fanout="pair",
            ),
            WorkflowStep(
                id="preprocess",
                stage="image",
                cmd="preprocess",
                args={
                    "manifest": "{outdir}/pairs/{pair_id}/nrrd_manifest.parquet",
                    "only": "{pair_id}",
                    "output": "{outdir}/pairs/{pair_id}/pre",
                    "pad-mm": "{pad_mm}",
                    "out-manifest": "{outdir}/pairs/{pair_id}/pre_manifest.parquet",
                },
                inputs=["{outdir}/pairs/{pair_id}/nrrd_manifest.parquet"],
                outputs=["{outdir}/pairs/{pair_id}/pre_manifest.parquet"],
                fanout="pair",
            ),
            WorkflowStep(
                id="extract",
                stage="features",
                cmd="extract",
                args={
                    "manifest": "{outdir}/pairs/{pair_id}/pre_manifest.parquet",
                    "only": "{pair_id}",
                    "pattern": "{pattern}",
                    "engine": "{engine}",
                    "output": "{outdir}/pairs/{pair_id}/features.csv",
                },
                inputs=["{outdir}/pairs/{pair_id}/pre_manifest.parquet"],
                outputs=["{outdir}/pairs/{pair_id}/features.csv"],
                fanout="pair",
            ),
            WorkflowStep(
                id="merge",
                stage="features",
                cmd="results merge",
                args={
                    "features": "{outdir}/pairs/*/features.csv",
                    "pairs": "{pair_manifest}",
                    "coverage": "{outdir}/coverage.parquet",
                    "clinical": "{clinical_csv}",
                    "clinical-id-col": "patient_id",
                    "time-col": "{time_col}",
                    "event-col": "{event_col}",
                    "output": "{outdir}/analysis_ready.csv",
                },
                inputs=["{clinical_csv}"],
                outputs=["{outdir}/analysis_ready.csv", "{outdir}/coverage.parquet"],
            ),
        ],
    )


LIBRARY = {
    "nrrd_survival": template_nrrd_survival,
    "dicom_survival": template_dicom_survival,
    "dicom_to_ml": template_dicom_to_ml,
    "tcia_to_ml": template_tcia_to_ml,
    "rt_pair_radiomics": template_rt_pair_radiomics,
}


# ─── Resolver ────────────────────────────────────────────────────────────────


def _resolve(
    value: Any,
    vars: Dict[str, Any],
    patient_id: Optional[str] = None,
    pair_id: Optional[str] = None,
) -> Any:
    """Substitute {var} placeholders in strings, recursively.

    `patient_id` also fills `{patient_id}` and its alias `{case_id}`; `pair_id`
    fills `{pair_id}`. Used by the executors to expand a fan-out step into one
    invocation per case / pair.
    """
    if isinstance(value, str):
        out = value
        # Iterate so vars that reference other vars ({nrrd_dir} -> {outdir}/nrrd)
        # fully resolve; bounded to stop on self-referencing vars.
        for _ in range(8):
            prev = out
            if patient_id is not None:
                out = out.replace("{patient_id}", patient_id).replace("{case_id}", patient_id)
            if pair_id is not None:
                out = out.replace("{pair_id}", pair_id)
            for k, v in vars.items():
                out = out.replace("{" + k + "}", str(v))
            if out == prev:
                break
        return out
    if isinstance(value, list):
        return [_resolve(v, vars, patient_id=patient_id, pair_id=pair_id) for v in value]
    if isinstance(value, dict):
        return {
            k: _resolve(v, vars, patient_id=patient_id, pair_id=pair_id) for k, v in value.items()
        }
    return value


_CASE_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")


def _list_cohort_patients(cohort_root: Optional[str], exclude: Sequence[Any] = ()) -> List[str]:
    """Discover patient subdirectories under a cohort root. Returns [] when
    the path is unset or doesn't exist, so callers can skip per-patient steps
    cleanly without exception handling. Directories listed in `exclude`
    (e.g. the run dir / outdir when they sit under the cohort root) are skipped.
    """
    if not cohort_root:
        return []
    root = Path(str(cohort_root))
    if not root.is_dir():
        return []
    skip = {Path(str(e)).resolve() for e in exclude if e}
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.resolve() not in skip)


def _check_ids(ids: Sequence[str], what: str = "case") -> None:
    bad = [c for c in ids if not _CASE_ID_RE.fullmatch(c)]
    if bad:
        raise ValueError(
            f"unsafe {what} id(s) {bad[:5]}: ids must match [A-Za-z0-9_.-]+ "
            "(rename the directories or fix the manifest id column)"
        )


def resolve_cases(
    plan: "WorkflowPlan", manifest: Optional[str] = None, exclude: Sequence[Any] = ()
) -> List[str]:
    """Case list shared by every executor: the manifest's case_id (else
    patient_id) column when given, else the plan's `case_manifest` var when that
    file exists (written by a pre step, e.g. tcia manifest), else cohort_root
    sub-directories. For a pair manifest only cases that own at least one
    status=ok pair are returned. Ids are validated (same rule as nf/main.nf)
    before anything launches."""
    cm = plan.vars.get("case_manifest")
    cm = str(_resolve(cm, plan.vars)) if cm else None
    if manifest:
        cases = read_case_ids(manifest)
    elif cm and Path(cm).is_file():
        cases = read_case_ids(cm)
    else:
        cases = _list_cohort_patients(plan.vars.get("cohort_root"), exclude)
    _check_ids(cases)
    return cases


def resolve_pairs(manifest: str):
    """Validated status=ok rows of a pair manifest (DataFrame, file order).
    Raises PairManifestError / ValueError on a bad contract or unsafe ids."""
    from qradiomics import pairs as pm

    df = pm.ok_pairs(pm.load_pairs(manifest))
    _check_ids(df["case_id"].tolist())
    _check_ids(df["pair_id"].tolist(), "pair")
    return df


# ─── Runner ──────────────────────────────────────────────────────────────────


ON_ERROR = ("finish", "ignore", "terminate", "retry")


class WorkflowRunner:
    """Execute a WorkflowPlan.

    Executor modes:
      inline    — sequential subprocess loop (this Python process); DEFAULT
      nextflow  — run the packaged generic qradiomics/nf/main.nf over the
                  cohort (per-case fan-out, -resume caching); see
                  plan_to_nextflow_params / run_plan_nextflow
      prefect   — scaffold to a temporary .py flow and run it via Python
                  (secondary; integrates with Prefect orchestration if a
                  Prefect server is configured in the environment)
    """

    def __init__(
        self,
        plan: WorkflowPlan,
        executor: str = "inline",
        dry_run: bool = False,
        manifest: Optional[str] = None,
        run_dir: Optional[str] = None,
        profile: str = "local",
        resume: bool = True,
        on_error: str = "finish",
        max_forks: int = 0,
        container: Optional[str] = None,
    ):
        if on_error not in ON_ERROR:
            raise ValueError(f"on_error {on_error!r} not in {ON_ERROR}")
        self.plan = plan
        self.executor = executor
        self.dry_run = dry_run
        # Only used by executor="nextflow".
        self.manifest = manifest
        self.run_dir = run_dir
        self.profile = profile
        self.resume = resume
        self.max_forks = max_forks
        self.container = container
        # inline: "ignore" drops a failed case/pair and carries on with the rest.
        self.on_error = on_error
        # Inside a container the host interpreter path does not exist.
        self.python = "python" if container else sys.executable

    def run(self) -> List[Dict[str, Any]]:
        if self.executor == "inline":
            return self._run_inline()
        if self.executor == "nextflow":
            return self._run_nextflow()
        if self.executor == "prefect":
            return self._run_prefect()
        raise ValueError(f"Unknown executor '{self.executor}'")

    def _run_inline(self) -> List[Dict[str, Any]]:
        """Sequential by-step execution; case / pair fan-out steps run across a
        bounded thread pool (size = QR_INLINE_WORKERS, default 4).

        Stays usable when Nextflow / Prefect are absent — the trade-off is
        bounded parallelism: only fan-out tasks within a single step are
        concurrent, never across steps. For real production scale, run with
        --executor nextflow instead. With on_error="ignore" a failed case/pair
        is dropped from later fan-out steps and the run continues; otherwise
        the first failure stops the run.
        """
        results: List[Dict[str, Any]] = []
        workers = max(1, int(os.environ.get("QR_INLINE_WORKERS", "4")))
        patients: Optional[List[str]] = None
        pair_rows: Optional[List[Any]] = None  # [(case_id, pair_id)]
        failed_cases: set = set()
        failed_pairs: set = set()
        for step in self.plan.steps:
            if step.fanout is None:
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
                continue
            if step.fanout == "case":
                if patients is None:
                    patients = resolve_cases(
                        self.plan,
                        self.manifest,
                        exclude=[self.plan.vars.get("outdir"), self.run_dir],
                    )
                tasks = [(p, None) for p in patients]
                desc = "no patients found"
            else:
                if pair_rows is None:
                    pair_rows = self._pair_tasks()
                tasks = [(c, p) for c, p in pair_rows]
                desc = "no pairs found"
            if not tasks:
                results.append(
                    {
                        "id": step.id,
                        "skipped": desc,
                        "cohort_root": self.plan.vars.get("cohort_root"),
                    }
                )
                continue
            tasks = [t for t in tasks if t[0] not in failed_cases and t[1] not in failed_pairs]
            res = self._run_tasks(step, tasks, workers)
            results.extend(res)
            bad = [r for r in res if r.get("returncode", 0) != 0]
            for r in bad:
                (failed_pairs if r.get("pair_id") else failed_cases).add(
                    r.get("pair_id") or r.get("patient_id")
                )
            if bad and self.on_error != "ignore":
                break
        return results

    def _pair_tasks(self) -> List[Any]:
        if not self.manifest:
            raise ValueError("pair fan-out needs --manifest (a pair manifest)")
        df = resolve_pairs(self.manifest)
        return list(zip(df["case_id"], df["pair_id"]))

    def _run_per_patient(
        self, step: WorkflowStep, patients: List[str], workers: int
    ) -> List[Dict[str, Any]]:
        """Fan a per-case step across `workers` threads, one task per id."""
        return self._run_tasks(step, [(p, None) for p in patients], workers)

    def _run_tasks(
        self, step: WorkflowStep, tasks: List[Any], workers: int
    ) -> List[Dict[str, Any]]:
        """Run `step` once per (case_id, pair_id|None) task across `workers` threads."""

        def info(pid, pair):
            d = {"id": step.id, "patient_id": pid}
            if pair is not None:
                d["pair_id"] = pair
            return d

        out: List[Dict[str, Any]] = []
        if self.dry_run:
            for pid, pair in tasks:
                out.append(
                    {
                        **info(pid, pair),
                        "cmd": self._build_command(step, patient_id=pid, pair_id=pair),
                        "status": "dry-run",
                    }
                )
            return out

        def _run_one(task) -> Dict[str, Any]:
            pid, pair = task
            cmd = self._build_command(step, patient_id=pid, pair_id=pair)
            for o in step.outputs:
                Path(_resolve(o, self.plan.vars, patient_id=pid, pair_id=pair)).parent.mkdir(
                    parents=True, exist_ok=True
                )
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            return {
                **info(pid, pair),
                "cmd": cmd,
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-500:] if proc.returncode else "",
            }

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_one, t): t for t in tasks}
            for fut in as_completed(futures):
                out.append(fut.result())
        return out

    def _run_nextflow(self) -> List[Dict[str, Any]]:
        """Run the plan through the generic nf/main.nf (see run_plan_nextflow)."""
        return [
            run_plan_nextflow(
                self.plan,
                manifest=self.manifest,
                run_dir=self.run_dir,
                profile=self.profile,
                resume=self.resume,
                dry_run=self.dry_run,
                on_error=self.on_error,
                max_forks=self.max_forks,
                container=self.container,
            )
        ]

    def _run_prefect(self) -> List[Dict[str, Any]]:
        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp) / "qradiomics_flow.py"
            py.write_text(scaffold_prefect(self.plan))
            cmd = ["python", str(py)]
            if self.dry_run:
                return [{"status": "dry-run", "cmd": cmd, "scaffold": str(py)}]
            proc = subprocess.run(cmd, check=False)
            return [{"executor": "prefect", "cmd": cmd, "returncode": proc.returncode}]

    def _build_command(
        self, step: WorkflowStep, patient_id: Optional[str] = None, pair_id: Optional[str] = None
    ) -> List[str]:
        # Use `python -m qradiomics.cli.main` rather than the bare `qr` shim
        # so the runner is robust to PATH and pre-v0.9 shim leftovers.
        parts: List[str] = [self.python, "-m", "qradiomics.cli.main"] + step.cmd.split()
        for k, v in step.args.items():
            parts.append(f"--{k}")
            parts.append(str(_resolve(v, self.plan.vars, patient_id=patient_id, pair_id=pair_id)))
        return parts


# ─── Nextflow backend ────────────────────────────────────────────────────────


class NextflowNotFoundError(RuntimeError):
    """Raised when the `nextflow` binary is not on PATH."""


def nextflow_main_path() -> Path:
    """Path of the packaged generic pipeline (qradiomics/nf/main.nf)."""
    return Path(__file__).resolve().parent / "nf" / "main.nf"


def read_case_ids(manifest: str) -> List[str]:
    """Unique case ids of a manifest (case_id column, else patient_id), in file
    order. A pair manifest (has pair_id) yields only cases with a status=ok pair."""
    from qradiomics import pairs as pm

    df = pm.read_manifest(manifest)
    if pm.is_pair_manifest(df) and "status" in df.columns:
        df = df[df["status"] == "ok"]
    col = "case_id" if "case_id" in df.columns else "patient_id"
    if col not in df.columns:
        raise ValueError(f"manifest {manifest} has no case_id / patient_id column")
    return list(dict.fromkeys(v for v in df[col] if v))


def _step_script(
    runner: "WorkflowRunner",
    step: WorkflowStep,
    pid: Optional[str],
    pair_id: Optional[str] = None,
) -> List[str]:
    """Shell lines for one step: mkdir of output parents, then the qr command."""
    import shlex

    vars_ = runner.plan.vars
    lines = [
        "mkdir -p "
        + shlex.quote(str(Path(_resolve(o, vars_, patient_id=pid, pair_id=pair_id)).parent))
        for o in step.outputs
    ]
    lines.append(shlex.join(runner._build_command(step, patient_id=pid, pair_id=pair_id)))
    return lines


def _sha(*parts: str) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


_DIGEST_RE = re.compile(r"@(sha256:[0-9a-f]{64})$")


def container_digest(container: Optional[str]) -> str:
    """'sha256:<hex>' of a digest-pinned image ref (``repo/img@sha256:<64 hex>``);
    '' when no container. A tag-only ref raises: tags move, so -resume and the
    provenance record could silently describe a different image."""
    if not container:
        return ""
    m = _DIGEST_RE.search(container)
    if not m:
        raise ValueError(
            f"container {container!r} is not pinned by digest; use repo/image@sha256:<64 hex>"
        )
    return m.group(1)


def tool_versions() -> Dict[str, str]:
    """Versions that change numeric output: qradiomics, pyradiomics, pysera ('absent' if missing)."""
    from importlib import metadata

    from qradiomics import __version__

    out = {"qradiomics": __version__}
    for pkg in ("pyradiomics", "pysera"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "absent"
    return out


def spec_hash(plan: WorkflowPlan) -> str:
    """Hash of the extraction pattern(s) a plan names (their content is not in the
    command line, so editing a pattern must still invalidate -resume)."""
    from qradiomics import PatternLoader

    ids = sorted(
        {str(_resolve(st.args["pattern"], plan.vars)) for st in plan.steps if "pattern" in st.args}
    )
    loader = PatternLoader() if ids else None
    parts = []
    for pid in ids:
        tmpl = loader.get_pattern(pid)
        parts.append(f"{pid}={tmpl.model_dump_json() if tmpl is not None else 'missing'}")
    return _sha(*parts)


def input_fingerprint(
    manifest: Optional[str] = None,
    *,
    container: Optional[str] = None,
    spec_hash: Optional[str] = None,
    series_uid: Optional[str] = None,
    rtstruct_sha: Optional[str] = None,
    extra: Sequence[str] = (),
) -> str:
    """sha256 fingerprint that Nextflow processes take as an input, so -resume
    invalidates when any of these change:

    - qradiomics / pyradiomics / pysera versions
    - container digest (``sha256:...``, or the pinned image ref) and extraction spec hash
    - manifest file content -- except a pair manifest, whose per-row identity
      (series_uid, rtstruct_sha) is hashed per case/pair instead, so editing one
      pair does not invalidate the whole cohort
    - series_uid / rtstruct_sha of one pair, plus any `extra` strings (commands)

    It does NOT cover image / mask file contents."""
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(tool_versions().items()):
        h.update(f"{k}={v}\0".encode())
    for tag, val in (
        ("container", container),
        ("spec", spec_hash),
        ("series_uid", series_uid),
        ("rtstruct_sha", rtstruct_sha),
    ):
        h.update(f"{tag}={val or ''}\0".encode())
    for e in extra:
        h.update(e.encode() + b"\0")
    if manifest:
        from qradiomics import pairs as pm

        if not pm.is_pair_manifest(pm.read_manifest(manifest)):
            h.update(Path(manifest).read_bytes())
    return h.hexdigest()[:16]


def _cohort_outputs(plan: WorkflowPlan) -> List[str]:
    """Resolved declared outputs of the cohort-level (non fan-out) steps."""
    return [
        str(_resolve(o, plan.vars)) for st in plan.steps if st.fanout is None for o in st.outputs
    ]


def _split_regions(plan: WorkflowPlan):
    """(pre, case_steps, pair_steps, post). Order must be cohort* case* pair* cohort*."""
    fan = [i for i, st in enumerate(plan.steps) if st.fanout]
    if not fan:
        return list(plan.steps), [], [], []
    first, last = fan[0], fan[-1]
    region = plan.steps[first : last + 1]
    if any(st.fanout is None for st in region):
        raise ValueError(
            "cohort-level step between per-patient steps; "
            "reorder the plan so fan-out steps are contiguous"
        )
    seen_pair = False
    for st in region:
        if st.fanout == "pair":
            seen_pair = True
        elif seen_pair:
            raise ValueError("case-level step after a pair-level step; order case steps first")
    return (
        plan.steps[:first],
        [st for st in region if st.fanout == "case"],
        [st for st in region if st.fanout == "pair"],
        plan.steps[last + 1 :],
    )


def plan_to_nextflow_params(
    plan: WorkflowPlan,
    cases: List[str],
    outdir: str,
    manifest: Optional[str] = None,
    pairs: Any = None,
    on_error: str = "finish",
    max_forks: int = 0,
    container: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a plan into the params of the generic nf/main.nf.

    Steps before the first fan-out step run once (`pre`); fanout="case" steps
    run once per case id (`cases[].cmds`); fanout="pair" steps run once per
    status=ok pair of that case (`cases[].pairs[].cmds`, `pairs` = the DataFrame
    from resolve_pairs); the remaining steps run once after everything (`post`).
    Plan order must be cohort* case* pair* cohort*; anything else raises
    ValueError. Every case/pair carries `fp`, a fingerprint of its commands plus
    the identity (series_uid / rtstruct_sha) it depends on.
    """
    import shlex

    if on_error not in ON_ERROR:
        raise ValueError(f"on_error {on_error!r} not in {ON_ERROR}")
    runner = WorkflowRunner(plan, container=container)
    pre_steps, case_steps, pair_steps, post_steps = _split_regions(plan)
    fanned = bool(case_steps or pair_steps)
    if fanned and not cases:
        raise ValueError("plan has per-patient steps but the case list is empty")
    if pair_steps and pairs is None:
        raise ValueError("plan has pair-level steps but no pair rows (pass a pair --manifest)")
    sh = spec_hash(plan)
    cdig = container_digest(container)
    pin = container or ""

    def flat(steps: List[WorkflowStep], pid: Optional[str] = None, pair: Optional[str] = None):
        return [line for st in steps for line in _step_script(runner, st, pid, pair)]

    case_entries = []
    for pid in cases if fanned else []:
        rows = pairs[pairs["case_id"] == pid] if pairs is not None else None
        pair_entries = []
        for r in rows.to_dict("records") if rows is not None else []:
            if not pair_steps:
                break
            pair_id = r["pair_id"]
            cmds = flat(pair_steps, pid, pair_id)
            outs = [
                str(_resolve(o, plan.vars, patient_id=pid, pair_id=pair_id))
                for st in pair_steps
                for o in st.outputs
            ]
            prov = (
                shlex.join(
                    [
                        runner.python,
                        "-m",
                        "qradiomics.cli.main",
                        "workflow",
                        "prov",
                        "--pair-id",
                        pair_id,
                    ]
                    + [x for o in outs for x in ("--file", o)]
                    + (["--container", pin] if pin else [])
                )
                + f' --output "$WD/{pair_id}.prov.json"'
            )
            cmds.append(prov)
            pair_entries.append(
                {
                    "id": pair_id,
                    "cmds": cmds,
                    "outputs": outs,
                    "fp": input_fingerprint(
                        None,
                        container=pin,
                        spec_hash=sh,
                        series_uid=r.get("series_uid", ""),
                        rtstruct_sha=r.get("rtstruct_sha", ""),
                        extra=(*cmds, r.get("roi_candidates", ""), r.get("resample_mm", "")),
                    ),
                }
            )
        ccmds = flat(case_steps, pid)
        uids = sorted(
            r.get("series_uid", "") for r in (rows.to_dict("records") if rows is not None else [])
        )
        case_entries.append(
            {
                "id": pid,
                "cmds": ccmds,
                "fp": input_fingerprint(
                    None, container=pin, spec_hash=sh, series_uid=",".join(uids), extra=ccmds
                ),
                "pairs": pair_entries,
            }
        )
    return {
        "plan_name": plan.name,
        "fingerprint": input_fingerprint(manifest, container=pin, spec_hash=sh),
        "cwd": os.getcwd(),
        "outdir": str(Path(outdir).resolve()),
        "pre": flat(pre_steps),
        "cases": case_entries,
        "post": flat(post_steps) if fanned else [],
        "has_pairs": bool(pair_steps),
        "on_error": on_error,
        "max_forks": int(max_forks),
        "container": pin,
        "container_digest": cdig,
    }


def write_provenance(
    pair_id: str, files: Sequence[str], output: str, container: Optional[str] = None
) -> Dict[str, Any]:
    """Write <pair>.prov.json: sha256 of every declared output + tool versions
    (+ container ref). A declared output that is missing raises FileNotFoundError
    so the producing Nextflow task fails instead of publishing a hollow record."""
    import hashlib

    outs: Dict[str, Any] = {}
    for f in files:
        fp = Path(f)
        if not fp.is_file():
            raise FileNotFoundError(f"provenance: declared output missing: {f}")
        h = hashlib.sha256()
        with fp.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        outs[str(fp)] = {"sha256": h.hexdigest(), "bytes": fp.stat().st_size}
    rec = {
        "pair_id": pair_id,
        "versions": tool_versions(),
        "container": container or "",
        "outputs": outs,
    }
    Path(output).write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def nextflow_argv(
    params_file: Path, run_dir: Path, profile: str = "local", resume: bool = True
) -> List[str]:
    argv = [
        "nextflow",
        "run",
        str(nextflow_main_path()),
        "-params-file",
        str(params_file),
        "-work-dir",
        str(run_dir / "work"),
        "-profile",
        profile,
    ]
    if resume:
        argv.append("-resume")
    return argv


def run_plan_nextflow(
    plan: WorkflowPlan,
    manifest: Optional[str] = None,
    run_dir: Optional[str] = None,
    profile: str = "local",
    resume: bool = True,
    dry_run: bool = False,
    on_error: str = "finish",
    max_forks: int = 0,
    container: Optional[str] = None,
) -> Dict[str, Any]:
    """Run `plan` through the packaged nf/main.nf. Returns a result dict."""
    if not dry_run and shutil.which("nextflow") is None:
        raise NextflowNotFoundError(
            "nextflow not on PATH; install it (https://nextflow.io, needs Java 17+)"
            " or use --executor inline"
        )
    container_digest(container)  # reject unpinned refs before anything launches
    if container and not {"docker", "singularity", "apptainer"} & set(profile.split(",")):
        profile = f"{profile},docker"
    outdir = str(_resolve(plan.vars.get("outdir", "runs/nextflow"), plan.vars))
    run = Path(run_dir or Path(outdir) / "nf_run").resolve()
    # The case list of a plan whose cases come from a pre step's output
    # (case_manifest, e.g. tcia manifest) is unknown until that step ran:
    # run the pre steps inline once, then hand Nextflow only cases + post.
    cm = plan.vars.get("case_manifest")
    pre_ran = False
    if not manifest and cm and not dry_run and not Path(str(_resolve(cm, plan.vars))).is_file():
        first = next((i for i, st in enumerate(plan.steps) if st.fanout), len(plan.steps))
        pre = WorkflowPlan(name=plan.name, vars=plan.vars, steps=plan.steps[:first])
        res = WorkflowRunner(pre).run()
        if any(r.get("returncode", 0) != 0 for r in res):
            return {"executor": "nextflow", "stage": "pre", "results": res, "returncode": 1}
        pre_ran = True
    cases = resolve_cases(plan, manifest, exclude=[outdir, run])
    pairs = None
    if any(st.fanout == "pair" for st in plan.steps):
        if not manifest:
            raise ValueError("plan has pair-level steps; pass --manifest <pair manifest>")
        pairs = resolve_pairs(manifest)
    params = plan_to_nextflow_params(
        plan,
        cases,
        outdir,
        manifest=manifest,
        pairs=pairs,
        on_error=on_error,
        max_forks=max_forks,
        container=container,
    )
    if pre_ran:
        params["pre"] = []
    if resume and not dry_run:
        missing = [o for o in _cohort_outputs(plan) if not Path(o).exists()]
        if missing:
            log.warning(
                "declared cohort output(s) missing (%s); running without -resume", missing[:3]
            )
            resume = False
    params_file = run / "params.json"
    argv = nextflow_argv(params_file, run, profile, resume)
    if dry_run:
        return {"executor": "nextflow", "status": "dry-run", "cmd": argv, "params": params}
    run.mkdir(parents=True, exist_ok=True)
    params_file.write_text(json.dumps(params, indent=2))
    proc = subprocess.run(argv, check=False, cwd=run)
    return {
        "executor": "nextflow",
        "cmd": argv,
        "params_file": str(params_file),
        "returncode": proc.returncode,
    }


# ─── Scaffolders ─────────────────────────────────────────────────────────────


def _reject_pair_fanout(plan: WorkflowPlan, target: str) -> None:
    if any(st.fanout == "pair" for st in plan.steps):
        raise ValueError(
            f"scaffold {target}: fanout='pair' steps are only supported by the inline and "
            "nextflow executors (qr workflow run -e ...)"
        )


def scaffold_shell(plan: WorkflowPlan) -> str:
    """Render the plan as a bash script that calls qr in sequence."""
    _reject_pair_fanout(plan, "shell")
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
    _reject_pair_fanout(plan, "prefect")
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
    lines.append('                    key=f\'{stage}-{p.name.replace(".", "-")}\',')
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
        lines.append(
            f'    """Stage `{stage}` — {len(steps_in_stage)} step(s):'
            f' {", ".join(s.id for s in steps_in_stage)}."""'
        )
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
    phase_a_task_names = [
        n for n, s in zip(stage_task_names, stages_order) if s not in phase_b_stages
    ]
    phase_b_task_names = [n for n, s in zip(stage_task_names, stages_order) if s in phase_b_stages]

    # ─── Subflow: Phase A (features) ────────────────────────────────────────
    if phase_a_task_names:
        lines.append("@flow(name='phase_a_features')")
        lines.append("def phase_a_features():")
        lines.append(
            '    """Phase A — PACS/TCIA → features. Nextflow-suited, file-level fan-out."""'
        )
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
    _reject_pair_fanout(plan, "nextflow")
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
    "nextflow": scaffold_nextflow,  # default for production / large cohorts
    "prefect": scaffold_prefect,  # secondary, scheduling / observability
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
