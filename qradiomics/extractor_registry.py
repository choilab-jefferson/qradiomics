"""
Extraction engine registry — dict-based lookup mirroring
qradiomics/classification/registry.py's MODEL_REGISTRY pattern.

Provides:
    EXTRACTOR_REGISTRY  name -> zero-arg constructor
    ALL_ENGINES          tuple of every registered engine name
    DEFAULT_ENGINE        single-engine backward-compat default
    build_extractor(name)         instantiate one engine
    resolve_engines(spec, ...)    parse a comma-list/"all" spec into names
    run_multi_extraction(...)     run one or many engines, merge if >1
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

import pandas as pd

from qradiomics.extractor import RadiomicsExtractor
from qradiomics.pysera_extractor import PyseraExtractor
from qradiomics.rtools_extractor import RtoolsExtractor

logger = logging.getLogger(__name__)

EXTRACTOR_REGISTRY: Dict[str, Callable[[], Any]] = {
    "pyradiomics": RadiomicsExtractor,
    "pysera": PyseraExtractor,
    "rtools": RtoolsExtractor,
}

ALL_ENGINES = tuple(EXTRACTOR_REGISTRY)
DEFAULT_ENGINE = "pyradiomics"


def build_extractor(name: str) -> Any:
    """Instantiate the named extraction engine."""
    if name not in EXTRACTOR_REGISTRY:
        raise KeyError(f"Unknown extraction engine '{name}'. Available: {list(EXTRACTOR_REGISTRY)}")
    return EXTRACTOR_REGISTRY[name]()


def resolve_engines(spec: Optional[str], pattern_extractor: Optional[str] = None) -> List[str]:
    """Resolve a comma-separated engine spec (or 'all') into a name list.

    Resolution order: ``spec`` if given (non-None, non-empty) else
    ``pattern_extractor`` (same comma/'all' support) else ``[DEFAULT_ENGINE]``.
    Dedupes while preserving order. Raises KeyError (matching
    build_extractor's message style) if any resolved name is unregistered.
    """
    chosen = spec if spec else pattern_extractor
    if not chosen:
        return [DEFAULT_ENGINE]

    if chosen.strip().lower() == "all":
        return list(ALL_ENGINES)

    names: List[str] = []
    seen = set()
    for item in chosen.split(","):
        name = item.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)

    unknown = [n for n in names if n not in EXTRACTOR_REGISTRY]
    if unknown:
        raise KeyError(
            f"Unknown extraction engine(s) {unknown}. Available: {list(EXTRACTOR_REGISTRY)}"
        )

    return names


def run_multi_extraction(
    engines: List[str],
    job_id: UUID,
    manifest_path: Path,
    job_dir: Path,
    extraction_settings: Dict[str, Any],
    jobs: int = 1,
) -> Dict[str, Any]:
    """Run one or more extraction engines and merge their outputs.

    Single-engine case is a pure passthrough (backward-compat guarantee):
    the sole engine's run_extraction result is returned unchanged, and
    job_dir/features.csv is written exactly where that engine already puts
    it — no subdirectory, no merge step.

    Multi-engine case: each engine runs into its own job_dir/_engine_<name>
    subdirectory, columns (except patient_id) are prefixed with
    "<engine>_", and the per-engine features.csv files are outer-merged on
    patient_id into job_dir/features.csv.
    """
    if len(engines) == 1:
        extractor = build_extractor(engines[0])
        return extractor.run_extraction(job_id, manifest_path, job_dir, extraction_settings, jobs)

    per_engine_results: Dict[str, Any] = {}
    merged_df: Optional[pd.DataFrame] = None
    subdirs: List[Path] = []

    for engine in engines:
        subdir = job_dir / f"_engine_{engine}"
        subdir.mkdir(parents=True, exist_ok=True)
        subdirs.append(subdir)

        extractor = build_extractor(engine)
        result = extractor.run_extraction(job_id, manifest_path, subdir, extraction_settings, jobs)
        per_engine_results[engine] = result

        engine_csv = subdir / "features.csv"
        if not engine_csv.exists():
            logger.warning("Engine '%s' produced no features.csv, skipping in merge", engine)
            continue

        df = pd.read_csv(engine_csv)
        if "patient_id" not in df.columns or df.empty:
            continue

        rename_map = {c: f"{engine}_{c}" for c in df.columns if c != "patient_id"}
        df = df.rename(columns=rename_map)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on="patient_id", how="outer")

    features_path = job_dir / "features.csv"
    if merged_df is not None and not merged_df.empty:
        merged_df.to_csv(features_path, index=False)
        feature_count = len([c for c in merged_df.columns if c != "patient_id"])
        patients_processed = merged_df["patient_id"].nunique()
        status = "extracted"
    else:
        with open(features_path, "w") as f:
            f.write("patient_id\n")
        feature_count = 0
        patients_processed = 0
        status = "error"

    # Known limitation: failed/skipped counts are summed per-engine, so a
    # patient rejected by multiple engines (e.g. missing mask) is counted
    # once per engine it failed/was skipped on, not once overall. Accepted
    # as-is per spec — patients_processed uses the deduped merged output so
    # it does not suffer the same overcount.
    patients_failed = sum(r.get("patients_failed", 0) for r in per_engine_results.values())
    patients_skipped = sum(r.get("patients_skipped", 0) for r in per_engine_results.values())

    for subdir in subdirs:
        try:
            shutil.rmtree(subdir, ignore_errors=True)
        except Exception:
            logger.warning("Failed to clean up %s (best-effort)", subdir, exc_info=True)

    return {
        "features_uri": f"file://{features_path.resolve()}",
        "feature_count": feature_count,
        "patients_processed": patients_processed,
        "patients_failed": patients_failed,
        "patients_skipped": patients_skipped,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "per_engine": per_engine_results,
    }
