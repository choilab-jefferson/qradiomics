"""AB-parity sweep over a manifest of (image, mask) pairs."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
ExtractFn = Callable[[str, str], Dict[str, float]]


@dataclass
class FeatureExtractor:
    """A named extraction function. The function takes
    (image_path, mask_path) and returns a feature dict."""
    label: str
    fn: ExtractFn


@dataclass
class SweepResult:
    """Outcome of an AB-parity run."""
    n_cases: int = 0
    n_success: int = 0
    n_failed_legacy: int = 0
    n_failed_candidate: int = 0
    n_zero_diff: int = 0           # successful cases with no feature diffs
    n_with_diffs: int = 0          # successful cases that had ≥1 diff
    total_diff_features: int = 0
    out_dir: Optional[Path] = None
    per_case: List[Dict] = field(default_factory=list)


def compare_feature_dicts(
    legacy: Dict[str, float],
    candidate: Dict[str, float],
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-9,
) -> List[Tuple[str, float, float, float, float]]:
    """Return diff rows: (feature, legacy, candidate, abs_err, rel_err).

    Only features present in BOTH dicts and where the diff exceeds the
    tolerance are returned. Non-float values must match exactly (string
    inequality counts as a diff with abs/rel = inf).
    """
    diffs = []
    keys = set(legacy) & set(candidate)
    missing_in_candidate = set(legacy) - set(candidate)
    extra_in_candidate = set(candidate) - set(legacy)
    for k in missing_in_candidate:
        diffs.append((f"<missing>{k}", _coerce(legacy[k]), 0.0,
                      float("inf"), float("inf")))
    for k in extra_in_candidate:
        diffs.append((f"<extra>{k}", 0.0, _coerce(candidate[k]),
                      float("inf"), float("inf")))
    for k in keys:
        a, b = legacy[k], candidate[k]
        try:
            af, bf = float(a), float(b)
        except (TypeError, ValueError):
            if str(a) != str(b):
                diffs.append((k, 0.0, 0.0, float("inf"), float("inf")))
            continue
        abs_err = abs(af - bf)
        rel_err = abs_err / max(abs(bf), 1e-300)
        if abs_err > max(abs_tol, rel_tol * abs(bf)):
            diffs.append((k, af, bf, abs_err, rel_err))
    return diffs


def _coerce(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def run_ab_sweep(
    manifest_csv: PathLike,
    legacy: FeatureExtractor,
    candidate: FeatureExtractor,
    out_dir: PathLike,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-9,
    case_id_col: str = "patient_id",
    image_col: str = "image_path",
    mask_col: str = "mask_path",
    limit: Optional[int] = None,
) -> SweepResult:
    """Run two FeatureExtractors over a manifest and emit AB-parity reports.

    Args:
        manifest_csv: CSV with patient_id + image_path + mask_path columns.
        legacy: the reference implementation.
        candidate: the new (qradiomics.atomic) implementation under test.
        out_dir: directory for per_case_summary.csv + feature_diffs.csv.
        abs_tol, rel_tol: tolerance per feature; default matches HeartB10
            golden (1e-9, same as ADR-005 stage-3a verification).
        case_id_col, image_col, mask_col: manifest column names.
        limit: optional max cases to process (useful for smoke runs).

    Returns:
        SweepResult with totals + per-case details.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "per_case_summary.csv"
    diffs_path = out / "feature_diffs.csv"

    result = SweepResult(out_dir=out)
    summary_rows = []
    diff_rows = []

    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            if limit is not None and idx > limit:
                break
            case_id = row.get(case_id_col, f"row{idx}")
            img = row[image_col]
            msk = row[mask_col]
            result.n_cases += 1

            try:
                legacy_feats = legacy.fn(img, msk)
            except Exception as e:
                result.n_failed_legacy += 1
                summary_rows.append({"case_id": case_id,
                    "status": f"legacy_error:{type(e).__name__}",
                    "n_features": 0, "n_diffs": 0,
                    "max_abs": "", "max_rel": ""})
                logger.warning("[%s] legacy failed: %s", case_id, e)
                continue
            try:
                cand_feats = candidate.fn(img, msk)
            except Exception as e:
                result.n_failed_candidate += 1
                summary_rows.append({"case_id": case_id,
                    "status": f"candidate_error:{type(e).__name__}",
                    "n_features": 0, "n_diffs": 0,
                    "max_abs": "", "max_rel": ""})
                logger.warning("[%s] candidate failed: %s", case_id, e)
                continue

            diffs = compare_feature_dicts(legacy_feats, cand_feats,
                                          abs_tol=abs_tol, rel_tol=rel_tol)
            n_diffs = len(diffs)
            n_total = max(len(legacy_feats), len(cand_feats))
            max_abs = max((d[3] for d in diffs), default=0.0)
            max_rel = max((d[4] for d in diffs), default=0.0)
            result.n_success += 1
            if n_diffs == 0:
                result.n_zero_diff += 1
            else:
                result.n_with_diffs += 1
                result.total_diff_features += n_diffs
            summary_rows.append({"case_id": case_id,
                "status": "ok" if n_diffs == 0 else "diff",
                "n_features": n_total,
                "n_diffs": n_diffs,
                "max_abs": f"{max_abs:.6g}",
                "max_rel": f"{max_rel:.6g}"})
            for d in diffs:
                diff_rows.append({"case_id": case_id,
                    "feature": d[0],
                    "legacy": d[1],
                    "candidate": d[2],
                    "abs": d[3],
                    "rel": d[4]})
            print(f"  [{idx}] {case_id}: "
                  f"{n_diffs}/{n_total} diffs "
                  f"(max_abs={max_abs:.2e})",
                  flush=True)

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f,
            fieldnames=["case_id", "status", "n_features",
                        "n_diffs", "max_abs", "max_rel"])
        w.writeheader()
        w.writerows(summary_rows)
    with open(diffs_path, "w", newline="") as f:
        w = csv.DictWriter(f,
            fieldnames=["case_id", "feature", "legacy",
                        "candidate", "abs", "rel"])
        w.writeheader()
        w.writerows(diff_rows)
    result.per_case = summary_rows
    return result
