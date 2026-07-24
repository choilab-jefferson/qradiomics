"""The 24-check radiomics-QA rubric encoded as data.

This mirrors the canonical document at
``reports/radiomics_qa/RADIOMICS_QA_RUBRIC.md``. Each check is a
:class:`RubricCheck` carrying its id, human name, family, gating flag,
default weight and a one-line criterion. The scorer in
:mod:`qradiomics.analytics.qa.scorecard` reads these to build a
:class:`~qradiomics.analytics.qa.scorecard.Scorecard`.

Keeping the rubric as data (rather than scattering it across the
detector code) means the canonical doc and the executable scorer share
a single source of truth; a drift test can diff the two.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["RubricCheck", "RUBRIC", "GATING_IDS", "get_check"]


@dataclass(frozen=True)
class RubricCheck:
    """One row of the rubric."""

    id: str
    name: str
    family: str
    gating: bool
    weight: float = 1.0
    criterion: str = ""


# The 24 checks, in canonical order. Families match section 2 of the doc.
RUBRIC: tuple[RubricCheck, ...] = (
    # --- ML-validity ---
    RubricCheck("C01", "Feature-selection leakage", "ML-validity", True,
                criterion="All selection/scaling/harmonization/stability fit "
                          "inside CV folds; no full-data fit before split."),
    RubricCheck("C05", "Calibration reported", "ML-validity", True,
                criterion="Calibration curve AND (Brier or ECE) present."),
    RubricCheck("C07", "Events-per-variable / sample size", "ML-validity", False,
                criterion="EPV >= 10 PASS; 5-10 PARTIAL; < 5 FAIL."),
    RubricCheck("C23", "Reporting-checklist adherence", "ML-validity", False,
                criterion="% of TRIPOD+AI / CLAIM items satisfied."),
    RubricCheck("C24", "Reproducible split / seed", "ML-validity", False,
                criterion="Deterministic seed AND serialized split indices."),
    # --- External validation & discrimination ---
    RubricCheck("C02", "External validation", "External-validation", True,
                criterion="0=FAIL / internal-only=PARTIAL / 1=PASS / multi=PASS+."),
    RubricCheck("C06", "Discrimination with CIs", "External-validation", False,
                criterion="Discrimination metric reported with uncertainty."),
    RubricCheck("C21", "Comparison to baseline / clinical standard",
                "External-validation", False,
                criterion="Non-radiomic baseline comparison reported."),
    RubricCheck("C22", "Clinical utility (decision-curve)",
                "External-validation", False,
                criterion="DCA / net-benefit analysis present."),
    RubricCheck("C14", "Multiple-testing / feature reduction",
                "External-validation", False,
                criterion="Dimensionality reduction OR multiple-testing correction."),
    RubricCheck("C15", "Acquisition protocol logged", "External-validation", False,
                criterion="Acquisition parameters captured per case."),
    # --- IBSI compliance ---
    RubricCheck("C03", "IBSI-1 feature compliance", "IBSI", True,
                criterion="IBSI-aligned config AND a parity/golden test."),
    RubricCheck("C04", "IBSI-2 filter compliance", "IBSI", False,
                criterion="NA if no conv filters; else IBSI-2 aligned + parity test."),
    RubricCheck("C12", "Discretization-sensitivity audit", "IBSI", False,
                criterion="Bin scheme documented AND sensitivity quantified."),
    RubricCheck("C16", "Resampling & preprocessing documented", "IBSI", False,
                criterion="Full preprocessing chain in one versioned config."),
    RubricCheck("C17", "IBSI-compliant feature naming", "IBSI", False,
                criterion="Features carry IBSI tags / standard names."),
    # --- Reproducibility (ICC) ---
    RubricCheck("C08", "Test-retest / repeatability ICC", "Reproducibility", False,
                criterion="% features ICC > 0.75 / > 0.90; NA if no repeat data."),
    RubricCheck("C09", "Stability features actually filtered",
                "Reproducibility", False,
                criterion="Sub-threshold features dropped before training."),
    # --- Robustness ---
    RubricCheck("C10", "Image-perturbation robustness", "Robustness", False,
                criterion="% features ICC(1,1) >= 0.90 under perturbation; NA if absent."),
    RubricCheck("C11", "Segmentation-variability robustness", "Robustness", False,
                criterion="% features ICC > 0.75 across segmentations; NA if absent."),
    # --- Harmonization ---
    RubricCheck("C13", "Harmonization adequacy (ComBat)", "Harmonization", False,
                criterion="Post-ComBat batch-association collapses toward ~5%; "
                          "NA if single-scanner."),
    # --- Open science ---
    RubricCheck("C18", "Feature-set documentation", "Open-science", False,
                criterion="Every output feature fully provenanced."),
    RubricCheck("C19", "Code availability", "Open-science", False,
                criterion="Extraction + modeling code openly available + license."),
    RubricCheck("C20", "Data / model availability", "Open-science", False,
                criterion="Data DOI/access statement + released trained model."),
)

# Relative weights derived from the METRICS category-importance ranking
# (Kocak 2024: study design / metrics / image-processing / testing rank above
# feature processing / preparation / segmentation / open science). These are a
# principled relative weighting, NOT the exact per-item Delphi weights; the
# four gating checks and the highest-importance categories carry the most
# weight, open-science the least (METRICS items 28-30 are the lowest-weighted).
_WEIGHTS: dict[str, float] = {
    # gating + top-importance
    "C01": 2.5, "C02": 2.5, "C03": 2.0, "C05": 2.0,
    # discrimination, sample size, reduction, preprocessing
    "C06": 1.5, "C07": 1.5, "C14": 1.3, "C16": 1.3, "C15": 1.2,
    # robustness / reproducibility / harmonization
    "C08": 1.2, "C09": 1.2, "C10": 1.0, "C11": 1.0, "C12": 1.0, "C13": 1.0,
    "C04": 1.0,
    # reporting / determinism / utility
    "C24": 1.0, "C23": 1.0, "C21": 1.0, "C22": 0.8, "C17": 0.8, "C18": 0.8,
    # open science (lowest, per METRICS)
    "C19": 0.4, "C20": 0.4,
}
RUBRIC = tuple(replace(c, weight=_WEIGHTS.get(c.id, 1.0)) for c in RUBRIC)

GATING_IDS: frozenset[str] = frozenset(c.id for c in RUBRIC if c.gating)

_BY_ID: dict[str, RubricCheck] = {c.id: c for c in RUBRIC}


def get_check(check_id: str) -> RubricCheck:
    """Look up a rubric check by id (e.g. ``"C01"``)."""
    return _BY_ID[check_id]
