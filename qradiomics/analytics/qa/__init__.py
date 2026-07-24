"""Radiomics QA scorer — operationalizes the 24-check rubric as code.

The canonical rubric lives at ``reports/radiomics_qa/RADIOMICS_QA_RUBRIC.md``;
this package mirrors it as data (:mod:`rubric`), provides the scoring
datamodel (:mod:`scorecard`) and the automatically-computable detectors
(:mod:`detectors`). The CLI entry point is ``qr qa score``.

Typical programmatic use::

    from qradiomics.analytics.qa import Scorecard, detect_c01_leakage

    card = Scorecard()
    card.add(detect_c01_leakage("path/to/pipeline.py"))
    print(card.weighted_pct, card.band)
"""
from qradiomics.analytics.qa.detectors import (
    detect_c01_leakage,
    detect_c02_external,
    detect_c05_calibration,
    detect_c06_discrimination,
    detect_c07_epv,
    detect_c08_c09_stability,
    detect_c10_c11_perturbation,
    detect_c13_harmonization,
    detect_c17_naming,
    detect_c18_documentation,
    detect_c24_seed_split,
)
from qradiomics.analytics.qa.rubric import RUBRIC, GATING_IDS, RubricCheck, get_check
from qradiomics.analytics.qa.scorecard import (
    Check,
    Scorecard,
    Verdict,
    band_for,
)

__all__ = [
    "RUBRIC",
    "GATING_IDS",
    "RubricCheck",
    "get_check",
    "Check",
    "Scorecard",
    "Verdict",
    "band_for",
    "detect_c01_leakage",
    "detect_c02_external",
    "detect_c05_calibration",
    "detect_c06_discrimination",
    "detect_c07_epv",
    "detect_c08_c09_stability",
    "detect_c10_c11_perturbation",
    "detect_c13_harmonization",
    "detect_c17_naming",
    "detect_c18_documentation",
    "detect_c24_seed_split",
]
