"""Check / Scorecard datamodel + weighted scoring, bands and gating cap.

Implements section 3 of ``RADIOMICS_QA_RUBRIC.md``:

* per-check numeric value (PASS/PASS+ = 1.0, PARTIAL = 0.5, FAIL = 0.0,
  GRADED = measured fraction, NA = dropped from numerator and denominator);
* weighted percentage over **applicable** checks;
* bands (very low / low / moderate / good / excellent);
* gating cap (any GATING check at FAIL caps the band at *moderate*).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

from qradiomics.analytics.qa.rubric import GATING_IDS, get_check

__all__ = ["Verdict", "Check", "Scorecard", "band_for", "MODERATE_CAP_PCT"]

# A failed gate caps the band at *moderate*; the moderate band is 40-60%,
# so we clamp the *reported* percentage used for banding to just below 60.
MODERATE_CAP_PCT = 59.999


class Verdict(str, Enum):
    """Per-check verdicts. Values double as their numeric score except
    GRADED (which carries an explicit fraction) and NA (excluded)."""

    PASS_PLUS = "PASS+"
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    GRADED = "GRADED"
    NA = "NA"


_VERDICT_VALUE = {
    Verdict.PASS_PLUS: 1.0,
    Verdict.PASS: 1.0,
    Verdict.PARTIAL: 0.5,
    Verdict.FAIL: 0.0,
}


@dataclass
class Check:
    """A scored check: a rubric id + a verdict + evidence.

    ``fraction`` is required (and used) only when ``verdict`` is GRADED.
    ``evidence`` holds detector findings (e.g. file:line for leakage).
    """

    id: str
    verdict: Verdict
    fraction: Optional[float] = None
    message: str = ""
    evidence: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return get_check(self.id).name

    @property
    def gating(self) -> bool:
        return self.id in GATING_IDS

    @property
    def weight(self) -> float:
        return get_check(self.id).weight

    @property
    def applicable(self) -> bool:
        return self.verdict is not Verdict.NA

    @property
    def value(self) -> Optional[float]:
        """Numeric score in [0, 1], or None when NA (excluded)."""
        if self.verdict is Verdict.NA:
            return None
        if self.verdict is Verdict.GRADED:
            if self.fraction is None:
                raise ValueError(f"{self.id}: GRADED verdict needs a fraction")
            return max(0.0, min(1.0, float(self.fraction)))
        return _VERDICT_VALUE[self.verdict]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["name"] = self.name
        d["gating"] = self.gating
        d["weight"] = self.weight
        d["value"] = self.value
        return d


def band_for(pct: float) -> str:
    """Map a percentage to a band label (section 3.3)."""
    if pct < 20:
        return "Very low"
    if pct < 40:
        return "Low"
    if pct < 60:
        return "Moderate"
    if pct < 80:
        return "Good"
    return "Excellent"


@dataclass
class Scorecard:
    """A collection of scored checks with weighted % + band + gating cap."""

    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def applicable_checks(self) -> list[Check]:
        return [c for c in self.checks if c.applicable]

    @property
    def weighted_pct(self) -> float:
        """Raw weighted percentage over applicable checks (no cap)."""
        applicable = self.applicable_checks
        denom = sum(c.weight for c in applicable)
        if denom == 0:
            return 0.0
        num = sum(c.weight * c.value for c in applicable if c.value is not None)
        return 100.0 * num / denom

    @property
    def failed_gates(self) -> list[Check]:
        return [c for c in self.checks
                if c.gating and c.verdict is Verdict.FAIL]

    @property
    def gate_capped(self) -> bool:
        return bool(self.failed_gates)

    @property
    def banding_pct(self) -> float:
        """Percentage used for banding: capped to moderate if a gate failed."""
        pct = self.weighted_pct
        if self.gate_capped:
            return min(pct, MODERATE_CAP_PCT)
        return pct

    @property
    def band(self) -> str:
        return band_for(self.banding_pct)

    def to_dict(self) -> dict:
        return {
            "weighted_pct": round(self.weighted_pct, 2),
            "banding_pct": round(self.banding_pct, 2),
            "band": self.band,
            "gate_capped": self.gate_capped,
            "failed_gates": [c.id for c in self.failed_gates],
            "n_applicable": len(self.applicable_checks),
            "n_total": len(self.checks),
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            "# Radiomics QA Scorecard",
            "",
            f"- **Weighted score**: {self.weighted_pct:.1f}%",
            f"- **Band**: {self.band}"
            + (f"  _(capped to moderate: failed gate(s) "
               f"{', '.join(c.id for c in self.failed_gates)})_"
               if self.gate_capped else ""),
            f"- **Applicable checks**: {len(self.applicable_checks)} / "
            f"{len(self.checks)}",
            "",
            "| ID | Check | Gate | Verdict | Score | Notes |",
            "|----|-------|:----:|---------|:-----:|-------|",
        ]
        for c in self.checks:
            val = "-" if c.value is None else f"{c.value:.2f}"
            gate = "G" if c.gating else ""
            msg = c.message.replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {c.id} | {c.name} | {gate} | {c.verdict.value} | "
                f"{val} | {msg} |"
            )
        # Evidence detail for any check that produced findings.
        detailed = [c for c in self.checks if c.evidence]
        if detailed:
            lines += ["", "## Evidence", ""]
            for c in detailed:
                lines.append(f"### {c.id} — {c.name}")
                for ev in c.evidence:
                    lines.append(f"- {ev}")
                lines.append("")
        return "\n".join(lines)
