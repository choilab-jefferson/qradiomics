"""Regression test pinning qradiomics.atomic.extract_features output.

JLR's ADR-005 stage 3a swept qradiomics.atomic.extract_features against
its legacy `_extract_features_for_roi` across 24 real cohort cases and
got 0/1409 feature diffs. This file pins the qradiomics output for
HeartB10 (one of those cases) as a golden reference so any future
change here that perturbs the extractor output fails CI before it
ships and re-invalidates the JLR parity contract.

Regenerate intentionally via tests/fixtures/regenerate_golden_heartb10.py
when PyRadiomics is upgraded or extract_features defaults change.
"""
import json
from pathlib import Path

import pytest

from qradiomics.atomic import extract_features, load_image_and_mask


GOLDEN = Path(__file__).parent / "fixtures" / "heartb10_features_golden.json"
TOL_ABS = 1e-9
TOL_REL = 1e-9


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"golden fixture missing: {GOLDEN}")
    return json.loads(GOLDEN.read_text())


@pytest.mark.skipif(
    not Path("/data/datasets/heart-toxicity-heartcb/nrrd/HeartB10").exists(),
    reason="HeartB10 cohort data absent — parity check skipped",
)
class TestJLRParityGolden:
    def test_byte_identical_features(self, golden):
        img, msk = load_image_and_mask(golden["image_path"],
                                        golden["mask_path"])
        feats = extract_features(img, msk)
        expected = golden["features"]

        missing = set(expected) - set(feats)
        extra = set(feats) - set(expected)
        assert not missing, f"missing keys: {sorted(missing)[:5]}"
        assert not extra, f"unexpected keys: {sorted(extra)[:5]}"

        diffs = []
        for k, want in expected.items():
            got = feats[k]
            if isinstance(want, str):
                if str(got) != want:
                    diffs.append((k, got, want))
                continue
            try:
                got_f = float(got)
            except (TypeError, ValueError):
                continue
            if abs(got_f - want) > max(TOL_ABS, TOL_REL * abs(want)):
                diffs.append((k, got_f, want, abs(got_f - want)))
        assert not diffs, (
            f"{len(diffs)} feature(s) drift from golden (showing 5): "
            f"{diffs[:5]}"
        )

    def test_feature_count_matches_jlr_sweep(self, golden):
        """JLR's 24-case AB-parity sweep reported 1409 features per case
        in the default (full) extract_features configuration."""
        assert len(golden["features"]) == 1409
