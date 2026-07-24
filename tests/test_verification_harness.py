"""Tests for qradiomics.verification — AB-parity harness."""
from pathlib import Path

import pytest

from qradiomics.verification import (
    FeatureExtractor,
    compare_feature_dicts,
    run_ab_sweep,
)


class TestCompareFeatureDicts:
    def test_identical_dicts_zero_diff(self):
        a = {"f1": 1.0, "f2": 2.0}
        diffs = compare_feature_dicts(a, dict(a))
        assert diffs == []

    def test_diff_above_tolerance_caught(self):
        a = {"f1": 1.0}
        b = {"f1": 1.0 + 1e-6}
        diffs = compare_feature_dicts(a, b, abs_tol=1e-9, rel_tol=1e-9)
        assert len(diffs) == 1
        key, la, ca, ab, rel = diffs[0]
        assert key == "f1"
        assert abs(la - 1.0) < 1e-12
        assert abs(ca - (1.0 + 1e-6)) < 1e-12

    def test_diff_below_tolerance_ignored(self):
        a = {"f1": 1.0}
        b = {"f1": 1.0 + 1e-12}
        diffs = compare_feature_dicts(a, b, abs_tol=1e-9, rel_tol=1e-9)
        assert diffs == []

    def test_missing_and_extra_keys_flagged(self):
        a = {"f1": 1.0, "f2": 2.0}
        b = {"f1": 1.0, "f3": 3.0}
        diffs = compare_feature_dicts(a, b)
        labels = sorted(d[0] for d in diffs)
        assert any("missing" in l for l in labels)
        assert any("extra" in l for l in labels)


_HEARTCB = Path("/data/datasets/heart-toxicity-heartcb/nrrd")


@pytest.mark.skipif(
    not _HEARTCB.exists(),
    reason="HeartCB cohort absent — sweep integration skipped",
)
def test_run_ab_sweep_self_pair_is_zero_diff(tmp_path):
    """Sweep qradiomics.atomic against itself across 2 real cases — must
    yield 0 diffs for every successful case."""
    from qradiomics.atomic import extract_features, load_image_and_mask

    # Build a tiny manifest in-memory
    cases = [
        ("HeartB10", "HeartB10_planCT_cropped-1mm.nrrd",
                      "HeartB10_planCT_manual_cropped-1mm-label.nrrd"),
        ("HeartB14", "HeartB14_planCT_cropped-1mm.nrrd",
                      "HeartB14_planCT_manual_cropped-1mm-label.nrrd"),
    ]
    manifest = tmp_path / "m.csv"
    manifest.write_text("patient_id,image_path,mask_path\n" + "\n".join(
        f"{pid},{_HEARTCB}/{pid}/{img},{_HEARTCB}/{pid}/{msk}"
        for pid, img, msk in cases
    ) + "\n")

    def extract(img_path, msk_path):
        img, msk = load_image_and_mask(img_path, msk_path)
        return extract_features(img, msk)

    result = run_ab_sweep(
        manifest, FeatureExtractor("ref", extract),
        FeatureExtractor("cand", extract),
        tmp_path / "out", abs_tol=1e-9, rel_tol=1e-9,
    )
    assert result.n_cases == 2
    assert result.n_success == 2
    assert result.n_zero_diff == 2
    assert result.total_diff_features == 0

    # Verify report artefacts exist with expected columns
    summary_csv = (tmp_path / "out" / "per_case_summary.csv").read_text()
    assert "case_id" in summary_csv
    assert "n_diffs" in summary_csv
    assert "HeartB10" in summary_csv
