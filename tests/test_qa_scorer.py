"""Unit tests for the radiomics QA scorer (qradiomics.analytics.qa)."""
from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from qradiomics.analytics.qa import (
    Check,
    Scorecard,
    Verdict,
    detect_c01_leakage,
    detect_c07_epv,
    detect_c17_naming,
)

# --------------------------------------------------------------------------- #
# C01 leakage — bad vs good toy pipelines
# --------------------------------------------------------------------------- #

LEAKY_PIPELINE = textwrap.dedent(
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("features.csv")
    X = df.drop(columns=["y"])
    y = df["y"]

    # LEAK: scaler + selection fit on the FULL matrix before the split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(k=10)
    X_sel = selector.fit_transform(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, random_state=0)
    """
)

SAFE_PIPELINE = textwrap.dedent(
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    df = pd.read_csv("features.csv")
    X = df.drop(columns=["y"])
    y = df["y"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(k=10)),
        ("clf", LogisticRegression()),
    ])
    scores = cross_val_score(pipe, X, y, cv=cv)
    """
)


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(src)
    return p


def test_c01_flags_leaky_pipeline(tmp_path):
    p = _write(tmp_path, "leaky.py", LEAKY_PIPELINE)
    check = detect_c01_leakage(p)
    assert check.id == "C01"
    assert check.verdict is Verdict.FAIL
    assert check.evidence, "leakage findings should report file:line"
    # the fit_transform lines are flagged
    assert any("fit_transform" in e for e in check.evidence)


def test_c01_passes_safe_pipeline(tmp_path):
    p = _write(tmp_path, "safe.py", SAFE_PIPELINE)
    check = detect_c01_leakage(p)
    assert check.id == "C01"
    assert check.verdict is Verdict.PASS
    assert not check.evidence


def test_c01_na_when_no_source(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    check = detect_c01_leakage(empty)
    assert check.verdict is Verdict.NA


# Outcome-aware feature reduction fit once on the whole training set, then the
# reduced matrix is cross-validated — the canonical radiomics selection leak
# (mirrors pre_rt_cardiotox run_reproduce.py + models.py CV).
REDUCTION_LEAK_PIPELINE = textwrap.dedent(
    """
    import feature_reduction
    from sklearn.model_selection import StratifiedKFold

    def main(X_train, y):
        reduction = feature_reduction.fit_reduction(X_train, y, cmi_threshold=0.05)
        X_red = reduction.transform(X_train)
        cv = StratifiedKFold(n_splits=10)
        for tr, va in cv.split(X_red, y):
            fit_model(X_red[tr], y[tr])
    """
)

# Same reduction, but fit INSIDE each CV fold on the training rows only — safe.
REDUCTION_SAFE_INFOLD = textwrap.dedent(
    """
    import feature_reduction
    from sklearn.model_selection import StratifiedKFold

    def main(X_train, y):
        cv = StratifiedKFold(n_splits=10)
        for tr, va in cv.split(X_train, y):
            reduction = feature_reduction.fit_reduction(X_train[tr], y[tr])
            X_tr = reduction.transform(X_train[tr])
            fit_model(X_tr, y[tr])
    """
)


def test_c01_flags_outcome_aware_reduction_with_cv(tmp_path):
    p = _write(tmp_path, "leaky_reduce.py", REDUCTION_LEAK_PIPELINE)
    check = detect_c01_leakage(p)
    assert check.verdict is Verdict.FAIL
    assert any("fit_reduction" in e for e in check.evidence)


def test_c01_passes_infold_reduction(tmp_path):
    p = _write(tmp_path, "safe_reduce.py", REDUCTION_SAFE_INFOLD)
    check = detect_c01_leakage(p)
    # Fit is inside the CV-fold loop -> not flagged as fit-once leakage.
    assert check.verdict is Verdict.PASS


# --------------------------------------------------------------------------- #
# C07 EPV
# --------------------------------------------------------------------------- #

def test_c07_epv_explicit_pass():
    assert detect_c07_epv(n_events=100, n_features=8).verdict is Verdict.PASS


def test_c07_epv_explicit_partial():
    # 70 / 10 = 7 -> 5..10 -> PARTIAL
    assert detect_c07_epv(n_events=70, n_features=10).verdict is Verdict.PARTIAL


def test_c07_epv_explicit_fail():
    # 12 / 30 = 0.4 -> FAIL
    assert detect_c07_epv(n_events=12, n_features=30).verdict is Verdict.FAIL


def test_c07_epv_from_table():
    df = pd.DataFrame({
        "event": [0, 1, 1, 0, 1, 1, 1, 0],   # 5 events
        "f1": range(8), "f2": range(8), "f3": range(8),
    })
    # 5 events / 3 features = 1.67 -> FAIL
    check = detect_c07_epv(features=df, events_col="event")
    assert check.verdict is Verdict.FAIL
    assert "5/3" in check.message


# --------------------------------------------------------------------------- #
# C17 naming compliance
# --------------------------------------------------------------------------- #

def test_c17_naming_fraction():
    cols = [
        "patient_id",                       # ignored
        "original_glcm_Contrast",           # compliant
        "wavelet-LLH_firstorder_Mean",      # compliant
        "log-sigma-3-0-mm-3D_glrlm_GrayLevelNonUniformity",  # compliant
        "morph.vol",                        # compliant (IBSI code)
        "my_random_feature",                # NOT compliant
        "age",                              # NOT compliant
    ]
    check = detect_c17_naming(cols, ignore=["patient_id"])
    assert check.verdict is Verdict.GRADED
    # 4 of 6 candidate columns compliant
    assert check.fraction == pytest.approx(4 / 6)


def test_c17_naming_na_when_empty():
    check = detect_c17_naming(["patient_id"], ignore=["patient_id"])
    assert check.verdict is Verdict.NA


# --------------------------------------------------------------------------- #
# Scorecard weighting + gating cap
# --------------------------------------------------------------------------- #

def test_scorecard_weighted_pct_drops_na():
    card = Scorecard()
    card.add(Check("C07", Verdict.PASS))        # 1.0
    card.add(Check("C06", Verdict.FAIL))        # 0.0
    card.add(Check("C17", Verdict.GRADED, fraction=0.5))  # 0.5
    card.add(Check("C04", Verdict.NA))          # excluded
    # (1 + 0 + 0.5) / 3 applicable = 0.5 -> 50%
    assert card.weighted_pct == pytest.approx(50.0)
    assert card.n_applicable if hasattr(card, "n_applicable") else True
    assert len(card.applicable_checks) == 3


def test_scorecard_gating_cap():
    card = Scorecard()
    # high raw score but a failed GATE (C01) must cap band at moderate
    for cid in ("C07", "C06", "C17", "C24"):
        card.add(Check(cid, Verdict.PASS))
    card.add(Check("C01", Verdict.FAIL))   # gating fail
    assert card.weighted_pct > 60          # raw would be 'good'/'excellent'
    assert card.gate_capped is True
    assert card.failed_gates[0].id == "C01"
    assert card.band == "Moderate"
    assert card.banding_pct < 60


def test_scorecard_no_cap_when_gates_pass():
    card = Scorecard()
    for cid in ("C01", "C05", "C07", "C06", "C17"):
        card.add(Check(cid, Verdict.PASS))
    assert card.gate_capped is False
    assert card.band == "Excellent"


def test_scorecard_serialization_roundtrip():
    card = Scorecard()
    card.add(Check("C01", Verdict.FAIL, message="leak", evidence=["f.py:10"]))
    card.add(Check("C07", Verdict.GRADED, fraction=0.8))
    d = card.to_dict()
    assert d["gate_capped"] is True
    assert d["failed_gates"] == ["C01"]
    md = card.to_markdown()
    assert "Radiomics QA Scorecard" in md
    assert "C01" in md
    js = card.to_json()
    assert "weighted_pct" in js
