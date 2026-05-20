"""Tests for qradiomics.delta — cross-timepoint feature derivation.

API (mirrored from qradiomics-dev):
    DeltaPair(early=str, late=str)
    compute_delta(features, pair, *, feature_cols=None,
                  key_cols=('patient_id', 'roi_name'),
                  timepoint_col='timepoint')
    compute_trend(features, *, feature_cols=None,
                  key_cols=('patient_id', 'roi_name'),
                  timepoint_col='timepoint',
                  time_index_col=None)
"""
import numpy as np
import pandas as pd
import pytest

from qradiomics.delta import DeltaPair, compute_delta, compute_trend


def _multi_tp_frame(patients=("A", "B"),
                    tps=(("b", 0), ("f", 28)),
                    roi="GTV"):
    rows = []
    for pid in patients:
        for tp, day in tps:
            offset = 0 if tp == "b" else 5 if tp == "f" else 12
            rows.append({
                "patient_id": pid,
                "roi_name": roi,
                "timepoint": tp,
                "relative_day": day,
                "f1": 10.0 + offset,
                "f2": 100.0 - offset,
            })
    return pd.DataFrame(rows)


class TestDelta:
    def test_basic_two_patients(self):
        df = _multi_tp_frame()
        d = compute_delta(df, DeltaPair(early="b", late="f"))
        # one row per (patient, roi) — both have b and f
        assert len(d) == 2
        # delta = late - early (f - b) → 5.0 for f1
        col_f1 = [c for c in d.columns if c.endswith("f1")][0]
        assert all(d[col_f1] == 5.0)

    def test_missing_timepoint_drops_patient(self):
        """compute_delta in dev only emits rows where BOTH early and late
        timepoints are present per (patient, roi). Patients missing one
        are dropped entirely from the output."""
        df = _multi_tp_frame()
        df = df.iloc[:3]  # drop B's "f" row → only A has both b and f
        d = compute_delta(df, DeltaPair(early="b", late="f"))
        assert set(d["patient_id"]) == {"A"}

    def test_nonexistent_timepoint(self):
        df = _multi_tp_frame()
        # Asking for non-existent late timepoint
        d = compute_delta(df, DeltaPair(early="b", late="nope"))
        col_f1 = [c for c in d.columns if c.endswith("f1")][0]
        assert d[col_f1].isna().all()

    def test_feature_cols_subset(self):
        df = _multi_tp_frame()
        d = compute_delta(df, DeltaPair(early="b", late="f"),
                          feature_cols=["f1"])
        # Only f1 derived; f2 absent
        assert any("f1" in c for c in d.columns)
        assert not any("f2" in c for c in d.columns)


class TestTrend:
    def test_three_timepoints_linear(self):
        df = pd.DataFrame([
            {"patient_id": "A", "roi_name": "GTV", "timepoint": "b",
             "relative_day": 0, "f": 1.0},
            {"patient_id": "A", "roi_name": "GTV", "timepoint": "m",
             "relative_day": 14, "f": 4.0},
            {"patient_id": "A", "roi_name": "GTV", "timepoint": "f",
             "relative_day": 28, "f": 7.0},
        ])
        t = compute_trend(df, time_index_col="relative_day")
        # slope (rise / run) for 1→7 over 0→28 = 6/28 ≈ 0.214
        slope_col = [c for c in t.columns if "slope" in c.lower() or "trend" in c.lower()][0]
        assert abs(float(t[slope_col].iloc[0]) - (6 / 28)) < 1e-9

    def test_two_timepoints_only(self):
        """With time_index_col missing, trend may fall back or yield NaN — just
        verify the call doesn't blow up."""
        df = pd.DataFrame([
            {"patient_id": "A", "roi_name": "GTV", "timepoint": "b",
             "relative_day": 0, "f": 1.0},
            {"patient_id": "A", "roi_name": "GTV", "timepoint": "f",
             "relative_day": 28, "f": 7.0},
        ])
        t = compute_trend(df, time_index_col="relative_day")
        # Should produce some result without crashing
        assert len(t) >= 1
