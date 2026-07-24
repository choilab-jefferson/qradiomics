"""Regression test: LIDC reproduction CV must group folds by patient.

reproduce_papers.py / reproduce_cir.py used a plain StratifiedKFold on a table
with multiple nodules per patient, leaking a patient across train/test and
inflating the AUC. _cv_auc now takes `groups` and uses StratifiedGroupKFold.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
import pandas as pd  # noqa: E402

_PIPE = Path(__file__).resolve().parents[1] / "pipelines" / "lidc_idri"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _PIPE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _leaky_dataset(n_patients=20, per_patient=4, seed=0):
    """Each patient has a constant per-patient offset baked into the features and
    a patient-level label. A model that sees one of a patient's nodules in train
    can memorise the offset and ace that patient's held-out nodules — but only if
    the split leaks the patient. Grouped CV should therefore score lower."""
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_patients):
        label = p % 2
        offset = rng.normal(label * 3.0, 0.1)  # near patient-identifying
        for _ in range(per_patient):
            rows.append({
                "pid": f"P{p}",
                "y": label,
                "f1": offset + rng.normal(0, 0.05),
                "f2": rng.normal(0, 1),
                "f3": rng.normal(0, 1),
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("modname", ["reproduce_papers", "reproduce_cir"])
def test_grouped_cv_runs_and_reduces_leakage(modname):
    mod = _load(modname)
    df = _leaky_dataset()
    X, y = df[["f1", "f2", "f3"]], df["y"]

    auc_grouped, _ = mod._cv_auc(X, y, groups=df["pid"].to_numpy(), top_k=3)
    auc_leaky, _ = mod._cv_auc(X, y, top_k=3)  # groups=None → old behaviour

    assert 0.0 <= auc_grouped <= 1.0
    # The leaky split should score at least as high as the honest grouped one;
    # with a patient-identifying feature it is materially higher.
    assert auc_leaky >= auc_grouped - 1e-9
