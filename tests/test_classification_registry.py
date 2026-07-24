"""Tests for qradiomics.classification.registry — model registry membership,
and the TPOT AutoML backend added for qradiomics#8 (opt-in only, helpful
ImportError when tpot is not installed, sklearn-compatible fit/predict).

Any test that exercises the TPOT registry entry with a real fit is forced to
a tiny/bounded config (generations=1, population_size=2, cv=2) on a ~20-row
synthetic dataset so it completes in a few seconds rather than running a
real genetic search (see qradiomics#8 spec) -- it is additionally marked
`slow` so quick local runs can deselect it with `-m "not slow"`.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from qradiomics.classification.registry import (
    ALL_MODELS,
    DEFAULT_MODELS,
    EXTENDED_MODELS,
    MODEL_REGISTRY,
    OPT_IN_ONLY_MODELS,
    build_model,
)


class TestRegistryMembership:
    def test_tpot_is_registered(self):
        assert "TPOT" in MODEL_REGISTRY

    def test_all_models_means_literally_every_registered_model(self):
        # ALL_MODELS is a public symbol; its name must match its contents —
        # TPOT included, even though it is opt-in-only for the *default*
        # model set (DEFAULT_MODELS below).
        assert "TPOT" in ALL_MODELS

    def test_tpot_excluded_from_default_models(self):
        assert "TPOT" not in DEFAULT_MODELS

    def test_tpot_excluded_from_extended_models(self):
        assert "TPOT" not in EXTENDED_MODELS

    def test_tpot_is_opt_in_only(self):
        assert "TPOT" in OPT_IN_ONLY_MODELS

    def test_all_models_equals_registry(self):
        # True by construction now that ALL_MODELS = tuple(MODEL_REGISTRY).
        assert set(ALL_MODELS) == set(MODEL_REGISTRY)

    def test_default_models_is_subset_of_registry(self):
        assert set(DEFAULT_MODELS) <= set(MODEL_REGISTRY)

    def test_flaml_unaffected_still_present_everywhere(self):
        # Guards against accidentally applying the new opt-in exclusion to
        # the pre-existing AutoML entry.
        assert "FLAML" in MODEL_REGISTRY
        assert "FLAML" in ALL_MODELS
        assert "FLAML" in DEFAULT_MODELS
        assert "FLAML" in EXTENDED_MODELS


class TestTpotImportError:
    def test_missing_dependency_raises_helpful_import_error(self, monkeypatch):
        # Simulate `tpot` not being installed: any `import tpot` (or
        # `from tpot import ...`) raises ImportError while the module is
        # present-but-None in sys.modules.
        for mod in list(sys.modules):
            if mod == "tpot" or mod.startswith("tpot."):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        monkeypatch.setitem(sys.modules, "tpot", None)

        with pytest.raises(ImportError, match=r"pip install qradiomics\[automl\]"):
            build_model("TPOT", random_state=0)


class TestTpotEndToEndTiny:
    """Real (but tiny/bounded) tpot fit -- no mocking of tpot itself, but
    generations/population_size/cv are forced to the minimum via
    Pipeline.set_params so the genetic search is trivial."""

    @pytest.mark.slow
    def test_fit_predict_predict_proba_tiny(self):
        tpot = pytest.importorskip("tpot")
        del tpot
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=20, n_features=4, n_informative=3, n_redundant=0,
            random_state=0,
        )
        pipe, grid = build_model("TPOT", random_state=0)
        assert grid == {}  # AutoML backends manage their own tuning

        pipe.set_params(
            clf__generations=1, clf__population_size=2, clf__cv=2,
            clf__max_time_mins=1, clf__n_jobs=1,
        )
        pipe.fit(X, y)

        proba = pipe.predict_proba(X[:5])
        assert proba.shape == (5, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

        preds = pipe.predict(X[:5])
        assert preds.shape == (5,)
        assert set(np.unique(preds)) <= {0, 1}

        assert set(pipe.named_steps["clf"].classes_.tolist()) <= {0, 1}


class TestBuildModelUnknown:
    def test_unknown_model_raises_keyerror(self):
        with pytest.raises(KeyError):
            build_model("NOT_A_REAL_MODEL")
