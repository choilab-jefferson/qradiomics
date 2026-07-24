"""Tests for qradiomics.analytics — ComBat harmonization, linear
residualization, and ICC-based robustness filtering."""
import numpy as np
import pandas as pd
import pytest

from qradiomics.analytics import (
    combat_harmonize,
    residualize_linear,
    feature_icc,
    icc_filter,
)


@pytest.fixture
def synthetic_batched():
    """Two sites; site B has a +3 location shift and 2x scale.
    A continuous covariate carries the biological signal."""
    rng = np.random.default_rng(0)
    n_per, n_feat = 80, 15
    covar = rng.normal(0, 1, 2 * n_per)
    true = (covar[:, None] * rng.normal(0, 1, (1, n_feat))
            + rng.normal(0, 1, (2 * n_per, n_feat)))
    batch = np.array(["A"] * n_per + ["B"] * n_per)
    loc = np.where(batch == "B", 3.0, 0.0)[:, None]
    scale = np.where(batch == "B", 2.0, 1.0)[:, None]
    obs = true * scale + loc
    fcols = [f"f{i}" for i in range(n_feat)]
    df = pd.DataFrame(obs, columns=fcols)
    df["site"] = batch
    df["covar"] = covar
    df["PID"] = range(len(df))
    return df, fcols


class TestComBat:
    def test_removes_location_effect(self, synthetic_batched):
        df, fcols = synthetic_batched
        before = abs(df[df.site == "A"][fcols].mean()
                     - df[df.site == "B"][fcols].mean()).mean()
        harm = combat_harmonize(df, fcols, "site", continuous_covariates=["covar"])
        after = abs(harm[harm.site == "A"][fcols].mean()
                    - harm[harm.site == "B"][fcols].mean()).mean()
        assert after < before * 0.25

    def test_preserves_biological_signal(self, synthetic_batched):
        df, fcols = synthetic_batched
        harm = combat_harmonize(df, fcols, "site", continuous_covariates=["covar"])
        corr_before = np.mean([np.corrcoef(df[f], df.covar)[0, 1] for f in fcols])
        corr_after = np.mean([np.corrcoef(harm[f], harm.covar)[0, 1] for f in fcols])
        assert abs(corr_after - corr_before) < 0.12

    def test_single_site_is_noop(self, synthetic_batched):
        df, fcols = synthetic_batched
        df = df.copy()
        df["site"] = "only"
        harm = combat_harmonize(df, fcols, "site")
        pd.testing.assert_frame_equal(harm[fcols], df[fcols])


class TestResidualize:
    def test_removes_confounder_correlation(self, synthetic_batched):
        df, fcols = synthetic_batched
        # make f0 strongly volume-dependent
        df = df.copy()
        df["volume"] = df["covar"] * 5 + np.random.default_rng(1).normal(0, 0.1, len(df))
        df["f0"] = df["volume"] * 2 + 1.0
        res = residualize_linear(df, ["f0"], ["volume"])
        r_after = abs(np.corrcoef(res.f0, res.volume)[0, 1])
        assert r_after < 0.1

    def test_scale_preserved(self, synthetic_batched):
        df, fcols = synthetic_batched
        res = residualize_linear(df, fcols, ["covar"], preserve_scale=True)
        for f in fcols:
            assert abs(res[f].mean() - df[f].mean()) < 1e-6

    def test_scale_centred_when_disabled(self, synthetic_batched):
        df, fcols = synthetic_batched
        res = residualize_linear(df, fcols, ["covar"], preserve_scale=False)
        for f in fcols:
            assert abs(res[f].mean()) < 1e-6


class TestICC:
    def test_identical_series_icc_one(self, synthetic_batched):
        df, fcols = synthetic_batched
        icc = feature_icc(df["f0"], df["f0"], df["PID"])
        assert icc > 0.99

    def test_icc_filter_keeps_robust(self, synthetic_batched):
        df, fcols = synthetic_batched
        harm = combat_harmonize(df, fcols, "site", continuous_covariates=["covar"])
        kept, table = icc_filter(df, harm, fcols, "PID", threshold=0.0)
        assert len(kept) == len(fcols)
        assert table["icc"].notna().all()
