"""Multi-site harmonization and confounder residualization.

Two post-extraction corrections that the radiomics literature treats
as mandatory for multi-centre studies:

* **ComBat** (Johnson, Li & Rabinovic 2007) — removes site / scanner
  batch effects (location & scale shifts) per feature while preserving
  the biological signal carried by user-specified covariates. This is
  a dependency-free parametric empirical-Bayes implementation matching
  the canonical ``neuroCombat`` algorithm; we avoid the unmaintained
  ``neuroCombat`` package (last release 2018) for Python 3.11+ safety.

* **Linear residualization** — regresses each feature on continuous
  confounders (tumour / organ volume, blood glucose, ...) and keeps
  the residual, optionally re-adding the feature's global mean so the
  physical scale and downstream hazard-ratio interpretation survive
  (``preserve_scale=True``, the default).

Both operate on a tidy ``(n_samples, n_features)`` DataFrame and return
a DataFrame of the same shape so they chain cleanly after ``qr extract``
/ ``qr results merge``.

Caveat from the Panchal pre-RT cardiotox analysis: on a heart-bbox ROI
with only two sites, the *volume* confound dominates the *site*
confound, and feature-level residualization removes only a small,
statistically marginal slice of the AUC. These tools are correct and
necessary for genuinely multi-centre, multi-scanner cohorts, but they
are not a substitute for the source-decomposed modelling in
``experiments/pre_rt_cardiotox`` when the confound is volume rather
than scanner. Always report the before/after with bootstrap CIs.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["combat_harmonize", "residualize_linear"]


def _design_matrix(batch: np.ndarray,
                   covariates: Optional[np.ndarray]) -> tuple[np.ndarray, int]:
    """Build [batch one-hot | covariates] design; return (X, n_batch)."""
    batches = np.unique(batch)
    n_batch = len(batches)
    batch_oh = np.zeros((len(batch), n_batch), dtype=float)
    for j, b in enumerate(batches):
        batch_oh[batch == b, j] = 1.0
    if covariates is not None and covariates.size:
        X = np.hstack([batch_oh, covariates])
    else:
        X = batch_oh
    return X, n_batch


def _aprior(delta_hat: np.ndarray) -> float:
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (2 * s2 + m * m) / s2


def _bprior(delta_hat: np.ndarray) -> float:
    m = delta_hat.mean()
    s2 = delta_hat.var()
    return (m * s2 + m ** 3) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _it_eb(s_data, gamma_hat, delta_hat, gamma_bar, t2, a_prior, b_prior,
           conv=1e-4, max_iter=500):
    """Iterative parametric empirical-Bayes shrinkage for one batch."""
    n = (~np.isnan(s_data)).sum(axis=1)
    g_old = gamma_hat.copy()
    d_old = delta_hat.copy()
    change = 1.0
    it = 0
    while change > conv and it < max_iter:
        g_new = _postmean(gamma_hat, gamma_bar, n, d_old, t2)
        resid = s_data - g_new[:, None]
        sum2 = np.nansum(resid * resid, axis=1)
        d_new = _postvar(sum2, n, a_prior, b_prior)
        change = max(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-12)),
                     np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-12)))
        g_old, d_old = g_new, d_new
        it += 1
    return g_old, d_old


def combat_harmonize(df: pd.DataFrame,
                     feature_cols: Sequence[str],
                     batch_col: str,
                     *,
                     categorical_covariates: Sequence[str] = (),
                     continuous_covariates: Sequence[str] = (),
                     parametric: bool = True,
                     eb: bool = True) -> pd.DataFrame:
    """ComBat-harmonize ``feature_cols`` across ``batch_col``.

    Args:
        df: tidy DataFrame, one row per sample.
        feature_cols: radiomics feature columns to harmonize.
        batch_col: site / scanner column (the batch).
        categorical_covariates: biological categorical columns to
            preserve (e.g. the outcome label) — one-hot encoded into
            the design so ComBat does not erase outcome-correlated
            variance.
        continuous_covariates: biological continuous columns to
            preserve (e.g. organ volume).
        parametric: parametric empirical Bayes (True) vs raw L/S (eb=False).
        eb: apply empirical-Bayes shrinkage. If False, use the raw
            per-batch gamma_hat / delta_hat.

    Returns:
        A copy of ``df`` with ``feature_cols`` harmonized in place;
        non-feature columns are untouched.
    """
    work = df.copy()
    batch = work[batch_col].astype(str).to_numpy()
    batches = np.unique(batch)
    n_batch = len(batches)
    if n_batch < 2:
        # nothing to harmonize
        return work

    # covariate design (categorical one-hot + continuous)
    cov_parts = []
    for c in categorical_covariates:
        codes = work[c].astype("category").cat.codes.to_numpy()
        levels = np.unique(codes)
        # drop first level to avoid collinearity with the intercept the
        # batch one-hot already supplies
        for lv in levels[1:]:
            cov_parts.append((codes == lv).astype(float)[:, None])
    for c in continuous_covariates:
        v = pd.to_numeric(work[c], errors="coerce").to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=np.nanmean(v))
        cov_parts.append(v[:, None])
    covariates = np.hstack(cov_parts) if cov_parts else None

    Z = work[list(feature_cols)].to_numpy(dtype=float).T  # (features, samples)
    n_feat, n_samp = Z.shape

    X, _ = _design_matrix(batch, covariates)            # (samples, n_batch + n_cov)
    # least squares: B_hat (params x features)
    B_hat, *_ = np.linalg.lstsq(X, Z.T, rcond=None)
    # grand mean = batch-size-weighted average of the batch coefficients
    batch_sizes = np.array([(batch == b).sum() for b in batches], dtype=float)
    grand_mean = (batch_sizes / n_samp) @ B_hat[:n_batch, :]      # (features,)
    # pooled variance from residuals
    resid = Z - (X @ B_hat).T
    var_pooled = (resid ** 2).mean(axis=1)               # (features,)
    var_pooled[var_pooled == 0] = 1e-12

    # standardize
    stand_mean = grand_mean[:, None] @ np.ones((1, n_samp))
    if covariates is not None:
        tmp = (X[:, n_batch:] @ B_hat[n_batch:, :]).T
        stand_mean = stand_mean + tmp
    Z_std = (Z - stand_mean) / np.sqrt(var_pooled)[:, None]

    # batch design for L/S
    batch_idx = [np.where(batch == b)[0] for b in batches]
    gamma_hat = np.zeros((n_batch, n_feat))
    delta_hat = np.zeros((n_batch, n_feat))
    for i, idx in enumerate(batch_idx):
        gamma_hat[i] = Z_std[:, idx].mean(axis=1)
        delta_hat[i] = Z_std[:, idx].var(axis=1)
        delta_hat[i][delta_hat[i] == 0] = 1e-12

    if eb and parametric:
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)
        for i, idx in enumerate(batch_idx):
            gamma_bar = gamma_hat[i].mean()
            t2 = gamma_hat[i].var()
            a_prior = _aprior(delta_hat[i])
            b_prior = _bprior(delta_hat[i])
            g_star, d_star = _it_eb(Z_std[:, idx], gamma_hat[i], delta_hat[i],
                                    gamma_bar, t2, a_prior, b_prior)
            gamma_star[i] = g_star
            delta_star[i] = d_star
    else:
        gamma_star, delta_star = gamma_hat, delta_hat

    # adjust
    Z_adj = Z_std.copy()
    for i, idx in enumerate(batch_idx):
        Z_adj[:, idx] = ((Z_std[:, idx] - gamma_star[i][:, None])
                         / np.sqrt(delta_star[i])[:, None])
    Z_adj = Z_adj * np.sqrt(var_pooled)[:, None] + stand_mean

    work[list(feature_cols)] = Z_adj.T
    return work


def residualize_linear(df: pd.DataFrame,
                       feature_cols: Sequence[str],
                       confounders: Sequence[str],
                       *,
                       preserve_scale: bool = True) -> pd.DataFrame:
    """Regress each feature on ``confounders`` and keep the residual.

    With ``preserve_scale=True`` the feature's global mean is re-added
    so the output keeps the original physical range (important for
    hazard-ratio interpretability); otherwise residuals are centred at
    zero.
    """
    work = df.copy()
    conf = [c for c in confounders if c in work.columns]
    if not conf:
        return work
    X = work[conf].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean()).to_numpy(dtype=float)
    X = np.hstack([np.ones((len(X), 1)), X])             # intercept
    for feat in feature_cols:
        y = pd.to_numeric(work[feat], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < X.shape[1] + 1:
            continue
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        resid = y - X @ beta
        if preserve_scale:
            resid = resid + np.nanmean(y)
        work[feat] = resid
    return work
