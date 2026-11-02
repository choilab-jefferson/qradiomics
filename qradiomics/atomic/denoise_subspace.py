"""Outcome-free repeat/short-interval noise covariance whitening and SNR subspace projection.

Consolidates outcome-free signal-vs-noise decomposition for high-dimensional radiomics
and foundation-model embeddings:
  - Test-retest and monotonic-trajectory channel gating (``select_stable_dims``,
    ``gate_retest_channels``, ``mono_pass``).
  - Shrinkage noise covariance whitening ``W = Sigma_noise^{-1/2}`` where
    ``Sigma_noise = (1 - lam) * Cov(Z_noise) + lam * I`` (``whitener``).
  - Generalized eigenvalue SNR subspace projection maximizing ``Var_signal / Var_noise``
    (``SubspaceProjector``, ``fit_noise_whitening_and_snr_subspace``,
    ``apply_subspace_projection``).
  - Volume residualization / orthogonalization (``residualize_against_volume``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_SHRINKAGE_LAMBDA: float = 0.5
DEFAULT_RHO_GATE: float = 0.90

__all__ = [
    "DEFAULT_RHO_GATE",
    "DEFAULT_SHRINKAGE_LAMBDA",
    "SubspaceProjector",
    "apply_subspace_projection",
    "compute_noise_scale_mad",
    "fit_noise_whitening_and_snr_subspace",
    "fit_transform_ctfm_blocks",
    "gate_retest_channels",
    "mono_pass",
    "pca_scores",
    "residualize_against_volume",
    "select_stable_dims",
    "whitener",
]


def compute_noise_scale_mad(
    noise_deltas: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute robust per-channel noise scale ``s_j = 1.4826 * MAD(delta_noise_j)``."""
    N = np.asarray(noise_deltas, dtype=float)
    if N.ndim == 1:
        N = N[:, None]
    med = np.nanmedian(N, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(N - med), axis=0)
    return np.where(np.isfinite(mad) & (mad > eps), mad, 1.0)


def whitener(
    Nz: np.ndarray,
    lam: float = DEFAULT_SHRINKAGE_LAMBDA,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute symmetric inverse square root ``Sigma_noise^{-1/2}`` for shrinkage noise covariance.

    Given noise deltas ``Nz`` (already scaled by per-channel noise floor ``s_j``), forms:
        ``Sigma_noise = (1 - lam) * Cov(Nz) + lam * I``
    and returns ``W = V * diag(ev^{-1/2}) * V^T``.
    """
    Nz_arr = np.asarray(Nz, dtype=float)
    if Nz_arr.ndim == 1:
        Nz_arr = Nz_arr[:, None]
    d = Nz_arr.shape[1]
    if len(Nz_arr) <= 1:
        return np.eye(d, dtype=float)
    C = np.cov(Nz_arr, rowvar=False) if d > 1 else np.array([[np.var(Nz_arr[:, 0], ddof=1)]])
    C = np.nan_to_num(np.atleast_2d(C), nan=0.0, posinf=0.0, neginf=0.0)
    S = (1.0 - float(lam)) * C + float(lam) * np.eye(d, dtype=float)
    ev, V = np.linalg.eigh(S)
    ev_clamped = np.maximum(ev, eps)
    return (V * (ev_clamped**-0.5)) @ V.T


def mono_pass(RH: np.ndarray, null: float) -> np.ndarray:
    """Within-patient monotonic trend filter: share(|rho| >= 0.8) >= 2 * null and |median(rho)| >= 0.3."""
    RH_arr = np.asarray(RH, dtype=float)
    share = np.nanmean(np.abs(RH_arr) >= 0.8 - 1e-9, axis=0)
    med = np.nanmedian(RH_arr, axis=0)
    return (share >= 2.0 * float(null)) & (np.abs(med) >= 0.3)


def gate_retest_channels(
    cols: Sequence[str],
    rho_vec: np.ndarray,
    mad_vec: np.ndarray,
    cidx: Optional[Mapping[str, int]] = None,
    rho_gate: float = DEFAULT_RHO_GATE,
) -> List[str]:
    """Filter channel names meeting test-retest Spearman ``rho >= rho_gate`` and ``MAD > 0``."""
    if cidx is None:
        cidx = {c: i for i, c in enumerate(cols)}
    return [
        c
        for c in cols
        if np.isfinite(rho_vec[cidx[c]])
        and float(rho_vec[cidx[c]]) >= float(rho_gate)
        and float(mad_vec[cidx[c]]) > 0.0
    ]


def select_stable_dims(
    retest_pair_a: Optional[np.ndarray] = None,
    retest_pair_b: Optional[np.ndarray] = None,
    *,
    rho_vec: Optional[np.ndarray] = None,
    mad_vec: Optional[np.ndarray] = None,
    noise_deltas: Optional[np.ndarray] = None,
    rho_gate: float = DEFAULT_RHO_GATE,
    min_dims: int = 1,
) -> np.ndarray:
    """Select outcome-free stable feature dimensions by test-retest rank correlation and positive MAD.

    Can be called either with raw retest pair matrices ``(retest_pair_a, retest_pair_b)``
    of shape ``(N_pairs, D)`` or precomputed ``(rho_vec, mad_vec)``.

    Returns:
        Boolean mask of shape ``(D,)`` (or integer index array if ``min_dims`` fallback is needed,
        still returned as boolean mask of length ``D`` with at least ``min_dims`` True entries).
    """
    if rho_vec is None:
        if retest_pair_a is None or retest_pair_b is None:
            raise ValueError(
                "Provide either (retest_pair_a, retest_pair_b) or precomputed rho_vec."
            )
        A = np.asarray(retest_pair_a, dtype=float)
        B = np.asarray(retest_pair_b, dtype=float)
        if A.shape != B.shape or A.ndim != 2:
            raise ValueError(f"Expected matching 2D arrays, got {A.shape} and {B.shape}")
        d = A.shape[1]
        rho_vec = np.array(
            [
                (
                    spearmanr(A[:, j], B[:, j])[0]
                    if np.ptp(A[:, j]) > 0 and np.ptp(B[:, j]) > 0
                    else np.nan
                )
                for j in range(d)
            ],
            dtype=float,
        )
        if mad_vec is None and noise_deltas is None:
            noise_deltas = B - A

    rho_arr = np.asarray(rho_vec, dtype=float)
    d = len(rho_arr)
    if mad_vec is None:
        if noise_deltas is not None:
            N = np.asarray(noise_deltas, dtype=float)
            med = np.nanmedian(N, axis=0)
            mad_arr = 1.4826 * np.nanmedian(np.abs(N - med), axis=0)
        else:
            mad_arr = np.ones(d, dtype=float)
    else:
        mad_arr = np.asarray(mad_vec, dtype=float)

    mask = (
        np.isfinite(rho_arr) & (rho_arr >= float(rho_gate)) & np.isfinite(mad_arr) & (mad_arr > 0)
    )
    if int(mask.sum()) < min_dims:
        # Fallback to top-k finite rho channels with positive MAD
        valid_score = np.where(np.isfinite(rho_arr) & (mad_arr > 0), rho_arr, -np.inf)
        top_idx = np.argsort(valid_score)[::-1][: max(min_dims, 1)]
        mask = np.zeros(d, dtype=bool)
        mask[top_idx] = True
    return mask


@dataclass
class SubspaceProjector:
    """Outcome-free noise-whitened generalized-eigenvalue SNR subspace projector.

    Attributes:
        selected_dims: Boolean mask ``(D_orig,)`` of retained channels before whitening.
        noise_scale: Per-channel robust noise scale ``s_j`` of shape ``(D_sel,)``.
        whitening_matrix: Symmetric matrix ``W = Sigma_noise^{-1/2}`` of shape ``(D_sel, D_sel)``.
        mean_whitened: Training-set mean in whitened space of shape ``(D_sel,)``.
        components: Orthonormal projection eigenvectors ``(k, D_sel)`` maximizing SNR.
        component_scale: Standard deviation of projected training scores ``(k,)``.
        eigenvalues_snr: Generalized SNR eigenvalues ``(k,)`` in descending order.
        lam: Shrinkage parameter ``lam`` used for ``Sigma_noise``.
    """

    selected_dims: np.ndarray
    noise_scale: np.ndarray
    whitening_matrix: np.ndarray
    mean_whitened: np.ndarray
    components: np.ndarray
    component_scale: np.ndarray
    eigenvalues_snr: np.ndarray
    lam: float = DEFAULT_SHRINKAGE_LAMBDA

    def transform(self, X: np.ndarray, *, standardize_scores: bool = True) -> np.ndarray:
        """Project feature matrix ``X`` of shape ``(N, D_orig)`` or ``(N, D_sel)`` into SNR subspace."""
        return apply_subspace_projection(X, self, standardize_scores=standardize_scores)


def fit_noise_whitening_and_snr_subspace(
    X_signal: np.ndarray,
    noise_deltas: np.ndarray,
    *,
    n_components: int = 8,
    lam: float = DEFAULT_SHRINKAGE_LAMBDA,
    selected_dims: Optional[np.ndarray] = None,
) -> SubspaceProjector:
    """Fit outcome-free noise covariance whitening ``W = Sigma_noise^{-1/2}`` and SNR subspace.

    Solves the generalized eigenvalue problem maximizing ``Var_signal / Var_noise`` by:
      1. Scaling each channel by its robust repeat/short-interval noise MAD ``s_j``.
      2. Estimating regularized noise correlation ``Sigma_noise = (1 - lam) Cov(N / s) + lam I``
         and its symmetric inverse square root ``W = Sigma_noise^{-1/2}``.
      3. Projecting ``X_signal`` into whitened noise-isotropic coordinates ``Xw = (X / s) @ W``.
      4. Computing the top ``n_components`` principal eigenvectors of centered ``Xw``
         (whose eigenvalues equal the generalized signal-to-noise variance ratios).

    Args:
        X_signal: Feature matrix ``(N_samples, D)`` used to estimate total/signal covariance.
        noise_deltas: Repeat or short-interval difference matrix ``(N_pairs, D)`` representing pure noise.
        n_components: Number of SNR subspace dimensions ``k`` to retain.
        lam: Shrinkage regularization toward identity ``(0 <= lam <= 1)``.
        selected_dims: Optional boolean or index mask selecting a channel subset first.

    Returns:
        Fitted :class:`SubspaceProjector`.
    """
    X = np.asarray(X_signal, dtype=float)
    N = np.asarray(noise_deltas, dtype=float)
    if X.ndim != 2 or N.ndim != 2 or X.shape[1] != N.shape[1]:
        raise ValueError(
            f"X_signal {X.shape} and noise_deltas {N.shape} must be 2D with matching column count."
        )

    d_total = X.shape[1]
    if selected_dims is None:
        sel_mask = np.ones(d_total, dtype=bool)
    else:
        sel_arr = np.asarray(selected_dims)
        if sel_arr.dtype == bool:
            sel_mask = sel_arr.copy()
        else:
            sel_mask = np.zeros(d_total, dtype=bool)
            sel_mask[sel_arr] = True

    X_sel = X[:, sel_mask]
    N_sel = N[:, sel_mask]
    s = compute_noise_scale_mad(N_sel)
    Nz = N_sel / s
    W = whitener(Nz, lam=lam)

    Xw = (X_sel / s) @ W
    mu_w = np.nanmean(Xw, axis=0)
    Xw_centered = np.nan_to_num(Xw - mu_w, nan=0.0)

    k = max(1, min(int(n_components), Xw_centered.shape[1], max(len(Xw_centered) - 1, 1)))
    if len(Xw_centered) > 1 and Xw_centered.shape[1] > 1:
        pca = PCA(n_components=k, svd_solver="full").fit(Xw_centered)
        comps = pca.components_  # (k, D_sel)
        ev_snr = pca.explained_variance_
        scores = Xw_centered @ comps.T
    else:
        comps = np.eye(Xw_centered.shape[1], dtype=float)[:k]
        ev_snr = np.var(Xw_centered @ comps.T, axis=0)
        scores = Xw_centered @ comps.T

    sd = np.std(scores, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)

    return SubspaceProjector(
        selected_dims=sel_mask,
        noise_scale=s,
        whitening_matrix=W,
        mean_whitened=mu_w,
        components=comps,
        component_scale=sd,
        eigenvalues_snr=ev_snr,
        lam=float(lam),
    )


def apply_subspace_projection(
    X: np.ndarray,
    projector: SubspaceProjector,
    *,
    standardize_scores: bool = True,
) -> np.ndarray:
    """Apply a fitted :class:`SubspaceProjector` to new samples ``X``."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[None, :]
    d_sel = int(projector.selected_dims.sum())
    if X_arr.shape[1] == len(projector.selected_dims) and d_sel != len(projector.selected_dims):
        X_sel = X_arr[:, projector.selected_dims]
    elif X_arr.shape[1] == d_sel:
        X_sel = X_arr
    else:
        raise ValueError(
            f"Input feature count {X_arr.shape[1]} does not match projector input ({len(projector.selected_dims)}) or selected ({d_sel}) dims."
        )

    Xw = (X_sel / projector.noise_scale) @ projector.whitening_matrix
    Xw_centered = np.nan_to_num(Xw - projector.mean_whitened, nan=0.0)
    scores = Xw_centered @ projector.components.T
    if standardize_scores:
        scores = scores / projector.component_scale
    return scores


def residualize_against_volume(
    X: np.ndarray,
    log_volume: np.ndarray,
    *,
    X_eval: Optional[np.ndarray] = None,
    log_volume_eval: Optional[np.ndarray] = None,
    standardize: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Orthogonalize feature columns ``X`` against log lesion volume via OLS linear regression.

    Fits ``X[:, j] ~ alpha_j + beta_j * log_volume`` on ``(X, log_volume)`` and returns the
    volume-independent residuals. If ``X_eval`` and ``log_volume_eval`` are provided, applies
    the training-fitted regression coefficients to ``X_eval`` without leakage.
    """
    X_tr = np.asarray(X, dtype=float)
    v_tr = np.asarray(log_volume, dtype=float).ravel()
    if len(X_tr) != len(v_tr):
        raise ValueError(f"Length mismatch: X has {len(X_tr)} rows, log_volume has {len(v_tr)}")

    squeeze_1d = X_tr.ndim == 1
    if squeeze_1d:
        X_tr = X_tr[:, None]

    design_tr = np.column_stack([np.ones(len(v_tr), dtype=float), v_tr])
    coef, _, _, _ = np.linalg.lstsq(design_tr, X_tr, rcond=None)
    res_tr = X_tr - design_tr @ coef

    if standardize:
        sd = np.std(res_tr, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        res_tr = res_tr / sd
    else:
        sd = np.ones(X_tr.shape[1], dtype=float)

    if X_eval is not None:
        if log_volume_eval is None:
            raise ValueError("log_volume_eval is required when X_eval is provided.")
        X_te = np.asarray(X_eval, dtype=float)
        if squeeze_1d and X_te.ndim == 1:
            X_te = X_te[:, None]
        v_te = np.asarray(log_volume_eval, dtype=float).ravel()
        design_te = np.column_stack([np.ones(len(v_te), dtype=float), v_te])
        res_te = (X_te - design_te @ coef) / sd
        if squeeze_1d:
            return res_tr[:, 0], res_te[:, 0]
        return res_tr, res_te

    return res_tr[:, 0] if squeeze_1d else res_tr


def pca_scores(
    A: np.ndarray,
    B: np.ndarray,
    k: int,
    centered: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standardized PCA score projection for train (``A``) and test (``B``)."""
    A_arr = np.asarray(A, dtype=float)
    B_arr = np.asarray(B, dtype=float)
    if centered:
        mu = A_arr.mean(axis=0)
        A_proc, B_proc = A_arr - mu, B_arr - mu
    else:
        sc = StandardScaler().fit(A_arr)
        A_proc, B_proc = sc.transform(A_arr), sc.transform(B_arr)
    nc = max(1, min(int(k), A_proc.shape[1], max(len(A_proc) - 1, 1)))
    pca = PCA(n_components=nc, svd_solver="full").fit(A_proc)
    P, Q = pca.transform(A_proc), pca.transform(B_proc)
    mu_p, sd_p = P.mean(axis=0), P.std(axis=0)
    sd_p = np.where(sd_p > 0, sd_p, 1.0)
    return (P - mu_p) / sd_p, (Q - mu_p) / sd_p


def fit_transform_ctfm_blocks(
    W_tr: np.ndarray,
    C_tr: np.ndarray,
    W_te: np.ndarray,
    C_te: np.ndarray,
    k: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit and transform two-block embeddings (Block 1: whitened SNR, Block 2: core pyramid)."""
    w_mu = W_tr.mean(axis=0)
    nc_w = max(1, min(int(k), len(W_tr) - 1, W_tr.shape[1]))
    pca_w = PCA(n_components=nc_w, svd_solver="full").fit(W_tr - w_mu)
    Zw_tr = pca_w.transform(W_tr - w_mu)
    Zw_te = pca_w.transform(W_te - w_mu)
    sd_w = Zw_tr.std(axis=0)
    sd_w = np.where(sd_w > 0, sd_w, 1.0)
    Zw_tr, Zw_te = Zw_tr / sd_w, Zw_te / sd_w

    sc_c = StandardScaler().fit(C_tr)
    c_tr_sc, c_te_sc = sc_c.transform(C_tr), sc_c.transform(C_te)
    nc_c = max(1, min(int(k), len(C_tr) - 1, C_tr.shape[1]))
    pca_c = PCA(n_components=nc_c, svd_solver="full").fit(c_tr_sc)
    Zc_tr = pca_c.transform(c_tr_sc)
    Zc_te = pca_c.transform(c_te_sc)
    sd_c = Zc_tr.std(axis=0)
    sd_c = np.where(sd_c > 0, sd_c, 1.0)
    Zc_tr, Zc_te = Zc_tr / sd_c, Zc_te / sd_c

    return np.hstack([Zw_tr, Zc_tr]), np.hstack([Zw_te, Zc_te])
