"""Unit tests for upstreamed 3D solid-core remodeling, contour-set geometry,
CT-FM small-ROI pooling, multi-stage compartment decomposition, and outcome-free
noise-whitened SNR subspace projection (Issue #20).
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from qradiomics.atomic import (
    SubspaceProjector,
    apply_subspace_projection,
    compute_noise_scale_mad,
    fit_noise_whitening_and_snr_subspace,
    fit_transform_ctfm_blocks,
    gate_retest_channels,
    mono_pass,
    pca_scores,
    residualize_against_volume,
    select_stable_dims,
    whitener,
)
from qradiomics.deep import (
    COMPARTMENTS,
    aggregate_feature_map,
    align_pyramid_stages,
    build_compartment_masks,
    centroid_sample_stage,
    compute_adaptive_focal_spacing,
    compute_adaptive_zoom_spacing,
    extract_multistage_compartment_features,
    focal_crop_key,
    pool_compartment_stage,
    pool_pyramid_compartments,
    project_mask_centroid_to_feature_grid,
    resize_mask_to_feature_grid,
    soft_pool_stage,
)
from qradiomics.shape import (
    ContourSetMetrics,
    SolidRemodelingMetrics,
    classify_contour_role,
    classify_remodeling_regime,
    compute_longitudinal_remodeling_delta,
    compute_one_solid_shape,
    compute_solid_core_shape,
    compute_solid_remodeling_features,
    cset_nodal_to_primary_ratio,
    cset_nodal_vol_fraction,
    cset_vol_entropy,
    decompose_remodeling_regimes,
    extract_contour_set_metrics,
    is_GTV_or_tumor_roi,
    is_nodal_roi,
    is_primary_tumor_roi,
    level,
    slope,
    slope_ok,
    theil_sen,
)


# ---------------------------------------------------------------------------
# 1. qradiomics.deep.pooling tests
# ---------------------------------------------------------------------------


def test_soft_mask_pooling_preserves_subvoxel_roi_between_grid_centers():
    """Sub-voxel ROI located between coarse stride-16 sample centers must not vanish under soft_mask=True."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(42)
    fm = torch.arange(1, 1 + 4 * 2 * 2 * 2, dtype=torch.float32).reshape(1, 4, 2, 2, 2)
    # 32x32x32 high-res mask with a 2x2x2 off-center lesion at [7:9, 7:9, 7:9]
    mask = torch.zeros((32, 32, 32), dtype=torch.float32)
    mask[7:9, 7:9, 7:9] = 1.0

    resized_soft, used_fb_soft = resize_mask_to_feature_grid(mask, (2, 2, 2), soft_mask=True)
    assert not used_fb_soft
    assert float(resized_soft.sum().item()) > 0.0

    feats = aggregate_feature_map(fm, mask, soft_mask=True, prefix="deep")
    assert len(feats) == 16
    for ch in range(4):
        assert np.isfinite(feats[f"deep_mean_{ch}"])
        assert np.isfinite(feats[f"deep_std_{ch}"])


def test_center_of_mass_fallback_when_occupancy_below_eps():
    """When resized mask sum < eps, fallback must project true center-of-mass (not bounding-box midpoint)."""
    torch = pytest.importorskip("torch")
    fm = torch.zeros((1, 2, 4, 4, 4), dtype=torch.float32)
    fm[0, 0, 3, 3, 3] = 10.0
    fm[0, 1, 3, 3, 3] = -5.0

    # Create a mask concentrated near corner (15, 15, 15) of a (16, 16, 16) grid
    mask = torch.zeros((16, 16, 16), dtype=torch.float32)
    mask[14:16, 14:16, 14:16] = 1.0

    iz, iy, ix = project_mask_centroid_to_feature_grid(mask > 0, (4, 4, 4))
    assert (iz, iy, ix) == (3, 3, 3)

    # Trigger fallback via high eps
    ch_mean, ch_std = soft_pool_stage(fm, mask, eps=100.0)
    assert torch.allclose(ch_mean, torch.tensor([10.0, -5.0]))
    assert torch.allclose(ch_std, torch.zeros(2))

    c_mean, c_std = centroid_sample_stage(fm, mask)
    assert torch.allclose(c_mean, torch.tensor([10.0, -5.0]))
    assert torch.allclose(c_std, torch.zeros(2))


def test_adaptive_zoom_spacing_small_vs_large_roi():
    """Small ROIs (< 10 mm) get fine sub-millimeter spacing; large ROIs clamp to base_resample_mm."""
    small_sp = compute_adaptive_zoom_spacing(6.4)
    large_sp = compute_adaptive_zoom_spacing(120.0)
    assert small_sp[0] < 3.0 and small_sp[1] < 1.0 and small_sp[2] < 1.0
    assert large_sp == (3.0, 1.0, 1.0)

    # SimpleITK mask test
    arr = np.zeros((20, 40, 40), dtype=np.uint8)
    arr[8:12, 16:22, 16:22] = 1
    mask_sitk = sitk.GetImageFromArray(arr)
    mask_sitk.SetSpacing((1.0, 1.0, 2.0))
    sp_sitk = compute_adaptive_focal_spacing(mask_sitk)
    assert sp_sitk[0] < 3.0 and sp_sitk[1] < 1.0
    k1 = focal_crop_key(mask_sitk, resample_mm=sp_sitk)
    assert isinstance(k1, str) and len(k1) == 16


# ---------------------------------------------------------------------------
# 2. qradiomics.deep.compartment tests
# ---------------------------------------------------------------------------


def test_multistage_compartment_decomposition_and_shell_fallback():
    """Verify whole, solid_core (HU >= -500), and boundary_shell (HU < -500) pooling across pyramid stages."""
    shape = (16, 16, 16)
    ct = np.full(shape, -800.0, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    mask[4:12, 4:12, 4:12] = True
    # Inner solid core [6:10, 6:10, 6:10] has HU = +50; outer shell inside mask has HU = -650
    ct[6:10, 6:10, 6:10] = 50.0
    ct[mask & (ct < 0)] = -650.0

    masks = build_compartment_masks(ct, mask)
    assert set(COMPARTMENTS).issubset(masks.keys())
    assert masks["whole"].sum() == 8**3
    assert masks["solid_core"].sum() == 4**3
    assert masks["boundary_shell"].sum() == 8**3 - 4**3

    # Shell fallback when entire lesion is >= -500 HU
    ct_all_solid = np.full(shape, 40.0, dtype=np.float32)
    masks_fb = build_compartment_masks(ct_all_solid, mask, shell_fallback=True)
    assert masks_fb["boundary_shell"].sum() > 0

    torch = pytest.importorskip("torch")
    # Multi-stage pyramid tensors: stages (-3, -2, -1) with channels (4, 2, 2) -> total 8 channels
    s_m3 = torch.ones((1, 4, 2, 2, 2), dtype=torch.float32) * 1.0
    s_m2 = torch.ones((1, 2, 4, 4, 4), dtype=torch.float32) * 2.0
    s_m1 = torch.ones((1, 2, 8, 8, 8), dtype=torch.float32) * 3.0
    aligned = align_pyramid_stages([s_m3, s_m2, s_m1], stages=(-3, -2, -1))
    assert aligned.shape == (1, 8, 8, 8, 8)

    vec, nvox = pool_pyramid_compartments(aligned, ct, mask)
    assert vec.shape == (3 * 2 * 8,)
    assert nvox["whole"] == 512 and nvox["core"] == 64 and nvox["shell"] == 448

    res = extract_multistage_compartment_features([s_m3, s_m2, s_m1], ct, mask)
    assert res["feature_vector"].shape == (48,)
    assert res["nvox_solid_core"] == 64
    assert res["nvox_boundary_shell"] == 448

    c_mean_empty, _, nv = pool_compartment_stage(
        aligned, np.zeros(shape, dtype=bool), fallback_mask=mask
    )
    assert nv == 0 and np.all(np.isfinite(c_mean_empty))


# ---------------------------------------------------------------------------
# 3. qradiomics.atomic.denoise_subspace tests
# ---------------------------------------------------------------------------


def test_noise_whitening_and_snr_subspace_recovers_high_snr_direction():
    """Outcome-free SNR subspace projection must recover known planted signal over high-variance noise."""
    rng = np.random.default_rng(123)
    n_samples, n_pairs, d = 80, 40, 12

    # Channel 0-1: true biological signal (low noise, large biological spread)
    # Channel 2-5: high-variance correlated scanner noise
    noise_cov = np.eye(d)
    noise_cov[2:6, 2:6] = 25.0
    noise_deltas = rng.multivariate_normal(np.zeros(d), noise_cov, size=n_pairs)

    X_signal = rng.multivariate_normal(np.zeros(d), noise_cov * 0.5, size=n_samples)
    latent_bio = rng.normal(0.0, 4.0, size=(n_samples, 1))
    X_signal[:, 0:2] += latent_bio @ np.array([[1.0, 0.8]])

    # Retest pairs A and B
    pair_a = X_signal[:n_pairs]
    pair_b = pair_a + noise_deltas * 0.1
    stable_mask = select_stable_dims(pair_a, pair_b, rho_gate=0.7, min_dims=4)
    assert stable_mask.shape == (d,)
    assert bool(stable_mask[0]) and bool(stable_mask[1])

    s_mad = compute_noise_scale_mad(noise_deltas)
    W = whitener(noise_deltas / s_mad, lam=0.3)
    assert W.shape == (d, d)
    assert np.allclose(W, W.T, atol=1e-6)

    proj = fit_noise_whitening_and_snr_subspace(
        X_signal, noise_deltas, n_components=3, lam=0.3, selected_dims=stable_mask
    )
    assert isinstance(proj, SubspaceProjector)
    scores = apply_subspace_projection(X_signal, proj)
    assert scores.shape == (n_samples, 3)
    # Top SNR component must strongly correlate with planted biological latent variable
    corr_top = abs(np.corrcoef(scores[:, 0], latent_bio[:, 0])[0, 1])
    assert corr_top > 0.80


def test_residualize_against_volume_and_helper_utilities():
    """Volume residualization removes linear correlation with log_volume without train/eval leakage."""
    rng = np.random.default_rng(7)
    log_vol = rng.normal(8.0, 1.5, size=60)
    pure_feat = rng.normal(0.0, 1.0, size=60)
    confounded = 2.5 * log_vol - 3.0 + pure_feat

    res_tr, res_te = residualize_against_volume(
        confounded[:40],
        log_vol[:40],
        X_eval=confounded[40:],
        log_volume_eval=log_vol[40:],
        standardize=True,
    )
    assert abs(np.corrcoef(res_tr, log_vol[:40])[0, 1]) < 1e-10
    assert res_te.shape == (20,)

    # mono_pass, gate_retest_channels, pca_scores, fit_transform_ctfm_blocks
    rh = np.array([[0.9, 0.1], [0.85, -0.2], [0.92, 0.05], [0.88, 0.15]])
    mp = mono_pass(rh, null=0.2)
    assert list(mp) == [True, False]

    kept = gate_retest_channels(
        ["c0", "c1"], np.array([0.95, 0.5]), np.array([1.2, 1.0]), rho_gate=0.9
    )
    assert kept == ["c0"]

    P, Q = pca_scores(rng.normal(size=(30, 6)), rng.normal(size=(10, 6)), k=3)
    assert P.shape == (30, 3) and Q.shape == (10, 3)

    Ztr, Zte = fit_transform_ctfm_blocks(
        rng.normal(size=(30, 6)),
        rng.normal(size=(30, 8)),
        rng.normal(size=(10, 6)),
        rng.normal(size=(10, 8)),
        k=2,
    )
    assert Ztr.shape == (30, 4) and Zte.shape == (10, 4)


# ---------------------------------------------------------------------------
# 4. qradiomics.shape.solid_remodeling tests
# ---------------------------------------------------------------------------


def test_solid_remodeling_sphere_vs_pancake_vs_cavitation():
    """Verify 3D solid-core shape descriptors distinguish sphere, flat pancake, and central cavitation."""
    grid = np.mgrid[0:32, 0:32, 0:32].astype(np.float32)
    dist_sq = (grid[0] - 16.0) ** 2 + (grid[1] - 16.0) ** 2 + (grid[2] - 16.0) ** 2

    # 1. Dense solid sphere (radius 8) with thin sub-solid outer halo (radius 8..9)
    mask_sphere = dist_sq <= 9.0**2
    hu_sphere = np.full((32, 32, 32), -800.0, dtype=np.float32)
    hu_sphere[mask_sphere] = -600.0
    hu_sphere[dist_sq <= 8.0**2] = 20.0
    hu_sphere[dist_sq <= 6.0**2] = 80.0

    m_sphere = compute_solid_core_shape(hu_sphere, mask_sphere, spacing_mm=(1.0, 1.0, 1.0))
    assert isinstance(m_sphere, SolidRemodelingMetrics)
    assert m_sphere.Sphericity > 0.65
    assert m_sphere.Flatness > 0.85
    assert m_sphere.SolidFraction > 0.65
    assert m_sphere.Core_Hu_Mean > m_sphere.Band_Hu_Mean

    # 2. Anisotropic pancake (collapsed along Z axis)
    pancake_mask = (
        ((grid[0] - 16.0) / 2.0) ** 2
        + ((grid[1] - 16.0) / 10.0) ** 2
        + ((grid[2] - 16.0) / 10.0) ** 2
    ) <= 1.0
    hu_pancake = np.full((32, 32, 32), -800.0, dtype=np.float32)
    hu_pancake[pancake_mask] = 40.0
    m_pancake = compute_solid_core_shape(hu_pancake, pancake_mask, spacing_mm=(1.0, 1.0, 1.0))
    assert m_pancake is not None
    assert m_pancake.Flatness < 0.40
    assert m_pancake.Pancake_Index > m_sphere.Pancake_Index

    # 3. Central cavitation + fragmentation follow-up scan
    hu_cav = hu_sphere.copy()
    # Hollow out interior [dist <= 6.5] to -750 HU (air/necrotic cavity)
    hu_cav[dist_sq <= 6.5**2] = -750.0
    m_cav = compute_solid_core_shape(hu_cav, mask_sphere, spacing_mm=(1.0, 1.0, 1.0))
    assert m_cav is not None
    assert m_cav.CoreToBand_LogRatio < m_sphere.CoreToBand_LogRatio

    delta = compute_longitudinal_remodeling_delta(m_sphere, m_cav)
    assert delta["Central_Cavitation_Neg_A0"] > 0.5
    assert delta["regime_central_cavitation_or_frag"] is True
    assert delta["primary_regime"] == "central_cavitation_or_frag"

    # Worker wrapper + dict interface
    f_dict = compute_solid_remodeling_features(hu_sphere, mask_sphere)
    assert "Pancake_Index_ap" in f_dict and "Ellipsoid_Packing" in f_dict
    one_res = compute_one_solid_shape(("scan_01", "GTV", hu_sphere, mask_sphere))
    assert one_res is not None and one_res["image_file"] == "scan_01"


def test_longitudinal_trajectory_slope_and_level():
    """Verify Theil-Sen slope, early-vs-late level difference, and regime decomposition."""
    days = np.array([1.0, 2.0, 5.0, 9.0, 12.0, 14.0])
    # Feature 0 decreases at -0.5/day; Feature 1 increases at +0.25/day
    Y = np.column_stack([-0.5 * days + 10.0, 0.25 * days + 1.0])

    assert slope_ok(days, span=(1.0, 15.0), min_n=4, min_span=7.0)
    ts = slope(days, Y, span=(1.0, 15.0), min_n=4, min_span=7.0)
    assert ts is not None
    assert np.allclose(ts, np.array([-0.5, 0.25]), atol=1e-6)

    lv = level(days, Y, early=(1.0, 3.0), late=(8.0, 15.0), min_part=2)
    assert lv is not None
    assert lv[0] < -4.0 and lv[1] > 2.0

    reg = decompose_remodeling_regimes(np.array([0.4, -0.6]))
    assert np.allclose(reg["Concentric_Clearance_Pos_A0"], [0.4, 0.0])
    assert np.allclose(reg["Central_Cavitation_Neg_A0"], [0.0, 0.6])

    cls_conc = classify_remodeling_regime({"log_V_A0": -0.5, "CoreToBand_LogRatio_A0": 0.35})
    assert cls_conc["regime_concentric_clearance"] is True
    assert cls_conc["major_solid_shrinkage_responder"] is True


# ---------------------------------------------------------------------------
# 5. qradiomics.shape.contour_set tests
# ---------------------------------------------------------------------------


def test_contour_set_classification_and_burden_metrics():
    """Verify primary-vs-nodal ROI classification, nodal volume fraction, Shannon entropy, and worst-lesion stats."""
    assert is_GTV_or_tumor_roi("GTV_Primary_RUL") is True
    assert is_GTV_or_tumor_roi("GTVn_4R") is True
    assert is_GTV_or_tumor_roi("Lung-GTV") is False
    assert is_GTV_or_tumor_roi("PTV + 2cm", include_ptv_ctv=True) is False
    assert is_GTV_or_tumor_roi("zPTV_eval", include_ptv_ctv=True) is False

    assert is_nodal_roi("GTVn_Station7") is True
    assert is_nodal_roi("CTV_Mediastinal_LN") is True
    assert is_nodal_roi("GTV_RUL_Mass") is False

    assert is_primary_tumor_roi("GTV_Primary_LUL") is True
    assert is_primary_tumor_roi("GTV") is True
    assert is_primary_tumor_roi("GTVn_4R") is False
    assert classify_contour_role("GTVp+n") == "composite"

    # Equal 2-target split has Shannon entropy log(2)
    ent = cset_vol_entropy([500.0, 500.0])
    assert np.isclose(ent, np.log(2.0))
    assert np.isclose(cset_nodal_vol_fraction(250.0, 750.0), 0.25)
    assert np.isclose(cset_nodal_to_primary_ratio(250.0, 500.0, log_scale=False), 0.5)

    # Multi-target 3D masks extraction
    ct = np.full((24, 24, 24), -700.0, dtype=np.float32)
    m_pri = np.zeros((24, 24, 24), dtype=bool)
    m_pri[4:12, 4:12, 4:12] = True  # 512 vox
    ct[m_pri] = 35.0

    m_nod = np.zeros((24, 24, 24), dtype=bool)
    m_nod[14:18, 14:20, 14:20] = True  # 144 vox
    ct[m_nod] = 15.0

    m_ptv = np.zeros((24, 24, 24), dtype=bool)
    m_ptv[2:14, 2:14, 2:14] = True  # 1728 vox

    m_oar = np.ones((24, 24, 24), dtype=bool)

    cset = extract_contour_set_metrics(
        {
            "GTV_Primary": m_pri,
            "GTVn_4R": m_nod,
            "PTV_Primary": m_ptv,
            "Lung-GTV": m_oar,  # must be ignored as OAR subtraction
        },
        ct_image=ct,
        spacing_mm=(1.0, 1.0, 1.0),
    )
    assert isinstance(cset, ContourSetMetrics)
    assert cset.cset_n_total == 3
    assert cset.cset_n_gtv == 2
    assert cset.cset_n_primary == 1
    assert cset.cset_n_nodal == 1
    assert np.isclose(cset.primary_vol_mm3, 512.0)
    assert np.isclose(cset.nodal_vol_mm3, 144.0)
    assert np.isclose(cset.cset_nodal_to_primary_ratio, 144.0 / 512.0)
    assert np.isclose(cset.cset_nodal_vol_fraction, 144.0 / (512.0 + 144.0))
    assert cset.cset_vol_entropy > 0.0
    assert 0.0 < cset.cset_sphericity_min <= 1.0
    assert cset.cset_hu_std_across_targets > 0.0


# ---------------------------------------------------------------------------
# 6. Focused scenario tests (parity with JeffLungRadiomics unit suite)
# ---------------------------------------------------------------------------


def test_whitener_is_symmetric_positive_definite():
    rng = np.random.default_rng(42)
    W = whitener(rng.normal(size=(50, 4)), lam=0.5)
    assert W.shape == (4, 4)
    assert np.isfinite(W).all()
    assert np.allclose(W, W.T, atol=1e-5)
    assert (np.linalg.eigvalsh(W) > 0).all()


def test_mono_pass_accepts_consistent_and_rejects_null_trends():
    assert mono_pass(np.full((15, 3), 0.85), null=0.1).all()
    assert not mono_pass(np.zeros((15, 3)), null=0.1).any()


def test_gate_retest_channels_requires_rho_and_nonzero_mad():
    cols = ["c1", "c2", "c3"]
    cidx = {"c1": 0, "c2": 1, "c3": 2}
    gated = gate_retest_channels(
        cols,
        np.array([0.95, 0.85, 0.92]),
        np.array([1.2, 0.5, 0.0]),
        cidx,
        rho_gate=0.90,
    )
    assert gated == ["c1"]


def test_pca_scores_shapes_and_finiteness():
    rng = np.random.default_rng(42)
    P, Q = pca_scores(rng.normal(size=(30, 8)), rng.normal(size=(10, 8)), k=3, centered=True)
    assert P.shape == (30, 3) and Q.shape == (10, 3)
    assert np.isfinite(P).all() and np.isfinite(Q).all()


def test_fit_transform_ctfm_blocks_multi_block_fit_transform():
    rng = np.random.default_rng(42)
    Ztr, Zte = fit_transform_ctfm_blocks(
        rng.normal(size=(25, 6)),
        rng.normal(size=(25, 8)),
        rng.normal(size=(10, 6)),
        rng.normal(size=(10, 8)),
        k=3,
    )
    assert Ztr.shape == (25, 6) and Zte.shape == (10, 6)
    assert np.isfinite(Ztr).all() and np.isfinite(Zte).all()


def test_pool_pyramid_compartments_core_shell_split():
    torch = pytest.importorskip("torch")
    torch.manual_seed(42)
    fm = torch.randn(1, 16, 8, 8, 8)
    hu = np.zeros((8, 8, 8), dtype=np.float32)
    hu[:4] = -300.0
    hu[4:] = -700.0
    vec, nvox = pool_pyramid_compartments(fm, hu, np.ones((8, 8, 8), dtype=bool))
    assert len(vec) == len(COMPARTMENTS) * 2 * 16
    assert nvox["whole"] == 512 and nvox["core"] == 256 and nvox["shell"] == 256
    assert np.isfinite(vec).all()


def test_compute_one_solid_shape_synthetic_volume():
    img_arr = np.full((16, 16, 16), -300.0, dtype=np.float32)
    img_arr[0:4] = -700.0
    msk_arr = np.zeros((16, 16, 16), dtype=np.uint8)
    msk_arr[2:14, 2:14, 2:14] = 1
    img = sitk.GetImageFromArray(img_arr)
    msk = sitk.GetImageFromArray(msk_arr)
    for im in (img, msk):
        im.SetSpacing([1.0, 1.0, 1.0])
    res = compute_one_solid_shape(("synth_cbct", "GTV", img, msk))
    assert res is not None
    assert "la" in res and "sph" in res and "sf_a0" in res and "c2b_log_a0" in res
    assert res["sf_a0"] > 0
    assert 0.0 < res["sph"] <= 1.0


def test_decompose_remodeling_regimes_level_and_slope_inputs():
    days = np.array([1.0, 2.0, 9.0, 14.0])
    Y = np.column_stack([0.3 * days, -0.2 * days])
    lv = level(days, Y, early=(1.0, 3.0), late=(8.0, 15.0), min_part=2)
    sl = slope(days, Y, span=(1.0, 15.0), min_n=3, min_span=7.0)
    assert lv is not None and sl is not None
    for delta in (lv, sl):
        reg = decompose_remodeling_regimes(delta)
        assert np.allclose(reg["Concentric_Clearance_Pos_A0"], np.maximum(delta, 0.0))
        assert np.allclose(reg["Central_Cavitation_Neg_A0"], np.maximum(-delta, 0.0))


def test_remodeling_constants_sizes():
    from qradiomics.shape import solid_remodeling as sr
    from qradiomics.shape.contour_set import CSET_GEO_COLS

    assert len(CSET_GEO_COLS) == 10
    assert len(sr.PRIMARY_SHAPE_ARCH_COLS) == 7
    assert len(sr.BASE_SOLID_ARCH_COLS) == 3
    assert len(sr.DYNAMIC_REMODELING_CLASSES) == 4
    assert "regime_concentric_clearance" in sr.DYNAMIC_REMODELING_CLASSES
    assert "regime_central_cavitation_or_frag" in sr.DYNAMIC_REMODELING_CLASSES
