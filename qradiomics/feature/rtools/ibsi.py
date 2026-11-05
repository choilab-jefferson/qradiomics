"""Opt-in IBSI-1 profile of the rtools engine.

Selected with ``extract_features(..., ibsi=True)``; the default (C++-exact)
outputs are untouched.  Implements the IBSI-1 reference-manual definitions
for the families the rtools engine covers:

* ``i`` -> intensity statistics (``stat_*``, 18 features);
* ``s``/``g`` -> voxel-based morphology (``morph_vol_approx``,
  ``morph_pca_*``); mesh-based morphology is not implemented;
* ``c`` -> GLCM (``cm_*``, 25 features), ``r`` -> GLRLM (``rlm_*``, 16
  features), each with the six IBSI aggregation methods
  ``2D_avg``, ``2D_comb`` (slice-wise merged), ``2_5D_avg`` (direction-wise
  merged), ``2_5D_comb`` (fully merged 2D), ``3D_avg``, ``3D_comb`` (merged).

Discretisation: fixed bin size (``bin_width``, lower bound ``bin_min`` or the
ROI minimum) or fixed bin number (``bin_number``, default 64).  No
interpolation and no re-segmentation are applied.  The ``2`` (largest-slice)
family has no IBSI counterpart.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np

AGGREGATIONS = ("2D_avg", "2D_comb", "2_5D_avg", "2_5D_comb", "3D_avg", "3D_comb")


# ---------------------------------------------------------------------------
# discretisation
# ---------------------------------------------------------------------------
def discretise(values: np.ndarray, *, bin_width: Optional[float] = None, bin_min: Optional[float] = None,
               bin_number: Optional[int] = None):
    """IBSI fixed bin size / fixed bin number discretisation (levels >= 1).

    Returns ``(levels, n_levels)``."""
    v = np.asarray(values, dtype=np.float64)
    if bin_width is not None:
        lo = float(v.min()) if bin_min is None else float(bin_min)
        g = np.floor((v - lo) / bin_width).astype(np.int64) + 1
        return g, int(g.max())
    n = int(bin_number or 64)
    lo, hi = float(v.min()), float(v.max())
    if hi == lo:
        return np.ones(v.shape, dtype=np.int64), 1
    g = np.floor(n * (v - lo) / (hi - lo)).astype(np.int64) + 1
    g[v == hi] = n
    return g, n


# ---------------------------------------------------------------------------
# statistics / morphology
# ---------------------------------------------------------------------------
def statistics(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    d = x - mu
    m2 = np.mean(d ** 2)
    p10, p25, p75, p90 = np.percentile(x, [10, 25, 75, 90])
    sel = x[(x >= p10) & (x <= p90)]
    med = float(np.median(x))
    return {
        "stat_mean": mu,
        "stat_var": m2,
        "stat_skew": np.mean(d ** 3) / m2 ** 1.5 if m2 > 0 else 0.0,
        "stat_kurt": np.mean(d ** 4) / m2 ** 2 - 3.0 if m2 > 0 else 0.0,
        "stat_median": med,
        "stat_min": x.min(),
        "stat_p10": p10,
        "stat_p90": p90,
        "stat_max": x.max(),
        "stat_iqr": p75 - p25,
        "stat_range": x.max() - x.min(),
        "stat_mad": np.mean(np.abs(d)),
        "stat_rmad": np.mean(np.abs(sel - sel.mean())),
        "stat_medad": np.mean(np.abs(x - med)),
        "stat_cov": math.sqrt(m2) / mu if mu != 0 else float("nan"),
        "stat_qcod": (p75 - p25) / (p75 + p25) if (p75 + p25) != 0 else float("nan"),
        "stat_energy": np.sum(x ** 2),
        "stat_rms": math.sqrt(np.mean(x ** 2)),
    }


def morphology_voxel(mask: np.ndarray, spacing_xyz) -> dict:
    sp = np.asarray(spacing_xyz, dtype=np.float64)
    pts = np.argwhere(mask)[:, ::-1] * sp
    n = len(pts)
    out = {"morph_vol_approx": n * float(np.prod(sp))}
    if n < 2:
        return out
    lam = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
    out.update({
        "morph_pca_maj_axis": 4 * math.sqrt(lam[0]),
        "morph_pca_min_axis": 4 * math.sqrt(lam[1]),
        "morph_pca_least_axis": 4 * math.sqrt(max(lam[2], 0.0)),
        "morph_pca_elongation": math.sqrt(lam[1] / lam[0]),
        "morph_pca_flatness": math.sqrt(max(lam[2], 0.0) / lam[0]),
    })
    return out


# ---------------------------------------------------------------------------
# directions
# ---------------------------------------------------------------------------
def _directions_3d():
    """13 unique 3D directions, numpy (z, y, x) order."""
    out = []
    for d in itertools.product((-1, 0, 1), repeat=3):
        if d > (0, 0, 0):
            out.append(d)
    return out


_DIRS2D = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, -1)]


def _shift(shape, off):
    c, n = [], []
    for size, o in zip(shape, off):
        if o >= 0:
            c.append(slice(0, size - o))
            n.append(slice(o, size))
        else:
            c.append(slice(-o, size))
            n.append(slice(0, size + o))
    return tuple(c), tuple(n)


# ---------------------------------------------------------------------------
# GLCM
# ---------------------------------------------------------------------------
def _glcm_stack(g: np.ndarray, ng: int, dirs):
    """Symmetric co-occurrence counts per (slice, direction):
    ``(Z, len(dirs), ng, ng)``."""
    Z = g.shape[0]
    out = np.zeros((Z, len(dirs), ng, ng), dtype=np.float64)
    zz = np.broadcast_to(np.arange(Z)[:, None, None], g.shape)
    for k, off in enumerate(dirs):
        c, n = _shift(g.shape, off)
        a, b = g[c], g[n]
        ok = (a > 0) & (b > 0)
        z = zz[c][ok]
        a, b = a[ok] - 1, b[ok] - 1
        np.add.at(out, (z, k, a, b), 1)
        np.add.at(out, (z, k, b, a), 1)
    return out


def _log2(p):
    return np.log2(np.where(p > 0, p, 1.0))


def cm_features(m: np.ndarray) -> dict:
    ng = m.shape[0]
    p = m / m.sum()
    i, j = np.meshgrid(np.arange(1, ng + 1), np.arange(1, ng + 1), indexing="ij")
    pi = p.sum(axis=1)
    pj = p.sum(axis=0)
    lv = np.arange(1, ng + 1)
    mu = float(np.sum(i * p))
    k = np.abs(i - j)
    pd = np.bincount(k.ravel(), weights=p.ravel(), minlength=ng)
    s = i + j
    ps = np.bincount(s.ravel(), weights=p.ravel(), minlength=2 * ng + 1)
    kk = np.arange(ng)
    ss = np.arange(2 * ng + 1)
    d_avg = float(np.sum(kk * pd))
    s_avg = float(np.sum(ss * ps))
    mui = float(np.sum(lv * pi))
    muj = float(np.sum(lv * pj))
    sdi = math.sqrt(float(np.sum((lv - mui) ** 2 * pi)))
    sdj = math.sqrt(float(np.sum((lv - muj) ** 2 * pj)))
    hxy = float(-np.sum(p * _log2(p)))
    hx = float(-np.sum(pi * _log2(pi)))
    hy = float(-np.sum(pj * _log2(pj)))
    pij = np.outer(pi, pj)
    hxy1 = float(-np.sum(p * _log2(pij)))
    hxy2 = float(-np.sum(pij * _log2(pij)))
    cl = i + j - mui - muj
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_var = float(np.sum(pd[1:] / kk[1:] ** 2))
    return {
        "joint_max": float(p.max()),
        "joint_avg": mu,
        "joint_var": float(np.sum((i - mu) ** 2 * p)),
        "joint_entr": hxy,
        "diff_avg": d_avg,
        "diff_var": float(np.sum((kk - d_avg) ** 2 * pd)),
        "diff_entr": float(-np.sum(pd * _log2(pd))),
        "sum_avg": s_avg,
        "sum_var": float(np.sum((ss - s_avg) ** 2 * ps)),
        "sum_entr": float(-np.sum(ps * _log2(ps))),
        "energy": float(np.sum(p * p)),
        "contrast": float(np.sum((i - j) ** 2 * p)),
        "dissimilarity": float(np.sum(k * p)),
        "inv_diff": float(np.sum(p / (1 + k))),
        "inv_diff_norm": float(np.sum(p / (1 + k / ng))),
        "inv_diff_mom": float(np.sum(p / (1 + k ** 2))),
        "inv_diff_mom_norm": float(np.sum(p / (1 + (k / ng) ** 2))),
        "inv_var": inv_var,
        "corr": (float(np.sum(i * j * p)) - mui * muj) / (sdi * sdj) if sdi * sdj > 0 else float("nan"),
        "auto_corr": float(np.sum(i * j * p)),
        "clust_tend": float(np.sum(cl ** 2 * p)),
        "clust_shade": float(np.sum(cl ** 3 * p)),
        "clust_prom": float(np.sum(cl ** 4 * p)),
        "info_corr1": (hxy - hxy1) / max(hx, hy) if max(hx, hy) > 0 else float("nan"),
        "info_corr2": math.sqrt(max(0.0, 1 - math.exp(-2 * (hxy2 - hxy)))),
    }


# ---------------------------------------------------------------------------
# GLRLM
# ---------------------------------------------------------------------------
def _glrlm_stack(g: np.ndarray, ng: int, dirs):
    """Run-length counts per (slice-of-run-start, direction):
    ``(Z, len(dirs), ng, max_len)``."""
    shape = np.array(g.shape)
    Z = g.shape[0]
    T = int(shape.max())
    out = np.zeros((Z, len(dirs), ng, T), dtype=np.float64)
    allidx = np.indices(g.shape).reshape(3, -1).T
    for k, off in enumerate(dirs):
        o = np.array(off)
        prev = allidx - o
        starts = allidx[np.any((prev < 0) | (prev >= shape), axis=1)]
        pos = starts[:, None, :] + np.arange(T)[None, :, None] * o
        inb = np.all((pos >= 0) & (pos < shape), axis=2)
        pos = np.where(inb[..., None], pos, 0)
        gl = np.where(inb, g[pos[..., 0], pos[..., 1], pos[..., 2]], 0)
        prevg = np.zeros_like(gl)
        prevg[:, 1:] = gl[:, :-1]
        start = (gl > 0) & (gl != prevg)
        run_id = np.cumsum(start.ravel()) - 1
        member = (gl > 0).ravel()
        lengths = np.bincount(run_id[member], minlength=int(start.sum()))
        levels = gl.ravel()[start.ravel()] - 1
        zs = pos[..., 0].ravel()[start.ravel()]
        np.add.at(out, (zs, k, levels, lengths - 1), 1)
    return out


def rlm_features(r: np.ndarray, n_voxels: float) -> dict:
    ng, nr = r.shape
    ns = r.sum()
    i = np.arange(1, ng + 1)[:, None].astype(np.float64)
    j = np.arange(1, nr + 1)[None, :].astype(np.float64)
    ri = r.sum(axis=1)
    rj = r.sum(axis=0)
    p = r / ns
    mu_i = float(np.sum(i * p))
    mu_j = float(np.sum(j * p))
    return {
        "sre": float(np.sum(r / j ** 2) / ns),
        "lre": float(np.sum(r * j ** 2) / ns),
        "lgre": float(np.sum(r / i ** 2) / ns),
        "hgre": float(np.sum(r * i ** 2) / ns),
        "srlge": float(np.sum(r / (i ** 2 * j ** 2)) / ns),
        "srhge": float(np.sum(r * i ** 2 / j ** 2) / ns),
        "lrlge": float(np.sum(r * j ** 2 / i ** 2) / ns),
        "lrhge": float(np.sum(r * i ** 2 * j ** 2) / ns),
        "glnu": float(np.sum(ri ** 2) / ns),
        "glnu_norm": float(np.sum(ri ** 2) / ns ** 2),
        "rlnu": float(np.sum(rj ** 2) / ns),
        "rlnu_norm": float(np.sum(rj ** 2) / ns ** 2),
        "r_perc": float(ns / n_voxels),
        "gl_var": float(np.sum((i - mu_i) ** 2 * p)),
        "rl_var": float(np.sum((j - mu_j) ** 2 * p)),
        "rl_entr": float(-np.sum(p * _log2(p))),
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _mean_dicts(ds):
    ds = [d for d in ds if d]
    return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]} if ds else {}


def _aggregate(prefix, stack2d, stack3d, fn, nvox2d, nvox3d):
    """Apply ``fn(matrix, n_voxels)`` with the six IBSI aggregations."""
    out = {}
    Z, ND = stack2d.shape[:2]
    res = {
        "2D_avg": _mean_dicts([fn(stack2d[z, d], nvox2d[z]) for z in range(Z) for d in range(ND)
                               if stack2d[z, d].sum() > 0]),
        "2D_comb": _mean_dicts([fn(stack2d[z].sum(axis=0), nvox2d[z] * ND) for z in range(Z)
                                if stack2d[z].sum() > 0]),
        "2_5D_avg": _mean_dicts([fn(stack2d[:, d].sum(axis=0), nvox3d) for d in range(ND)
                                 if stack2d[:, d].sum() > 0]),
        "2_5D_comb": fn(stack2d.sum(axis=(0, 1)), nvox3d * ND),
        "3D_avg": _mean_dicts([fn(stack3d[:, d].sum(axis=0), nvox3d) for d in range(stack3d.shape[1])
                               if stack3d[:, d].sum() > 0]),
        "3D_comb": fn(stack3d.sum(axis=(0, 1)), nvox3d * stack3d.shape[1]),
    }
    for agg, feats in res.items():
        for k, v in feats.items():
            out[f"{prefix}_{k}_{agg}"] = v
    return out


def extract_ibsi_arrays(image_zyx: np.ndarray, mask_zyx: np.ndarray, spacing_xyz, *,
                        bin_width: Optional[float] = None, bin_min: Optional[float] = None,
                        bin_number: Optional[int] = None, families: str = "isgcr") -> dict:
    """IBSI-1 features from numpy arrays (``(z, y, x)`` order)."""
    img = np.asarray(image_zyx, dtype=np.float64)
    mask = np.asarray(mask_zyx).astype(bool)
    vals = img[mask]
    out: dict = {}
    if "i" in families:
        out.update(statistics(vals))
    if "s" in families or "g" in families:
        out.update(morphology_voxel(mask, spacing_xyz))
    if "c" in families or "r" in families:
        lv, ng = discretise(vals, bin_width=bin_width, bin_min=bin_min, bin_number=bin_number)
        g = np.zeros(img.shape, dtype=np.int64)
        g[mask] = lv
        nvox2d = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(np.float64)
        nvox3d = float(mask.sum())
        if "c" in families:
            s2 = _glcm_stack(g, ng, _DIRS2D)
            s3 = _glcm_stack(g, ng, _directions_3d())
            out.update(_aggregate("cm", s2, s3, lambda m, _n: cm_features(m), nvox2d, nvox3d))
        if "r" in families:
            s2 = _glrlm_stack(g, ng, _DIRS2D)
            s3 = _glrlm_stack(g, ng, _directions_3d())
            out.update(_aggregate("rlm", s2, s3, rlm_features, nvox2d, nvox3d))
    return {k: float(v) for k, v in out.items()}


def extract_ibsi(image_path: Union[str, Path], mask_path: Union[str, Path], *, label: int = 1, **kw) -> dict:
    """IBSI-1 features from image/mask files (mask voxels ``== label``)."""
    import SimpleITK as sitk

    im = sitk.ReadImage(str(image_path), sitk.sitkFloat64)
    lab = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    return extract_ibsi_arrays(sitk.GetArrayFromImage(im), lab == label, im.GetSpacing(), **kw)


def round_sig(x: float, sig: int = 3) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


def compare_reference(values: dict, reference: dict):
    """Rows ``(tag, value, ref, tol, pass)``; a feature passes when
    ``|value - ref| <= tol`` or both agree at 3 significant figures."""
    rows = []
    for tag, (ref, tol) in reference.items():
        v = values.get(tag)
        if v is None:
            rows.append((tag, None, ref, tol, False))
            continue
        ok = abs(v - ref) <= tol or round_sig(v) == round_sig(ref)
        rows.append((tag, v, ref, tol, bool(ok)))
    return rows
