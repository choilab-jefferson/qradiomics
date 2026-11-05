"""``BinaryImageToStatisticsLabelMapFilter`` port (ITK 5.1.2
``ShapeLabelMapFilter`` + ``StatisticsLabelMapFilter``) -- the ``s`` (3D) and
``2`` (largest-area 2D slice) families of ``FeatureExtraction``."""
from __future__ import annotations

import itertools
import math

import numpy as np

from ._itk import (
    Axis,
    Geom,
    fmt,
    fmt_itk_matrix,
    fmt_vec,
    hypersphere_perimeter,
    hypersphere_radius_from_volume,
    not_almost_zero,
    runs,
)

_BIG = 2**63 - 1


def _sqrt(x: float) -> float:
    """``std::sqrt`` (NaN for negative input instead of raising)."""
    return math.sqrt(x) if x >= 0 else -math.nan


def _principal_axes(cm: np.ndarray):
    """vnl_symmetric_eigensystem + the determinant fix-up of the filters."""
    w, v = np.linalg.eigh(cm)
    axes = v.T.copy()
    det = np.real(np.prod(np.linalg.eigvals(axes)))
    axes[-1, :] *= det
    return w, axes


def _perimeter(rows, x0, length, bbox_min, bbox_size, spacing) -> float:
    """``ShapeLabelMapFilter::ComputePerimeter`` (intercept counting)."""
    D = len(spacing)
    lines: dict = {}
    for r, a, n in zip(map(tuple, rows), x0.tolist(), length.tolist()):
        lines.setdefault(r, []).append((a, n))
    intercepts: dict = {}

    def add(key, val):
        intercepts[key] = intercepts.get(key, 0) + val

    # line-image region: ITK dims 1..D-1 of the bounding box; rows are numpy
    # order (z, y) so reverse to ITK order for offsets.
    lo = [bbox_min[i] for i in range(1, D)]
    hi = [bbox_min[i] + bbox_size[i] for i in range(1, D)]
    nbs = [o for o in itertools.product((-1, 0, 1), repeat=D - 1) if any(o)]
    for pos in itertools.product(*[range(a, b) for a, b in zip(lo, hi)]):
        key_np = tuple(reversed(pos))
        ls = lines.get(key_np, [])
        add((1,) + (0,) * (D - 1), 2 * len(ls))
        for o in nbs:
            npos = tuple(p + d for p, d in zip(pos, o))
            ns = lines.get(tuple(reversed(npos)), [])
            no = (0,) + tuple(abs(d) for d in o)
            dno = (1,) + no[1:]
            if not ns:
                for _, n in ls:
                    add(no, n)
                    add(dno, 2 * n)
                continue
            li = ni = 0
            n_min = -_BIG + 1
            n_max = ns[0][0] - 1
            while li < len(ls):
                l_min = ls[li][0]
                l_max = l_min + ls[li][1] - 1
                add(no, max(0, min(l_max, n_max) - max(l_min, n_min) + 1))
                add(dno, max(0, min(l_max, n_max + 1) - max(l_min, n_min + 1) + 1))
                add(dno, max(0, min(l_max, n_max - 1) - max(l_min, n_min - 1) + 1))
                if n_max <= l_max:
                    n_min = ns[ni][0] + ns[ni][1]
                    ni += 1
                    n_max = ns[ni][0] - 1 if ni < len(ns) else _BIG - 1
                else:
                    li += 1

    def ic(*k):
        return intercepts.get(tuple(k), 0)

    s = [float(x) for x in spacing]
    if D == 2:
        dx, dy = s
        norm = math.sqrt(dx * dx + dy * dy)
        p = 0.0
        p += dy * ic(1, 0) / 2.0
        p += dx * ic(0, 1) / 2.0
        p += dx * dy / norm * ic(1, 1) / 2.0
        return p * (math.pi / 4.0)
    dx, dy, dz = s
    dxy = math.sqrt(dx * dx + dy * dy)
    dxz = math.sqrt(dx * dx + dz * dz)
    dyz = math.sqrt(dy * dy + dz * dz)
    dxyz = math.sqrt(dx * dx + dy * dy + dz * dz)
    vol = dx * dy * dz
    c1 = 0.04577789120476 * 2
    c4 = 0.03698062787608 * 2
    c7 = 0.03519563978232 * 2
    p = 0.0
    p += vol / dx * ic(1, 0, 0) / 2.0 * c1
    p += vol / dy * ic(0, 1, 0) / 2.0 * c1
    p += vol / dz * ic(0, 0, 1) / 2.0 * c1
    p += vol / dxy * ic(1, 1, 0) / 2.0 * c4
    p += vol / dxz * ic(1, 0, 1) / 2.0 * c4
    p += vol / dyz * ic(0, 1, 1) / 2.0 * c4
    p += vol / dxyz * ic(1, 1, 1) / 2.0 * c7
    return p * 4


def _feret(obj: np.ndarray, spacing) -> float:
    """``ShapeLabelMapFilter::ComputeFeretDiameter``: max distance between
    object pixels that touch a non-object pixel (image outside counts)."""
    D = obj.ndim
    pad = np.pad(obj, 1, constant_values=False)
    border = np.zeros_like(obj)
    for off in itertools.product((-1, 0, 1), repeat=D):
        if not any(off):
            continue
        sl = tuple(slice(1 + o, pad.shape[k] - 1 + o) for k, o in enumerate(off))
        border |= ~pad[sl]
    border &= obj
    pts = np.argwhere(border)[:, ::-1].astype(np.int64)  # ITK (x, y, z) order
    if len(pts) < 2:
        return 0.0
    sp = np.asarray(spacing, dtype=np.float64)
    cand = pts
    if len(pts) > 3000:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(pts * sp)
            cand = pts[np.unique(hull.vertices)]
        except Exception:  # degenerate (flat) point cloud
            cand = pts
    best = 0.0
    step = max(1, 4_000_000 // max(len(cand), 1))
    for a in range(0, len(cand), step):
        blk = cand[a:a + step]
        acc = np.zeros((len(blk), len(cand)))
        for i in range(D):
            d = (blk[:, None, i] - cand[None, :, i]) * sp[i]
            acc = acc + d * d
        best = max(best, float(acc.max()))
    return math.sqrt(best)


def shape_statistics(obj: np.ndarray, feat: np.ndarray, geom: Geom, nbins: int):
    """Return the ordered ``(name, text)`` pairs that ``FeatureExtraction``
    writes for one label object (``CalcShapeAndIntensityFeatures``)."""
    D = obj.ndim
    sp = np.asarray(geom.spacing, dtype=np.float64)
    rows, x0, length = runs(obj)
    # ITK index of each line start, (x, y, z) order
    lidx = np.concatenate([x0[:, None], rows[:, ::-1]], axis=1).astype(np.int64)
    n_lines = len(length)
    n = int(length.sum())
    size_img = np.array(obj.shape[::-1], dtype=np.int64)

    size_per_pixel = 1.0
    for s in sp:
        size_per_pixel *= s
    sppd = [size_per_pixel / s for s in sp]

    # --- centroid / bbox (index space, exact) ---------------------------
    centroid = np.zeros(D)
    for i in range(1, D):
        centroid[i] = float(np.sum(length * lidx[:, i]))
    centroid[0] = float(np.sum(lidx[:, 0] * length + (length * (length - 1)) / 2.0))
    mins = lidx.min(axis=0)
    maxs = lidx.max(axis=0)
    maxs[0] = max(maxs[0], int((lidx[:, 0] + length - 1).max()))
    bbox_size = maxs - mins + 1

    # --- border ------------------------------------------------------------
    bmax = size_img - 1
    on_b = np.zeros(n_lines, dtype=bool)
    for i in range(1, D):
        on_b |= (lidx[:, i] == 0) | (lidx[:, i] == bmax[i])
    end0 = lidx[:, 0] + length - 1
    first0 = lidx[:, 0] == 0
    last0 = end0 == bmax[0]
    n_border = int(length[on_b].sum())
    nb = ~on_b
    n_border += int(np.sum(nb & first0))
    n_border += int(np.sum(nb & (~first0 | (length > 1)) & last0))
    perim_border = 0.0
    for k in range(n_lines):  # sequential, mirrors the per-line accumulation
        if first0[k]:
            perim_border += sppd[0]
        if last0[k]:
            perim_border += sppd[0]
        for i in range(1, D):
            if lidx[k, i] == 0:
                perim_border += sppd[i] * int(length[k])
            if lidx[k, i] == bmax[i]:
                perim_border += sppd[i] * int(length[k])

    # --- central moments (physical) -----------------------------------------
    cm = np.zeros((D, D))
    short = length <= 2
    if short.any():
        pix = []
        for k in np.nonzero(short)[0]:
            for t in range(int(length[k])):
                p = lidx[k].copy()
                p[0] += t
                pix.append(p)
        pp = geom.points(np.array(pix))
        cm += pp.T @ pp
    if (~short).any():
        L = length[~short].astype(np.float64)
        pp = geom.points(lidx[~short])
        scale = sp[0] * geom.direction[:, 0]
        c1 = (L - 1.0) / 2.0
        c2 = (2.0 * L - 1.0) / 3.0
        for i in range(D):
            for j in range(D):
                cm[i, j] += float(np.sum(L * (pp[:, i] * pp[:, j] + c1 * (
                    pp[:, i] * scale[j] + scale[i] * pp[:, j] + c2 * scale[i] * scale[j]))))
    centroid /= n
    cm /= n
    pcent = geom.points(centroid[None, :])[0]
    cm -= np.outer(pcent, pcent)
    pm, paxes = _principal_axes(cm)

    elong = flat = 0.0
    if not_almost_zero(pm[0]):
        r = pm[1] / pm[0]
        flat = math.sqrt(r) if r > 0 else 0.0
    if not_almost_zero(pm[D - 2]):
        r = pm[D - 1] / pm[D - 2]
        elong = math.sqrt(r) if r > 0 else 0.0
    phys = n * size_per_pixel
    eq_r = hypersphere_radius_from_volume(D, phys)
    eq_p = hypersphere_perimeter(D, eq_r)
    edet = 1.0
    for v in pm:
        edet *= v
    edet = edet ** (1.0 / D) if edet >= 0 else float("nan")
    ell = np.zeros(D)
    for i in range(D):
        if edet != 0.0 and pm[i] / edet > 0.0:
            ell[i] = 2.0 * eq_r * math.sqrt(pm[i] / edet)

    feret = _feret(obj, sp)
    perim = _perimeter(rows, x0, length, mins, bbox_size, sp)
    roundness = eq_p / perim if perim else float("inf")
    pob_ratio = perim_border / perim if perim else float("nan")

    # --- statistics (feature image) ------------------------------------------
    fmin = float(feat.min())
    fmax = float(feat.max())
    v = feat[obj].astype(np.float64)  # raster order == line order
    axis = Axis(nbins, fmin, fmax, clip=False)
    hidx = axis.index(v)
    freq = np.bincount(hidx, minlength=nbins)
    total = n
    vmin = float(v.min())
    vmax = float(v.max())
    s1 = float(np.add.accumulate(v)[-1])
    s2 = float(np.add.accumulate(v ** 2)[-1])
    s3 = float(np.add.accumulate(v ** 3)[-1])
    s4 = float(np.add.accumulate(v ** 4)[-1])
    mean = s1 / total
    var = (s2 - s1 ** 2 / total) / (total - 1) if total > 1 else 0.0
    sigma = _sqrt(var)
    mean2 = mean * mean
    tiny = np.finfo(float).tiny
    skew = ((s3 - 3.0 * mean * s2) / total + 2.0 * mean * mean2) / (var * sigma) if abs(var * sigma) > tiny else 0.0
    kurt = (((s4 - 4.0 * mean * s3 + 6.0 * mean2 * s2) / total - 3.0 * mean2 * mean2) / (var * var) - 3.0
            if abs(var) > tiny else 0.0)
    median = 0.0
    cnt = 0.0
    for i in range(nbins):
        cnt += freq[i]
        if cnt >= total // 2:
            median = axis.center(i)
            break

    pts = geom.points(np.argwhere(obj)[:, ::-1])
    cog = np.zeros(D)
    wcm = np.zeros((D, D))
    for i in range(D):
        cog[i] = float(np.sum(pts[:, i] * v))
        for j in range(D):
            wcm[i, j] = float(np.sum(v * pts[:, i] * pts[:, j]))
    welong = wflat = 0.0
    wpm = np.zeros(D)
    waxes = np.zeros((D, D))
    if not_almost_zero(s1):
        cog /= s1
        wcm /= s1
        wcm -= np.outer(cog, cog)
        for i in range(D):
            wcm[i, i] += sp[i] * sp[i] / 12.0
        wpm, waxes = _principal_axes(wcm)
        if not_almost_zero(wpm[0]):
            welong = _sqrt(wpm[D - 1] / wpm[D - 2])
            wflat = _sqrt(wpm[1] / wpm[0])
    else:
        cog[:] = 0

    return [
        ("CenterOfGravity", fmt_vec(cog)),
        ("Centroid", fmt_vec(pcent)),
        ("Elongation", fmt(elong)),
        ("EquivalentEllipsoidDiameter", fmt_vec(ell)),
        ("EquivalentSphericalPerimeter", fmt(eq_p)),
        ("EquivalentSphericalRadius", fmt(eq_r)),
        ("FeretDiameter", fmt(feret)),
        ("Kurtosis", fmt(kurt)),
        ("Flatness", fmt(flat)),
        ("Maximum", fmt(vmax)),
        ("Mean", fmt(mean)),
        ("Median", fmt(median)),
        ("Minimum", fmt(vmin)),
        ("NumberOfLines", fmt(n_lines)),
        ("NumberOfPixels", fmt(n)),
        ("NumberOfPixelsOnBorder", fmt(n_border)),
        ("Perimeter", fmt(perim)),
        ("PerimeterOnBorder", fmt(perim_border)),
        ("PerimeterOnBorderRatio", fmt(pob_ratio)),
        ("PhysicalSize", fmt(phys)),
        ("PrincipalAxes", fmt_itk_matrix(paxes)),
        ("PrincipalMoments", fmt_vec(pm)),
        ("ReferenceCount", "1"),
        ("RegionElongation", fmt(elong)),
        ("Roundness", fmt(roundness)),
        ("Skewness", fmt(skew)),
        ("StandardDeviation", fmt(sigma)),
        ("Sum", fmt(s1)),
        ("Variance", fmt(var)),
        ("WeightedElongation", fmt(welong)),
        ("WeightedFlatness", fmt(wflat)),
        ("WeightedPrincipalAxes", fmt_itk_matrix(waxes)),
        ("WeightedPrincipalMoments", fmt_vec(wpm)),
    ]
