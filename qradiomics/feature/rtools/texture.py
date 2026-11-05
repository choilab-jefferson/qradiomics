"""``c`` (GLCM, ``ScalarImageToTextureFeaturesFilter``) and ``r`` (GLRLM,
``ScalarImageToRunLengthFeaturesFilter``) families -- ITK 5.1.2 semantics."""
from __future__ import annotations

import math

import numpy as np

from ._itk import Axis, Geom, euclidean, fmt

GLCM_NAMES = [
    "Energy", "Entropy", "Correlation", "InverseDifferenceMoment",
    "Inertia", "ClusterShade", "ClusterProminence", "HaralickCorrelation",
]
GLRLM_NAMES = [
    "ShortRunEmphasis", "LongRunEmphasis", "GreyLevelNonuniformity",
    "RunLengthNonuniformity", "LowGreyLevelRunEmphasis", "HighGreyLevelRunEmphasis",
    "ShortRunLowGreyLevelEmphasis", "ShortRunHighGreyLevelEmphasis",
    "LongRunLowGreyLevelEmphasis", "LongRunHighGreyLevelEmphasis",
]


def itk_offsets(dim: int = 3):
    """The 13 (3D) / 4 (2D) "previous" neighbours of a radius-1
    ``Neighborhood`` in ITK (x, y, z) order."""
    out = []
    n = 3 ** dim
    for d in range(n // 2):
        out.append(tuple((d // 3 ** k) % 3 - 1 for k in range(dim)))
    return out


def knuth_mean_sd(features: np.ndarray):
    """Mean / population SD across offsets with Knuth's recurrence."""
    m = features[0].astype(np.float64).copy()
    s = np.zeros_like(m)
    for k in range(1, features.shape[0]):
        x = features[k]
        mk = m + (x - m) / (k + 1)
        s = s + (x - m) * (x - mk)
        m = mk
    return m, np.sqrt(s / features.shape[0])


def _shift_pairs(shape, off_np):
    """Slices selecting (center, neighbour) pairs for a numpy-order offset."""
    c, n = [], []
    for size, o in zip(shape, off_np):
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
def glcm_matrix(img, mask, offset_itk, axis: Axis):
    nb = axis.size
    bins = axis.index(img.astype(np.float64))
    c, n = _shift_pairs(img.shape, offset_itk[::-1])
    ok = mask[c] & mask[n]
    a = bins[c][ok]
    b = bins[n][ok]
    good = (a >= 0) & (b >= 0)
    a, b = a[good], b[good]
    h = np.zeros((nb, nb), dtype=np.int64)  # h[i1, i0]; instance id = i0 + i1*nb
    np.add.at(h, (b, a), 1)
    np.add.at(h, (a, b), 1)
    return h


def haralick_features(h: np.ndarray) -> np.ndarray:
    """``HistogramToTextureFeaturesFilter::GenerateData`` (8 features)."""
    nb = h.shape[0]
    freq_abs = h.ravel().astype(np.float64)  # id order, i0 fastest
    total = float(freq_abs.sum())
    f = freq_abs / total
    ids = np.arange(nb * nb)
    i0 = ids % nb
    i1 = ids // nb
    acc = np.add.accumulate
    pixel_mean = float(acc(i0 * f)[-1])
    marg = np.array([float(acc(f[i0 == k])[-1]) for k in range(nb)])
    mm = marg[0]
    md = 0.0
    for k in range(1, nb):
        x = marg[k]
        mk = mm + (x - mm) / (k + 1)
        md = md + (x - mm) * (x - mk)
        mm = mk
    md = md / nb
    pvar = float(acc((i0 - pixel_mean) * (i0 - pixel_mean) * f)[-1])
    pvs = pvar * pvar
    if abs(pvs) <= 2 * np.finfo(float).eps:
        pvs = 1.0
    nz = f != 0
    f, i0, i1 = f[nz], i0[nz], i1[nz]
    d = i0 - i1
    energy = float(acc(f * f)[-1])
    ent_terms = np.where(f > 0.0001, f * np.log(np.where(f > 0, f, 1)) / math.log(2.0), 0.0)
    entropy = -float(acc(ent_terms)[-1])
    corr = float(acc(((i0 - pixel_mean) * (i1 - pixel_mean) * f) / pvs)[-1])
    idm = float(acc(f / (1.0 + d * d))[-1])
    inertia = float(acc((d * d) * f)[-1])
    cs_base = (i0 - pixel_mean) + (i1 - pixel_mean)
    shade = float(acc(cs_base ** 3 * f)[-1])
    prom = float(acc(cs_base ** 4 * f)[-1])
    har = float(acc((i0 * i1) * f)[-1])
    har = (har - mm * mm) / md if md != 0 else float("nan")
    return np.array([energy, entropy, corr, idm, inertia, shade, prom, har])


def glcm(img, mask, vmin: float, vmax: float, nbins: int):
    """``CalcGlcmFeatures``: (pairs) MeanOf*/StandardDeviationOf*."""
    upper = float(np.float32(vmax) + np.float32(1))
    axis = Axis(nbins, vmin, upper, clip=True)
    feats = np.array([haralick_features(glcm_matrix(img, mask, o, axis)) for o in itk_offsets(img.ndim)])
    mean, sd = knuth_mean_sd(feats)
    return ([("MeanOf" + n, fmt(v), "= ") for n, v in zip(GLCM_NAMES, mean)]
            + [("StandardDeviationOf" + n, fmt(v), "= ") for n, v in zip(GLCM_NAMES, sd)])


# ---------------------------------------------------------------------------
# GLRLM
# ---------------------------------------------------------------------------
def normalize_offset(off):
    off = list(off)
    sign = 1
    met = False
    for i in range(len(off) - 1, -1, -1):
        if met:
            off[i] *= sign
        elif off[i] != 0:
            sign = 1 if off[i] > 0 else -1
            met = True
            off[i] *= sign
    return tuple(off)


def _lines(shape, off_np):
    """All maximal lines of the grid along ``off_np``; returns an
    ``(L, T, ndim)`` index array and an in-bounds mask."""
    shape_a = np.array(shape)
    o = np.array(off_np)
    allidx = np.indices(shape).reshape(len(shape), -1).T
    prev = allidx - o
    start = np.any((prev < 0) | (prev >= shape_a), axis=1)
    starts = allidx[start]
    T = int(shape_a.max())
    pos = starts[:, None, :] + np.arange(T)[None, :, None] * o
    inb = np.all((pos >= 0) & (pos < shape_a), axis=2)
    pos = np.where(inb[..., None], pos, 0)
    return pos, inb


def glrlm_matrix(img, mask, offset_itk, geom: Geom, gaxis: Axis, daxis: Axis,
                 vmin: float, vmax: float, dmax: float):
    off_np = normalize_offset(offset_itk)[::-1]
    pos, inb = _lines(img.shape, off_np)
    flat = tuple(pos[..., k] for k in range(img.ndim))
    v = img[flat].astype(np.float64)
    m = mask[flat] & inb
    kk = gaxis.index(v)
    kk = np.where(inb, kk, -9)
    valid = m & (v >= vmin) & (v <= vmax)
    special = valid & (v == gaxis.maxs[0]) & (v != gaxis.mins[0])
    L, T = kk.shape
    # segment boundaries along t
    new_seg = np.ones((L, T), dtype=bool)
    new_seg[:, 1:] = kk[:, 1:] != kk[:, :-1]
    seg_id = np.cumsum(new_seg.ravel()) - 1
    seg_id = seg_id.reshape(L, T)
    nseg = int(seg_id.max()) + 1
    flat_t = np.broadcast_to(np.arange(T), (L, T))
    # segment end (last t of each segment)
    seg_end = np.zeros(nseg, dtype=np.int64)
    np.maximum.at(seg_end, seg_id.ravel(), flat_t.ravel())
    # first normal valid centre per segment
    big = T + 1
    normal = valid & ~special
    cand = np.where(normal, flat_t, big)
    first = np.full(nseg, big, dtype=np.int64)
    np.minimum.at(first, seg_id.ravel(), cand.ravel())
    # runs: normal centres
    sidx = np.nonzero(first < big)[0]
    # map segment -> line
    seg_line = np.zeros(nseg, dtype=np.int64)
    seg_line[seg_id.ravel()] = np.repeat(np.arange(L), T)
    cl = seg_line[sidx]
    ct = first[sidx]
    et = seg_end[sidx]
    centers = [(cl, ct, et)]
    # zero-length runs from special centres before the segment's first normal one
    sp_l, sp_t = np.nonzero(special)
    if len(sp_l):
        keep = sp_t < first[seg_id[sp_l, sp_t]]
        centers.append((sp_l[keep], sp_t[keep], sp_t[keep]))
    cl = np.concatenate([c[0] for c in centers])
    ct = np.concatenate([c[1] for c in centers])
    et = np.concatenate([c[2] for c in centers])
    grey = v[cl, ct]
    pc = geom.points(pos[cl, ct][:, ::-1])
    pe = geom.points(pos[cl, et][:, ::-1])
    dist = euclidean(pc, pe)
    ok = (dist >= 0) & (dist <= dmax)
    gi = gaxis.index(grey[ok])
    di = daxis.index(dist[ok])
    good = (gi >= 0) & (di >= 0)
    nb0, nb1 = gaxis.size, daxis.size
    h = np.zeros((nb1, nb0), dtype=np.int64)  # h[i1, i0]
    np.add.at(h, (di[good], gi[good]), 1)
    return h


def runlength_features(h: np.ndarray) -> np.ndarray:
    nb1, nb0 = h.shape
    total = float(h.sum())
    ids = np.arange(nb0 * nb1)
    i0 = ids % nb0
    i1 = ids // nb0
    f = h.ravel().astype(np.float64)
    nz = f != 0
    f, i0, i1 = f[nz], i0[nz], i1[nz]
    i2 = ((i0 + 1) * (i0 + 1)).astype(np.float64)
    j2 = ((i1 + 1) * (i1 + 1)).astype(np.float64)
    acc = np.add.accumulate

    def s(x):
        return float(acc(x)[-1]) if x.size else 0.0

    gv = np.zeros(nb0)
    rv = np.zeros(nb1)
    np.add.at(gv, i0, f)
    np.add.at(rv, i1, f)
    out = np.array([
        s(f / j2), s(f * j2), float(np.dot(gv, gv)), float(np.dot(rv, rv)),
        s(f / i2), s(f * i2), s(f / (i2 * j2)), s(f * i2 / j2), s(f * j2 / i2), s(f * i2 * j2),
    ])
    with np.errstate(divide="ignore", invalid="ignore"):
        return out / total


def glrlm(img, mask, geom: Geom, vmin: float, vmax: float, dist_max: float, nbins: int):
    """``CalcGlrmFeatures`` (ITK 5.1 run-length filter, physical run length)."""
    gmax = float(np.float32((np.float32(vmax) - np.float32(vmin)) * np.float32(2) + np.float32(vmin)))
    dmax = dist_max * 2 / 3
    gaxis = Axis(nbins, vmin, gmax, clip=True)
    daxis = Axis(nbins, 0.0, dmax, clip=True)
    feats = np.array([
        runlength_features(glrlm_matrix(img, mask, o, geom, gaxis, daxis, vmin, gmax, dmax))
        for o in itk_offsets(img.ndim)
    ])
    mean, sd = knuth_mean_sd(feats)
    return ([("MeanOf" + n, fmt(v), "= ") for n, v in zip(GLRLM_NAMES, mean)]
            + [("StandardDeviationOf" + n, fmt(v), "= ") for n, v in zip(GLRLM_NAMES, sd)])
