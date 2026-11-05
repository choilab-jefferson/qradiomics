"""``i`` (``LabelStatisticsImageFilter``) and ``g``
(``LabelGeometryImageFilter``) families of ``FeatureExtraction``."""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ._itk import Axis, Geom, fmt, fmt_vec, fmt_vnl_matrix, seqsum


def intensity(img: np.ndarray, mask: np.ndarray):
    """``CalcIntensityFeatures``; returns ``(pairs, inputMin, inputMax)``."""
    v = img[mask].astype(np.float64)
    count = v.size
    s = seqsum(v)
    ss = seqsum(v * v)
    vmin = float(v.min())
    vmax = float(v.max())
    mean = s / count
    var = (ss - s * s / count) / (count - 1.0) if count > 1 else 0.0
    sigma = math.sqrt(var) if var >= 0.0 else 0.0
    axis = Axis(3000, -1024.0, 2000.0, clip=True)
    idx = axis.index(v)
    freq = np.bincount(idx[idx >= 0], minlength=3000)
    total = 0.0
    b = 0
    half = count // 2
    while total <= half and b < 3000:
        total += freq[b]
        b += 1
    b -= 1
    lo, hi = axis.mins[b], axis.maxs[b]
    median = lo + (hi - lo) / 2
    pairs = [
        ("Minimum", fmt(vmin)),
        ("Maximum", fmt(vmax)),
        ("Median ", fmt(median)),
        ("Mean", fmt(mean)),
        ("Sigma", fmt(sigma)),
        ("Variance", fmt(var)),
        ("Sum", fmt(s)),
    ]
    # inputMin/Max are stored as InputPixelType (float)
    return pairs, float(np.float32(vmin)), float(np.float32(vmax))


def _relabel_largest_face_component(mask: np.ndarray) -> np.ndarray:
    """ConnectedComponentImageFilter (face connectivity) followed by
    RelabelComponentImageFilter (stable sort by size) -> label 1."""
    lab, n = ndimage.label(mask)
    if n <= 1:
        return lab == 1
    sizes = np.bincount(lab.ravel())[1:]
    best = int(np.argmax(sizes)) + 1  # argmax keeps the first (stable) max
    return lab == best


def geometry(img: np.ndarray, mask: np.ndarray, geom: Geom):
    """``CalcGeometryFeatures``."""
    obj = _relabel_largest_face_component(mask)
    D = obj.ndim
    idx = np.argwhere(obj)[:, ::-1].astype(np.int64)  # raster order, (x,y,z)
    vals = img[obj].astype(np.float64)
    n = len(idx)
    bb_min = idx.min(axis=0)
    bb_max = idx.max(axis=0)
    bb_size = bb_max - bb_min + 1
    bb_vol = 1.0
    for s in bb_size:
        bb_vol = bb_vol * float(s)
    centroid = idx.sum(axis=0).astype(np.float64) / n
    wsum = seqsum(vals)
    wmom = (idx * np.trunc(vals).astype(np.int64)[:, None]).sum(axis=0)
    wcent = wmom.astype(np.float64) / wsum
    raw2 = (idx.T.astype(np.float64) @ idx.astype(np.float64))
    pix2 = float(np.float32(1.0) / np.float32(12.0))
    ncm = raw2 / n - np.outer(centroid, centroid)
    ncm[np.diag_indices(D)] += pix2
    w, V = np.linalg.eigh(ncm)
    axes = (4 * np.sqrt(w)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ecc = float(np.sqrt((w[D - 1] - w[0]) / np.float64(w[D - 1])))
    elong = float(np.float32(axes[D - 1] / axes[0]))
    orient = math.atan2(V[1, D - 1], V[0, D - 1])
    if orient < 0.0:
        orient += math.pi

    # oriented bounding box (CalculateRotationMatrix + vertices)
    R = V[:, ::-1].copy()
    if float(np.float32(np.linalg.det(R))) < 0:
        R[:, D - 1] = -R[:, D - 1]
    R = R.T
    tp = R @ (idx - centroid).T
    lo = tp.min(axis=1).astype(np.float32) - np.float32(0.5)
    hi = tp.max(axis=1).astype(np.float32) + np.float32(0.5)
    obb = (hi - lo).astype(np.float32)
    obb_vol = 1.0
    for s in obb:
        obb_vol *= float(s)

    sp = geom.spacing
    return [
        ("Volume", fmt(float(n) * sp[0] * sp[1] * sp[2] / 1000)),
        ("IntegratedIntensity", fmt(wsum)),
        ("Centroid", fmt_vec(centroid)),
        ("WeightedCentroid", fmt_vec(wcent)),
        ("AxesLength", fmt_vec(axes)),
        ("MajorAxisLength", fmt(axes[D - 1])),
        ("MinorAxisLength", fmt(axes[0])),
        ("Eccentricity", fmt(ecc)),
        ("Elongation", fmt(elong)),
        ("Orientation", fmt(orient)),
        ("BoundingBoxVolume", fmt(bb_vol * sp[0] * sp[1] * sp[2] / 1000)),
        ("BoundingBoxSize", fmt_vec([int(s) for s in bb_size])),
        ("OrientedBoundingBoxVolume", fmt(obb_vol * sp[0] * sp[1] * sp[2] / 1000)),
        ("OrientedBoundingBoxSize", fmt_vec(obb)),
        ("Eigenvalues", fmt_vec(w), "= "),
        ("Eigenvectors", fmt_vnl_matrix(V)),
        ("RotationMatrix", fmt_vnl_matrix(R)),
    ]
