"""Bit-faithful numpy re-implementations of the ITK 5.1.2 primitives that
the radiomics-tools ``FeatureExtraction`` binary relies on.

Only what the binary actually exercises is ported:

* ``itk::Statistics::Histogram::Initialize`` bin-edge arithmetic (float32
  interval, double edges) and ``GetIndex`` look-up (``ClipBinsAtEnds``).
* ``ImageBase::TransformIndexToPhysicalPoint`` (sequential accumulation).
* ``GeometryUtilities`` hyper-sphere helpers.
* C++ ``std::ostream`` default number formatting (precision 6, ``%g``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Image geometry
# --------------------------------------------------------------------------
@dataclass
class Geom:
    """ITK image geometry. ``spacing``/``origin`` in ITK (x, y, z) order,
    ``direction`` as an ITK row-major matrix."""

    spacing: np.ndarray
    origin: np.ndarray
    direction: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.spacing)

    @property
    def index_to_physical(self) -> np.ndarray:
        # ITK: m_IndexToPhysicalPoint = Direction * diag(Spacing)
        return self.direction * self.spacing[np.newaxis, :]

    def points(self, idx: np.ndarray) -> np.ndarray:
        """Physical points for an ``(N, dim)`` (continuous) index array,
        accumulated in the same order as ``TransformIndexToPhysicalPoint``."""
        idx = np.asarray(idx, dtype=np.float64)
        m = self.index_to_physical
        out = np.empty(idx.shape, dtype=np.float64)
        for i in range(self.dim):
            acc = np.full(idx.shape[0], self.origin[i], dtype=np.float64)
            for j in range(self.dim):
                acc = acc + m[i, j] * idx[:, j]
            out[:, i] = acc
        return out


def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``Point::EuclideanDistanceTo`` (row-wise)."""
    acc = np.zeros(a.shape[0], dtype=np.float64)
    for i in range(a.shape[1]):
        d = a[:, i] - b[:, i]
        acc = acc + d * d
    return np.sqrt(acc)


# --------------------------------------------------------------------------
# itk::Statistics::Histogram (one axis)
# --------------------------------------------------------------------------
class Axis:
    """One histogram axis built exactly like ``Histogram::Initialize``."""

    def __init__(self, size: int, lower: float, upper: float, clip: bool = True):
        self.size = int(size)
        self.clip = clip
        f32 = np.float32
        interval = f32((f32(upper) - f32(lower)) / f32(size))
        lower = float(lower)
        j = np.arange(size - 1, dtype=np.float32)
        mins = np.empty(size, dtype=np.float64)
        maxs = np.empty(size, dtype=np.float64)
        mins[:-1] = lower + (j * interval).astype(np.float64)
        maxs[:-1] = lower + ((j + f32(1)) * interval).astype(np.float64)
        mins[-1] = lower + float(f32(size - 1) * interval)
        maxs[-1] = float(upper)
        self.mins = mins
        self.maxs = maxs

    def index(self, v: np.ndarray) -> np.ndarray:
        """``Histogram::GetIndex`` for one axis; ``-1`` marks a rejected
        measurement (``ClipBinsAtEnds`` true and outside the range)."""
        v = np.asarray(v, dtype=np.float64)
        idx = np.searchsorted(self.mins, v, side="right") - 1
        below = v < self.mins[0]
        top = self.maxs[-1]
        above = v >= top
        if self.clip:
            near = np.abs(v - top) <= 4 * np.spacing(abs(top)) + 0.1 * np.finfo(float).eps
            idx = np.where(below, -1, idx)
            idx = np.where(above, np.where(near, self.size - 1, -1), idx)
        else:
            idx = np.where(below, 0, idx)
            idx = np.where(above, self.size - 1, idx)
        return idx.astype(np.int64)

    def center(self, i: int) -> float:
        return (self.mins[i] + self.maxs[i]) / 2.0


# --------------------------------------------------------------------------
# GeometryUtilities
# --------------------------------------------------------------------------
def _gamma_n2p1(n: int) -> float:
    if n % 2 == 0:
        return float(math.factorial(n // 2))
    dfact = 1
    k = n
    while k >= 2:
        dfact *= k
        k -= 2
    return math.sqrt(math.pi) * dfact / math.pow(2, (n + 1) / 2.0)


def hypersphere_volume(dim: int, radius: float) -> float:
    return math.pow(math.pi, dim * 0.5) * math.pow(radius, dim) / _gamma_n2p1(dim)


def hypersphere_perimeter(dim: int, radius: float) -> float:
    return dim * hypersphere_volume(dim, radius) / radius


def hypersphere_radius_from_volume(dim: int, volume: float) -> float:
    return math.pow(volume * _gamma_n2p1(dim) / math.pow(math.pi, dim * 0.5), 1.0 / dim)


def not_almost_zero(x: float) -> bool:
    """``Math::NotAlmostEquals(x, 0.0)`` (4 ULPs / 0.1*eps absolute)."""
    return abs(x) > 0.1 * np.finfo(float).eps


# --------------------------------------------------------------------------
# std::ostream formatting
# --------------------------------------------------------------------------
def fmt(x) -> str:
    """C++ ``ostream << double`` with the default precision (6)."""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if math.isnan(x):
        # glibc prints the sign bit of a NaN (std::sqrt(-1) -> "-nan")
        return "-nan" if math.copysign(1.0, x) < 0 else "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return "%g" % x


def fmt_vec(xs) -> str:
    """``itk::FixedArray``/``Point``/``Vector`` stream format."""
    return "[" + ", ".join(fmt(x) for x in xs) + "]"


def fmt_itk_matrix(m) -> str:
    """``itk::Matrix`` stream format (one row per line)."""
    return "".join(" ".join(fmt(v) for v in row) + "\n" for row in np.asarray(m))


def fmt_vnl_matrix(m) -> str:
    """``vnl_matrix`` stream format (trailing blank per element)."""
    return "".join("".join(fmt(v) + " " for v in row) + "\n" for row in np.asarray(m))


def seqsum(a) -> float:
    """Left-to-right double accumulation (``for … sum += a[i]``)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    return float(np.add.accumulate(a)[-1])


def runs(mask: np.ndarray):
    """Run-length encode a boolean array along its last (ITK x) axis in
    raster order -- the ``LabelObject`` line container.

    Returns ``(rows, x0, length)``: ``rows`` is an ``(N, ndim-1)`` array of
    numpy-order leading coordinates (e.g. ``(z, y)``)."""
    X = mask.shape[-1]
    flat = mask.reshape(-1, X)
    pad = np.zeros((flat.shape[0], X + 2), dtype=np.int8)
    pad[:, 1:-1] = flat
    d = np.diff(pad, axis=1)
    rs, xs = np.nonzero(d == 1)
    _, xe = np.nonzero(d == -1)
    rows = np.stack(np.unravel_index(rs, mask.shape[:-1]), axis=1) if mask.ndim > 1 else np.zeros((len(rs), 0), int)
    return rows, xs.astype(np.int64), (xe - xs).astype(np.int64)
