"""Parity harness: native port vs. the legacy C++ ``FeatureExtraction``.

Synthetic image/mask pairs only (no PHI, no data directories).  Each
``FeatureExtraction`` output line is attributed to its family by position
(``2 i g s c r``) and compared value by value; the binary prints 6
significant digits, so "exact" means the port's printed text is identical.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

FAMILY_START = [("i", "Minimum"), ("g", "Volume"), ("s", "CenterOfGravity"),
                ("c", "MeanOfEnergy"), ("r", "MeanOfShortRunEmphasis")]
_SKIP = ("Axes", "Eigenvectors", "RotationMatrix")  # sign-ambiguous matrices


def tagged_values(text: str):
    """``[(family, key, value_text)]``; vectors expand to ``key[i]``."""
    out = []
    fam = None
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k.startswith("2D"):
            fam = "2"
        else:
            for f, first in FAMILY_START:
                if k == first and fam != f and (fam is None or "2igscr".index(f) > "2igscr".index(fam)):
                    fam = f
        if any(s in k for s in _SKIP):
            continue
        if v.startswith("["):
            for i, x in enumerate(v.strip("[]").split(",")):
                out.append((fam, f"{k}[{i}]", x.strip()))
        else:
            out.append((fam, k, v))
    return out


def _rel(a: str, b: str) -> float:
    try:
        x, y = float(a), float(b)
    except ValueError:
        return math.inf
    if math.isnan(x) and math.isnan(y):
        return 0.0
    if x == y:
        return 0.0
    return abs(x - y) / max(abs(x), abs(y), 1e-300)


def compare_texts(cxx: str, py: str):
    """Rows ``(family, key, cxx, py, rel_diff, exact)`` in binary order."""
    a, b = tagged_values(cxx), tagged_values(py)
    rows = []
    for i, (fam, k, v) in enumerate(a):
        if i >= len(b) or b[i][1] != k:
            rows.append((fam, k, v, None, math.inf, False))
            continue
        rows.append((fam, k, v, b[i][2], _rel(v, b[i][2]), v == b[i][2]))
    return rows


def synthetic_case(seed: int, outdir: Path):
    """Write one synthetic (image, mask) pair; returns ``(img, mask, label)``.

    Cycles through PET-like, CT-like (values outside the 3000-bin intensity
    histogram) and integer-valued (ties) images, anisotropic spacing,
    oblique direction, islands, border-touching masks and label != 1."""
    import SimpleITK as sitk

    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in rng.integers(12, 40, 3))
    z, y, x = np.indices(shape)
    c = np.array(shape) / 2 + rng.uniform(-3, 3, 3)
    rad = [rng.uniform(2, s / 2) for s in shape]
    r = np.sqrt(sum(((a - cc) / rr) ** 2 for a, cc, rr in zip((z, y, x), c, rad)))
    kind = seed % 4
    label = 2 if kind == 1 else 1
    lab = np.zeros(shape, np.int16)
    m = (r < 1) | ((r < 1.3) & (rng.random(shape) < 0.3))
    lab[m] = label
    if kind == 2:
        lab[:, :, :3][m[:, :, :3]] = 1
        lab[0:3, 0:3, 0:3] = 1
    if kind == 3:
        lab[rng.random(shape) < 0.02] = 1
    if seed % 3 == 0:
        img = rng.gamma(2, 2, shape)
    elif seed % 3 == 1:
        img = rng.normal(40, 300, shape)
    else:
        img = np.round(rng.normal(100, 30, shape))
    im = sitk.GetImageFromArray(img.astype(np.float32))
    im.SetSpacing(tuple(float(v) for v in rng.uniform(0.5, 3, 3)))
    im.SetOrigin(tuple(float(v) for v in rng.uniform(-300, 300, 3)))
    if seed % 5 == 4:
        a = rng.uniform(0, 0.5)
        im.SetDirection((math.cos(a), -math.sin(a), 0, math.sin(a), math.cos(a), 0, 0, 0, 1))
    mk = sitk.GetImageFromArray(lab)
    mk.CopyInformation(im)
    outdir = Path(outdir)
    ip, mp = outdir / f"img{seed}.nrrd", outdir / f"msk{seed}.nrrd"
    sitk.WriteImage(im, str(ip))
    sitk.WriteImage(mk, str(mp))
    return ip, mp, label


def family_table(rows):
    """``{family: (n, n_exact, max_rel)}``."""
    out = {}
    for fam, _k, _a, _b, rel, exact in rows:
        n, ne, mx = out.get(fam, (0, 0, 0.0))
        out[fam] = (n + 1, ne + int(exact), max(mx, rel))
    return out


def main(argv=None) -> int:
    """``python -m qradiomics.feature.rtools.parity [N_SEEDS]`` -- print the
    family x (n exact / n, max rel diff) table over synthetic cases."""
    import sys
    import tempfile
    import time

    from .extractor import default_binary_path, feature_text, feature_text_cxx

    argv = sys.argv[1:] if argv is None else argv
    n_seeds = int(argv[0]) if argv else 20
    if not default_binary_path().is_file():
        print("FeatureExtraction binary not found; nothing to compare")
        return 2
    agg: dict = {}
    t_py = t_cx = 0.0
    with tempfile.TemporaryDirectory() as td:
        for seed in range(n_seeds):
            ip, mp, label = synthetic_case(seed, Path(td))
            t0 = time.time()
            cx = feature_text_cxx(ip, mp, label=label)
            t1 = time.time()
            py = feature_text(ip, mp, label=label)
            t2 = time.time()
            t_cx += t1 - t0
            t_py += t2 - t1
            for fam, (n, ne, mx) in family_table(compare_texts(cx, py)).items():
                a = agg.get(fam, (0, 0, 0.0))
                agg[fam] = (a[0] + n, a[1] + ne, max(a[2], mx))
    print(f"cases={n_seeds} cxx={t_cx:.1f}s python={t_py:.1f}s")
    print("| family | values compared | printed-identical | max rel diff |")
    print("|---|---|---|---|")
    for fam in "2igscr":
        if fam in agg:
            n, ne, mx = agg[fam]
            print(f"| {fam} | {n} | {ne} | {mx:.3g} |")
    return 0 if all(v[0] == v[1] for v in agg.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
