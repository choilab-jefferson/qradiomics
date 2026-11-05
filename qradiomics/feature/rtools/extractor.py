"""Pure-Python port of radiomics-tools ``FeatureExtraction`` (ITK 5.1.2).

``feature_text`` reproduces the binary's output file line by line;
``extract_features`` returns the parsed scalar dict (identical to parsing the
binary's file with :func:`qradiomics.rtools_extractor.parse_feature_txt`).

Pipeline (FeatureExtraction.cxx ``main``):

1. read image as float32 and label as int16; ``mask = (label == Label)``;
2. keep the largest fully-connected object (``BinaryShapeKeepNObjects``);
3. crop image + mask to the mask bounding box (``GetRoi``/``BoundingCheck``
   /``ApplyRoi``); the image inherits the mask's geometry;
4. families ``2 i g s c r`` in that order (``c``/``r`` imply ``i``).
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
from scipy import ndimage

from ._itk import Geom
from .firstorder import geometry, intensity
from .shape import shape_statistics
from .texture import glcm, glrlm

PathLike = Union[str, Path]


def _geom(img) -> Geom:
    d = img.GetDimension()
    return Geom(
        spacing=np.array(img.GetSpacing(), dtype=np.float64),
        origin=np.array(img.GetOrigin(), dtype=np.float64),
        direction=np.array(img.GetDirection(), dtype=np.float64).reshape(d, d),
    )


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=np.ones((3,) * mask.ndim))
    if n <= 1:
        return lab == 1
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (int(np.argmax(sizes)) + 1)


def _preprocess(image, label_img, label: int):
    """Steps 1-3; returns cropped arrays (numpy z,y,x order), geometry and
    the ``distMax`` used by the run-length family."""
    import SimpleITK as sitk

    img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    lab_arr = sitk.GetArrayFromImage(label_img)
    mask = _keep_largest(lab_arr == label)
    if not mask.any():
        raise ValueError(f"label {label} not present in mask")
    D = mask.ndim
    nz = np.argwhere(mask)
    mstart = nz.min(axis=0)[::-1].astype(int)  # ITK (x, y, z)
    msize = (nz.max(axis=0)[::-1] - mstart + 1).astype(int)

    # BoundingCheck
    pt = label_img.TransformIndexToPhysicalPoint([int(v) for v in mstart])
    istart = list(image.TransformPhysicalPointToIndex(pt))
    isize = [0] * D
    ostart = [0] * D
    in_size = image.GetSize()
    for n in range(D):
        isize[n] = int(msize[n])
        ostart[n] = int(mstart[n])
        if istart[n] < 0:
            ostart[n] = ostart[n] - istart[n]
            isize[n] = isize[n] + istart[n]
            istart[n] = 0
        if not isize[n] + istart[n] <= in_size[n]:
            isize[n] = in_size[n] - istart[n] - 1
    osize = list(isize)

    def crop(arr, start, size):
        sl = tuple(slice(start[k], start[k] + size[k]) for k in reversed(range(D)))
        return arr[sl]

    img_c = np.ascontiguousarray(crop(img_arr, istart, isize))
    mask_c = np.ascontiguousarray(crop(mask, ostart, osize))
    lgeom = _geom(label_img)
    origin = lgeom.points(np.array([ostart], dtype=np.float64))[0]
    geom = Geom(spacing=lgeom.spacing, origin=origin, direction=lgeom.direction)
    # RegionToIndex: roiEnd = start + size + 1 (sic)
    roi_start = np.array(ostart, dtype=np.float64)
    roi_end = roi_start + np.array(osize, dtype=np.float64) + 1
    p = geom.points(np.stack([roi_start, roi_end]))
    dist_max = float(np.sqrt(np.sum((p[0] - p[1]) ** 2)))
    return img_c, mask_c, geom, dist_max


def _largest_slice(img: np.ndarray, mask: np.ndarray, geom: Geom):
    """``ExtractLargestAreaSlice`` + 2D label object 1 (first object in
    raster order of the fully-connected 2D labelling)."""
    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    z = 0
    best = 0
    for i, a in enumerate(areas):
        if a > best:
            best, z = a, i
    m2 = mask[z]
    lab, _ = ndimage.label(m2, structure=np.ones((3, 3)))
    obj = lab == 1
    d2 = geom.direction[:2, :2].copy()
    if np.linalg.det(d2) == 0.0:
        d2 = np.eye(2)
    g2 = Geom(spacing=geom.spacing[:2].copy(), origin=geom.origin[:2].copy(), direction=d2)
    return img[z], obj, g2


def _emit(lines, pairs, prefix=""):
    for p in pairs:
        name, value = p[0], p[1]
        sep = p[2] if len(p) > 2 else " = "
        lines.append(f"{prefix}{name}{sep}{value}")


def feature_text_from_images(image, label_img, *, label: int = 1,
                             feature_select: str = "2isgcr", number_of_bins: int = 64) -> str:
    """Return the exact text ``FeatureExtraction`` writes (SimpleITK inputs)."""
    sel = set(feature_select)
    want_i = bool(sel & {"i", "c", "r"})
    img, mask, geom, dist_max = _preprocess(image, label_img, label)
    lines: list[str] = []
    if "2" in sel:
        i2, m2, g2 = _largest_slice(img, mask, geom)
        _emit(lines, shape_statistics(m2, i2, g2, number_of_bins), prefix="2D")
    vmin = vmax = 0.0
    if want_i:
        pairs, vmin, vmax = intensity(img, mask)
        _emit(lines, pairs)
    if "g" in sel:
        _emit(lines, geometry(img, mask, geom))
    if "s" in sel:
        _emit(lines, shape_statistics(mask, img, geom, number_of_bins))
    if "c" in sel:
        _emit(lines, glcm(img, mask, vmin, vmax, number_of_bins))
    if "r" in sel:
        _emit(lines, glrlm(img, mask, geom, vmin, vmax, dist_max, number_of_bins))
    return "\n".join(lines) + "\n"


def feature_text(image_path: PathLike, mask_path: PathLike, **kw) -> str:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    label_img = sitk.ReadImage(str(mask_path), sitk.sitkInt16)
    return feature_text_from_images(image, label_img, **kw)


def default_binary_path() -> Path:
    import os

    override = os.environ.get("QRADIOMICS_RTOOLS_BIN")
    if override:
        return Path(override)
    return Path.home() / "gitRepos/radiomics-tools/bin/FeatureExtraction"


def feature_text_cxx(image_path: PathLike, mask_path: PathLike, *, label: int = 1,
                     feature_select: str = "2isgcr", number_of_bins: int = 64,
                     binary: PathLike | None = None, timeout: float = 600) -> str:
    """Run the legacy C++ binary (parity reference only)."""
    binary = Path(binary) if binary else default_binary_path()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "features.txt"
        subprocess.run([str(binary), str(image_path), str(mask_path), str(out), str(label),
                        feature_select, str(number_of_bins)],
                       capture_output=True, timeout=timeout, check=True)
        return out.read_text()


def extract_features(image_path: PathLike, mask_path: PathLike, *, label: int = 1,
                     feature_select: str = "2isgcr", number_of_bins: int = 64,
                     backend: str = "python", ibsi: bool = False, **ibsi_kw) -> dict[str, float]:
    """Scalar features keyed exactly like the binary's parsed output.

    ``backend="python"`` (default) is the native port; ``backend="cxx"`` shells
    out to the legacy binary (parity checks only).  ``ibsi=True`` switches to
    the opt-in IBSI-1 profile (see :mod:`qradiomics.feature.rtools.ibsi`)."""
    from qradiomics.rtools_extractor import parse_feature_txt

    if ibsi:
        from .ibsi import extract_ibsi

        return extract_ibsi(image_path, mask_path, label=label, **ibsi_kw)
    kw = dict(label=label, feature_select=feature_select, number_of_bins=number_of_bins)
    if backend == "python":
        text = feature_text(image_path, mask_path, **kw)
    elif backend == "cxx":
        text = feature_text_cxx(image_path, mask_path, **kw)
    else:
        raise ValueError(f"unknown backend {backend!r} (use 'python' or 'cxx')")
    return parse_feature_txt(text)
