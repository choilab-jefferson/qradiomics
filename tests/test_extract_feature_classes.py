"""Regression test: a pattern's feature_classes must actually restrict the
pyradiomics engine.

The extract command printed the pattern's feature_classes but never passed them
to the engine, so every class was always emitted regardless of the pattern.
"""
from uuid import uuid4

import numpy as np
import pytest

pytest.importorskip("radiomics")
import SimpleITK as sitk  # noqa: E402

from qradiomics.extractor import RadiomicsExtractor  # noqa: E402


@pytest.fixture
def phantom_manifest(tmp_path):
    arr = np.random.RandomState(0).randint(-200, 200, (16, 32, 32)).astype("int16")
    mask = np.zeros((16, 32, 32), "uint8")
    zz, yy, xx = np.ogrid[:16, :32, :32]
    mask[((zz - 8) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) < 25] = 1
    for a, name in [(arr, "CT.nrrd"), (mask, "label.nrrd")]:
        im = sitk.GetImageFromArray(a)
        im.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(im, str(tmp_path / name))
    man = tmp_path / "manifest.csv"
    man.write_text(
        "patient_id,modality,image_path,mask_path\n"
        f"p,CT,{tmp_path/'CT.nrrd'},{tmp_path/'label.nrrd'}\n"
    )
    return man


def _classes(csv_path):
    import pandas as pd
    cols = [c for c in pd.read_csv(csv_path).columns if c != "patient_id"]
    return sorted({c.split("_")[1] for c in cols}), len(cols)


def test_feature_classes_restricts_output(phantom_manifest, tmp_path):
    d_all = tmp_path / "all"; d_all.mkdir()
    d_fs = tmp_path / "fs"; d_fs.mkdir()
    RadiomicsExtractor().run_extraction(
        uuid4(), phantom_manifest, d_all, {"image_types": ["Original"]}, 1
    )
    RadiomicsExtractor().run_extraction(
        uuid4(), phantom_manifest, d_fs,
        {"image_types": ["Original"], "feature_classes": ["firstorder", "shape"]}, 1,
    )
    classes_all, n_all = _classes(d_all / "features.csv")
    classes_fs, n_fs = _classes(d_fs / "features.csv")

    assert classes_fs == ["firstorder", "shape"]
    assert set(classes_all) >= {"firstorder", "shape", "glcm", "glrlm"}
    assert n_fs < n_all
