"""Tests for pipelines/common/engine_extract.py's extract_features_multi.

Covers the two paths lidc_idri's extract_features.py / extract_cir.py /
extract_lungx.py now go through:
  * default (engine=None -> resolves to ["pyradiomics"] alone) -- must call
    qradiomics.atomic.extract_features directly, unchanged.
  * any other resolved engine list -- must route through
    qradiomics.extractor_registry.run_multi_extraction via a scratch
    one-row manifest.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.common import engine_extract
from qradiomics.extractor_registry import resolve_engines


def _tiny_ct_mask():
    """A tiny (8x8x8) CT + binary mask pair, geometrically aligned."""
    rng = np.random.default_rng(0)
    ct_arr = rng.uniform(-200, 200, size=(8, 8, 8)).astype(np.float32)
    ct = sitk.GetImageFromArray(ct_arr)

    mask_arr = np.zeros((8, 8, 8), dtype=np.uint8)
    mask_arr[2:6, 2:6, 2:6] = 1
    mask = sitk.GetImageFromArray(mask_arr)
    mask.CopyInformation(ct)
    return ct, mask


def test_default_engine_resolves_to_pyradiomics_only():
    assert resolve_engines(None) == ["pyradiomics"]


def test_pyradiomics_only_calls_extract_features_directly():
    ct, mask = _tiny_ct_mask()
    fake_features = {"original_firstorder_Mean": 1.0}

    with patch.object(
        engine_extract, "extract_features", return_value=fake_features
    ) as mock_extract, patch.object(engine_extract, "run_multi_extraction") as mock_multi:
        result = engine_extract.extract_features_multi(ct, mask, engine=None)

    mock_extract.assert_called_once()
    mock_multi.assert_not_called()
    assert result == fake_features


def test_single_non_default_engine_routes_through_registry():
    ct, mask = _tiny_ct_mask()

    def fake_run_multi(engines, job_id, manifest_path, job_dir, extraction_settings, jobs=1):
        assert engines == ["pysera"]
        pd.DataFrame([{"patient_id": "row", "pysera_firstorder_Mean": 5.0}]).to_csv(
            job_dir / "features.csv", index=False
        )
        return {"status": "extracted"}

    with patch.object(engine_extract, "extract_features") as mock_extract, patch.object(
        engine_extract, "run_multi_extraction", side_effect=fake_run_multi
    ) as mock_multi:
        result = engine_extract.extract_features_multi(ct, mask, engine="pysera")

    mock_extract.assert_not_called()
    mock_multi.assert_called_once()
    assert result == {"pysera_firstorder_Mean": 5.0}


def test_multi_engine_prefixes_and_merges_columns():
    ct, mask = _tiny_ct_mask()

    def fake_run_multi(engines, job_id, manifest_path, job_dir, extraction_settings, jobs=1):
        assert engines == ["pyradiomics", "pysera"]
        pd.DataFrame(
            [
                {
                    "patient_id": "row",
                    "pyradiomics_firstorder_Mean": 1.0,
                    "pysera_firstorder_Mean": 5.0,
                }
            ]
        ).to_csv(job_dir / "features.csv", index=False)
        return {"status": "extracted"}

    with patch.object(engine_extract, "run_multi_extraction", side_effect=fake_run_multi):
        result = engine_extract.extract_features_multi(ct, mask, engine="pyradiomics,pysera")

    assert result.get("pyradiomics_firstorder_Mean") == 1.0
    assert result.get("pysera_firstorder_Mean") == 5.0
    assert "patient_id" not in result


def test_no_features_produced_raises():
    ct, mask = _tiny_ct_mask()

    def fake_run_multi(engines, job_id, manifest_path, job_dir, extraction_settings, jobs=1):
        return {"status": "error"}  # no features.csv written

    with patch.object(engine_extract, "run_multi_extraction", side_effect=fake_run_multi):
        try:
            engine_extract.extract_features_multi(ct, mask, engine="pysera")
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
