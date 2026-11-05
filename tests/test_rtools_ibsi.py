"""IBSI-1 digital-phantom validation of the opt-in rtools IBSI profile.

The phantom and the consensus reference values live in
``tests/fixtures/ibsi1_digital_phantom.json`` (no data directory needed).
Configuration: no interpolation / re-segmentation, fixed bin size 1 from the
ROI minimum (the phantom's integer grey levels).  Every reference feature is
reported pass/fail; the test requires all implemented features to pass.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from qradiomics.feature.rtools.ibsi import compare_reference, extract_ibsi_arrays

FIXTURE = Path(__file__).parent / "fixtures" / "ibsi1_digital_phantom.json"


@pytest.fixture(scope="module")
def rows():
    fx = json.loads(FIXTURE.read_text())
    vals = extract_ibsi_arrays(np.array(fx["image_zyx"]), np.array(fx["mask_zyx"]), fx["spacing_xyz"],
                               bin_width=1.0)
    return compare_reference(vals, fx["reference"])


def _family(tag):
    return tag.split("_")[0]


def test_ibsi_phantom_report(rows):
    total, passed = Counter(), Counter()
    for tag, v, ref, tol, ok in rows:
        total[_family(tag)] += 1
        passed[_family(tag)] += ok
        print(f"{'PASS' if ok else 'FAIL'} {tag}: value={v} ref={ref} tol={tol}")
    for fam in sorted(total):
        print(f"family {fam}: {passed[fam]}/{total[fam]}")
    failed = [r for r in rows if not r[4]]
    assert not failed, failed


def test_default_outputs_unchanged_by_ibsi_import():
    """Importing the IBSI profile must not alter the C++-exact path."""
    import qradiomics.feature.rtools.extractor as ex

    assert ex.extract_features.__kwdefaults__["ibsi"] is False


def test_ibsi_profile_via_extract_features(tmp_path):
    from qradiomics.feature.rtools import extract_features
    from qradiomics.feature.rtools.parity import synthetic_case

    ip, mp, label = synthetic_case(2, tmp_path)
    feats = extract_features(ip, mp, label=label, ibsi=True, bin_number=32)
    assert "cm_joint_avg_3D_comb" in feats and "rlm_sre_3D_avg" in feats and "stat_mean" in feats
    default = extract_features(ip, mp, label=label)
    assert "MeanOfEnergy" in default and "cm_joint_avg_3D_comb" not in default
