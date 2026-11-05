"""Parity of the native rtools port against the legacy C++ binary.

Synthetic inputs only.  Skips when the ``FeatureExtraction`` binary is
absent (set ``QRADIOMICS_RTOOLS_BIN`` to point at it).  The binary prints 6
significant digits, so the port must reproduce every printed value exactly.
"""
from __future__ import annotations

import pytest

from qradiomics.feature.rtools import feature_text, feature_text_cxx
from qradiomics.feature.rtools.extractor import default_binary_path
from qradiomics.feature.rtools.parity import compare_texts, family_table, synthetic_case

pytestmark = pytest.mark.skipif(not default_binary_path().is_file(),
                                reason="radiomics-tools FeatureExtraction binary not found")

SEEDS = list(range(12))


@pytest.mark.parametrize("seed", SEEDS)
def test_port_matches_binary(seed, tmp_path):
    ip, mp, label = synthetic_case(seed, tmp_path)
    cxx = feature_text_cxx(ip, mp, label=label)
    py = feature_text(ip, mp, label=label)
    rows = compare_texts(cxx, py)
    assert len(rows) >= 130
    bad = [r for r in rows if not r[5]]
    table = family_table(rows)
    print(f"seed={seed} " + " ".join(f"{f}:{ne}/{n}" for f, (n, ne, _) in table.items()))
    assert not bad, bad[:10]


@pytest.mark.parametrize("select,bins", [("c", 32), ("r", 16), ("2s", 128), ("ig", 64)])
def test_port_matches_binary_options(select, bins, tmp_path):
    ip, mp, label = synthetic_case(7, tmp_path)
    cxx = feature_text_cxx(ip, mp, label=label, feature_select=select, number_of_bins=bins)
    py = feature_text(ip, mp, label=label, feature_select=select, number_of_bins=bins)
    rows = compare_texts(cxx, py)
    assert rows and all(r[5] for r in rows), [r for r in rows if not r[5]][:10]
