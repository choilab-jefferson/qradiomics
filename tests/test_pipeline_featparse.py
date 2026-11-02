"""Tests for pipelines/common/featparse.py — shared FeatureExtraction
key=value text parser extracted from extract_postpt.py, extract_via_docker.py
and extract_via_docker_multi.py (pipelines/heart_local/).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.common.featparse import parse_feature_txt  # noqa: E402


SAMPLE_TEXT = """\
# FeatureExtraction output
Volume=1234.5
SurfaceArea = 456.789
PrincipalAxes1=0.998271
PrincipalAxes2=-0.001234
Eigenvectors=[1,0,0]
BadValue=notanumber

Compactness=  0.512
"""


def test_parse_basic_key_value_pairs():
    feats = parse_feature_txt(SAMPLE_TEXT)
    assert feats["Volume"] == 1234.5
    assert feats["SurfaceArea"] == 456.789
    assert feats["Compactness"] == 0.512


def test_parse_skips_principal_axes_and_eigenvectors_lines():
    feats = parse_feature_txt(SAMPLE_TEXT)
    assert not any("PrincipalAxes" in k for k in feats)
    assert "Eigenvectors" not in feats


def test_parse_skips_non_numeric_and_blank_and_no_equals_lines():
    feats = parse_feature_txt(SAMPLE_TEXT)
    assert "BadValue" not in feats
    # comment line ("# FeatureExtraction output") has no '=' -> skipped, no
    # stray key produced from it.
    assert all("#" not in k for k in feats)


def test_parse_value_truncated_at_bracket_keeps_scalar_prefix():
    # A trailing bracketed unit/annotation after a numeric scalar is dropped,
    # keeping only the scalar.
    feats = parse_feature_txt("Elongation=0.85 [ratio]\n")
    assert feats == {"Elongation": 0.85}


def test_parse_value_entirely_bracketed_array_is_dropped():
    # When the value starts immediately with '[' (a bracketed array with no
    # scalar prefix), the truncated value is empty and fails float() -> the
    # key is dropped entirely (matches the original inline parsers' behavior).
    feats = parse_feature_txt("Compactness=[0.512]\n")
    assert feats == {}


def test_parse_empty_text():
    assert parse_feature_txt("") == {}
