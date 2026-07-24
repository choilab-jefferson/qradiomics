"""Regression test for the CSV union-header logic shared by qr extract / qr shape.

Both writers used to fix the header from the FIRST patient's keys, so a later
patient with a different key set (e.g. a degenerate mask that drops some 3D
shape features, or a per-patient spiculation failure that emits `spic_error`
instead of `spic_*`) crashed csv.DictWriter (default extrasaction="raise") or
misaligned columns. union_fieldnames builds a header that tolerates that.
"""
import csv
import io

from qradiomics.extractor import union_fieldnames


def test_union_preserves_first_order_and_appends_new_keys():
    rows = [
        {"patient_id": "A", "f1": 1, "f2": 2},
        {"patient_id": "B", "f1": 3, "f2": 4, "f3": 5},  # extra key
        {"patient_id": "C", "f1": 6},                     # missing keys
    ]
    assert union_fieldnames(rows) == ["patient_id", "f1", "f2", "f3"]


def test_union_header_writes_without_raising():
    rows = [
        {"patient_id": "A", "spic_s1": 0.5, "spic_s2": 0.1},
        {"patient_id": "B", "spic_error": "mesh failed"},  # different schema
    ]
    fieldnames = union_fieldnames(rows)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, restval="")
    writer.writeheader()
    writer.writerows(rows)  # must not raise
    out = buf.getvalue().splitlines()
    assert out[0] == "patient_id,spic_s1,spic_s2,spic_error"
    # B's missing feature columns are blank, error lands in its own column
    assert out[2] == "B,,,mesh failed"


def test_union_empty():
    assert union_fieldnames([]) == []
