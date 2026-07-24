"""Sanity tests for the Phantom-FDA Layout 4 ground-truth manifest.

The values are transcribed from the TCIA per-layout document and
documented in ``wiki/data/PHANTOM_FDA.md``. These tests guard the
transcription itself: shape count, expected sets of (diameter, shape,
density), defect flag on the right-lung 10mm/+100HU elliptical, and
the mm-coordinate sanity-check ranges (the wiki §4 cross-check
showing the values are only physically plausible in mm).
"""

from __future__ import annotations

import pytest

from qradiomics.phantom_fda import LAYOUT_4, Layout
from qradiomics.phantom_fda.layouts import LAYOUT_4_BY_ID


# ---------------------------------------------------------------------------
# Layout shape
# ---------------------------------------------------------------------------

def test_layout_4_has_twelve_nodules():
    assert len(LAYOUT_4) == 12
    assert isinstance(LAYOUT_4, Layout)
    assert LAYOUT_4.layout_id == 4
    assert LAYOUT_4.attachment == "attached"


def test_diameters_are_10_or_20_mm():
    diameters = {n.diameter_mm for n in LAYOUT_4.nodules}
    assert diameters == {10.0, 20.0}


def test_shapes_are_ell_lob_spi_only():
    shapes = {n.shape for n in LAYOUT_4.nodules}
    assert shapes == {"ELL", "LOB", "SPI"}


def test_densities_are_minus630_or_plus100():
    densities = {n.hu for n in LAYOUT_4.nodules}
    assert densities == {-630, 100}


def test_full_factorial_two_diameters_x_three_shapes_x_two_densities():
    """Two-each-of-twelve: layout 4 is the (10, 20) x (ELL, LOB, SPI) x
    (-630, +100) factorial, so each shape appears 4 times and each
    density appears 6 times."""
    for shape in ("ELL", "LOB", "SPI"):
        assert len(LAYOUT_4.by_shape(shape)) == 4, f"shape {shape}"
    for hu in (-630, 100):
        assert len(LAYOUT_4.by_density(hu)) == 6, f"hu {hu}"


# ---------------------------------------------------------------------------
# Per-nodule integrity
# ---------------------------------------------------------------------------

def test_each_nodule_has_unique_id_and_lung_assigned():
    ids = [n.nodule_id for n in LAYOUT_4.nodules]
    assert ids == list(range(1, 13))
    assert all(n.lung in ("L", "R") for n in LAYOUT_4.nodules)


def test_each_nodule_has_three_dimensional_center():
    for n in LAYOUT_4.nodules:
        assert len(n.center_mm) == 3
        assert all(isinstance(c, float) for c in n.center_mm)


def test_lookup_by_id_round_trips():
    for n in LAYOUT_4.nodules:
        assert LAYOUT_4_BY_ID[n.nodule_id] is n


def test_defect_flagged_on_nodule_2_only():
    """The 10mm / +100HU right-lung ELL has a known manufacturing
    defect (Nodule 6 replacement planned per TCIA notes)."""
    defective = [n for n in LAYOUT_4.nodules if n.exclude]
    assert len(defective) == 1
    n = defective[0]
    assert n.nodule_id == 2
    assert n.lung == "R"
    assert n.shape == "ELL"
    assert n.diameter_mm == 10.0
    assert n.hu == 100
    assert "defect" in n.notes.lower()


def test_included_subset_drops_the_defect():
    assert len(LAYOUT_4.included()) == 11
    assert 2 not in {n.nodule_id for n in LAYOUT_4.included()}


# ---------------------------------------------------------------------------
# Volume sanity (against nominal-sphere baseline)
# ---------------------------------------------------------------------------

import math


@pytest.mark.parametrize("diameter, lo, hi", [
    # 10 mm sphere = 524 mm^3; manufactured values are non-spherical
    # and span +1% to +6% over the sphere baseline.
    (10.0, 500.0, 600.0),
    # 20 mm sphere = 4189 mm^3.
    (20.0, 4100.0, 4500.0),
])
def test_volumes_within_plausible_range_for_diameter(diameter, lo, hi):
    for n in LAYOUT_4.nodules:
        if n.diameter_mm != diameter:
            continue
        assert lo <= n.volume_mm3 <= hi, (
            f"nodule #{n.nodule_id}: V={n.volume_mm3} mm^3 outside "
            f"[{lo}, {hi}] for nominal diameter {diameter} mm"
        )


def test_nominal_sphere_volumes_within_1_to_10_percent_of_measured():
    """Manufacturing tolerance vs theoretical sphere — keeps the
    transcription honest."""
    for n in LAYOUT_4.nodules:
        r = n.diameter_mm / 2.0
        v_sphere = (4.0 / 3.0) * math.pi * r ** 3
        rel = (n.volume_mm3 - v_sphere) / v_sphere
        assert -0.05 <= rel <= 0.10, (
            f"nodule #{n.nodule_id}: |V-V_sphere|/V_sphere = {rel:+.1%}"
        )


# ---------------------------------------------------------------------------
# Coordinate frame sanity (mm interpretation cross-check)
# ---------------------------------------------------------------------------
#
# Per wiki/data/PHANTOM_FDA.md §4: the (x, y, z) numbers are only
# physically plausible as mm in the DICOM world frame, not as voxel
# indices. Asserting the per-axis spans guards against future edits
# that might silently re-interpret the manifest.

def test_x_span_matches_phantom_thorax_width():
    xs = [n.center_mm[0] for n in LAYOUT_4.nodules]
    assert min(xs) == pytest.approx(133.0)
    assert max(xs) == pytest.approx(401.0)


def test_y_span_matches_phantom_thorax_depth():
    ys = [n.center_mm[1] for n in LAYOUT_4.nodules]
    assert min(ys) == pytest.approx(248.0)
    assert max(ys) == pytest.approx(354.0)


def test_z_span_matches_phantom_length_with_rows_separated_by_shape():
    zs_by_shape = {
        s: [n.center_mm[2] for n in LAYOUT_4.by_shape(s)]
        for s in ("ELL", "LOB", "SPI")
    }
    # The vasculature insert has three rows; the manifest's z column
    # clusters the four ELL nodules around z ~ 100-180, LOB around
    # z ~ 270-350, SPI around z ~ 475-530. Re-check whenever the
    # manifest is edited.
    assert max(zs_by_shape["ELL"]) < min(zs_by_shape["LOB"]), \
        "ELL row should be inferior to the LOB row in z"
    assert max(zs_by_shape["LOB"]) < min(zs_by_shape["SPI"]), \
        "LOB row should be inferior to the SPI row in z"
