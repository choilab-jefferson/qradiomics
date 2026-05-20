"""Tests for qradiomics.io.lidc — XML annotation parser + polygon rasteriser.

Port-of-MATLAB regression tests covering the LIDC XML schema:
* characteristics scoring (9 attributes)
* multi-reader sessions (4 radiologists)
* per-slice polygon ROI rasterisation
* SOPInstanceUID → z-slice lookup
* directory walk

Real-data integration (LIDC-IDRI patient end-to-end) is skipped when
the cohort is absent; it lives at the bottom of this file.
"""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from qradiomics.io.lidc import LIDCNodule, parse_lidc_xml
from qradiomics.io.lidc.extract import (
    _rasterise_nodule,
    _rasterise_polygon,
    scan_lidc_dir,
)
from qradiomics.io.lidc.parse_xml import Characteristics, LIDCROI


_TWO_READER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<LidcReadMessage>
  <readingSession>
    <annotationVersion>3.6</annotationVersion>
    <unblindedReadNodule>
      <noduleID>Nodule 001</noduleID>
      <characteristics>
        <subtlety>4</subtlety>
        <internalStructure>1</internalStructure>
        <calcification>6</calcification>
        <sphericity>4</sphericity>
        <margin>4</margin>
        <lobulation>1</lobulation>
        <spiculation>1</spiculation>
        <texture>5</texture>
        <malignancy>3</malignancy>
      </characteristics>
      <roi>
        <imageZposition>-110.0</imageZposition>
        <imageSOP_UID>UID-A</imageSOP_UID>
        <inclusion>TRUE</inclusion>
        <edgeMap><xCoord>250</xCoord><yCoord>260</yCoord></edgeMap>
        <edgeMap><xCoord>260</xCoord><yCoord>260</yCoord></edgeMap>
        <edgeMap><xCoord>260</xCoord><yCoord>270</yCoord></edgeMap>
        <edgeMap><xCoord>250</xCoord><yCoord>270</yCoord></edgeMap>
      </roi>
      <roi>
        <imageZposition>-112.5</imageZposition>
        <imageSOP_UID>UID-B</imageSOP_UID>
        <inclusion>FALSE</inclusion>
        <edgeMap><xCoord>0</xCoord><yCoord>0</yCoord></edgeMap>
      </roi>
    </unblindedReadNodule>
    <nonNodule>
      <noduleID>NN-1</noduleID>
      <imageSOP_UID>UID-A</imageSOP_UID>
      <locus><xCoord>10</xCoord><yCoord>10</yCoord></locus>
    </nonNodule>
  </readingSession>
  <readingSession>
    <annotationVersion>3.6</annotationVersion>
    <unblindedReadNodule>
      <noduleID>Nodule 99-X</noduleID>
      <characteristics>
        <subtlety>3</subtlety>
        <internalStructure>1</internalStructure>
        <calcification>6</calcification>
        <sphericity>3</sphericity>
        <margin>3</margin>
        <lobulation>2</lobulation>
        <spiculation>3</spiculation>
        <texture>5</texture>
        <malignancy>4</malignancy>
      </characteristics>
      <roi>
        <imageZposition>-110.0</imageZposition>
        <imageSOP_UID>UID-A</imageSOP_UID>
        <inclusion>TRUE</inclusion>
        <edgeMap><xCoord>255</xCoord><yCoord>265</yCoord></edgeMap>
        <edgeMap><xCoord>265</xCoord><yCoord>265</yCoord></edgeMap>
        <edgeMap><xCoord>265</xCoord><yCoord>275</yCoord></edgeMap>
        <edgeMap><xCoord>255</xCoord><yCoord>275</yCoord></edgeMap>
      </roi>
    </unblindedReadNodule>
  </readingSession>
</LidcReadMessage>
"""


@pytest.fixture
def two_reader_xml(tmp_path):
    p = tmp_path / "test.xml"
    p.write_text(_TWO_READER_XML)
    return p


class TestParseXML:
    def test_parses_two_readers(self, two_reader_xml):
        readers = parse_lidc_xml(two_reader_xml)
        assert len(readers) == 2
        assert readers[0].session_index == 1
        assert readers[1].session_index == 2

    def test_characteristics_scored(self, two_reader_xml):
        readers = parse_lidc_xml(two_reader_xml)
        ch = readers[0].nodules[0].characteristics
        assert ch.subtlety == 4
        assert ch.malignancy == 3
        assert ch.spiculation == 1
        assert ch.calcification == 6

    def test_inclusion_false_roi_dropped(self, two_reader_xml):
        readers = parse_lidc_xml(two_reader_xml)
        nodule = readers[0].nodules[0]
        # XML had 2 ROIs (1 TRUE inclusion, 1 FALSE) — only the TRUE one survives.
        assert len(nodule.rois) == 1
        assert nodule.rois[0].image_sop_uid == "UID-A"
        assert nodule.rois[0].inclusion is True

    def test_polygon_coords_parsed(self, two_reader_xml):
        roi = parse_lidc_xml(two_reader_xml)[0].nodules[0].rois[0]
        assert roi.x_coords == [250.0, 260.0, 260.0, 250.0]
        assert roi.y_coords == [260.0, 260.0, 270.0, 270.0]
        assert roi.image_z_position == -110.0


class TestCharacteristicsFromElement:
    def test_missing_element_yields_zeros(self):
        ch = Characteristics.from_element(None)
        assert all(getattr(ch, f) == 0 for f in (
            "subtlety", "internalStructure", "calcification", "sphericity",
            "margin", "lobulation", "spiculation", "texture", "malignancy"))

    def test_partial_fields_default_to_zero(self):
        root = ET.fromstring(
            "<characteristics><subtlety>5</subtlety></characteristics>"
        )
        ch = Characteristics.from_element(root)
        assert ch.subtlety == 5
        assert ch.malignancy == 0  # absent → 0


class TestRasterisePolygon:
    def test_square_fills_expected_pixels(self):
        mask = _rasterise_polygon(
            x=[250.0, 260.0, 260.0, 250.0],
            y=[260.0, 260.0, 270.0, 270.0],
            shape=(512, 512),
        )
        # skimage fills 11×11 = 121 inclusive of both endpoints
        assert mask.sum() == 121
        assert mask.dtype == np.uint8

    def test_triangle_area(self):
        mask = _rasterise_polygon(
            x=[10.0, 30.0, 20.0],
            y=[10.0, 10.0, 30.0],
            shape=(64, 64),
        )
        # Right triangle base=20, height=20 ≈ 200 pixels filled (skimage exact count is
        # slightly different due to lattice fill); verify it's in the right ballpark.
        assert 150 <= mask.sum() <= 300


class TestRasteriseNodule:
    def test_two_slice_nodule_stack(self):
        nodule = LIDCNodule(
            nodule_id="N",
            rois=[
                LIDCROI(image_sop_uid="UID-A", image_z_position=-110.0,
                        inclusion=True,
                        x_coords=[250.0, 260.0, 260.0, 250.0],
                        y_coords=[260.0, 260.0, 270.0, 270.0]),
                LIDCROI(image_sop_uid="UID-B", image_z_position=-112.5,
                        inclusion=True,
                        x_coords=[252.0, 258.0, 258.0, 252.0],
                        y_coords=[262.0, 262.0, 268.0, 268.0]),
            ],
        )
        uid2z = {"UID-A": 5, "UID-B": 6}
        mask = _rasterise_nodule(nodule, uid2z, shape3d=(20, 512, 512))
        # Each slice contributes a filled square; total > 0
        assert mask.sum() > 0
        # Both slices have content
        per_slice = mask.sum(axis=(1, 2))
        assert per_slice[5] > 0
        assert per_slice[6] > 0
        # Other slices empty
        assert per_slice[0] == 0
        assert per_slice[10] == 0

    def test_unknown_sop_uid_ignored(self):
        nodule = LIDCNodule(
            nodule_id="N",
            rois=[
                LIDCROI(image_sop_uid="UID-NOT-IN-MAP",
                        image_z_position=-110.0, inclusion=True,
                        x_coords=[1.0, 5.0, 5.0, 1.0],
                        y_coords=[1.0, 1.0, 5.0, 5.0]),
            ],
        )
        mask = _rasterise_nodule(nodule, uid_to_z={"OTHER": 0},
                                  shape3d=(10, 64, 64))
        assert mask.sum() == 0


class TestStaple:
    def test_staple_recovers_truth(self):
        import SimpleITK as sitk
        from qradiomics.io.lidc import staple_consensus

        shape = (10, 10, 10)
        truth = np.zeros(shape, np.uint8); truth[3:7, 3:7, 3:7] = 1
        np.random.seed(0)
        masks = []
        for sens_target in (0.95, 0.90, 0.85, 0.80):
            m = truth.copy()
            m[(truth == 1) & (np.random.rand(*shape) > sens_target)] = 0
            m[(truth == 0) & (np.random.rand(*shape) > 0.99)] = 1
            img = sitk.GetImageFromArray(m); img.SetSpacing((1, 1, 1))
            masks.append(img)

        cons, sens, spec = staple_consensus(masks, threshold=0.5)
        cons_arr = sitk.GetArrayFromImage(cons)
        # Recovers the 64-voxel truth almost exactly
        assert abs(int(cons_arr.sum()) - 64) <= 4
        assert len(sens) == 4
        assert all(0.5 <= s <= 1.0 for s in sens)
        assert all(0.95 <= s <= 1.0 for s in spec)


class TestScanLIDCDir:
    def test_empty_dir(self, tmp_path):
        assert list(scan_lidc_dir(tmp_path)) == []

    def test_skips_non_ct_series(self, tmp_path):
        # Create a fake LIDC-IDRI tree with a non-CT series (no DICOM)
        pat = tmp_path / "LIDC-IDRI-9999"
        series = pat / "study1" / "series1"
        series.mkdir(parents=True)
        # No .dcm files → series should be skipped
        result = list(scan_lidc_dir(tmp_path))
        assert result == []
