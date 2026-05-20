"""LIDC-IDRI XML annotation parser.

Port of `fn_nodule_info.m` from the LungCancerScreeningRadiomics MATLAB
pipeline. The LIDC XML schema is:

    <LidcReadMessage>
      <readingSession>           (4 of these per patient — one per radiologist)
        <unblindedReadNodule>    (one or more nodules per session)
          <noduleID>...</noduleID>
          <characteristics>      (optional; only present when nodule has >1 ROI slice)
            <subtlety>1-5</subtlety>
            <internalStructure>1-4</internalStructure>
            <calcification>1-6</calcification>
            <sphericity>1-5</sphericity>
            <margin>1-5</margin>
            <lobulation>1-5</lobulation>
            <spiculation>1-5</spiculation>
            <texture>1-5</texture>
            <malignancy>1-5</malignancy>
          </characteristics>
          <roi>                   (one ROI per slice)
            <imageZposition>...</imageZposition>
            <imageSOP_UID>...</imageSOP_UID>
            <inclusion>TRUE|FALSE</inclusion>
            <edgeMap><xCoord>...</xCoord><yCoord>...</yCoord></edgeMap>
            ...
          </roi>
        </unblindedReadNodule>
        <nonNodule>...</nonNodule>     (small lesions <3mm — we ignore these)
      </readingSession>
    </LidcReadMessage>

We surface three nested dataclasses (LIDCROI → LIDCNodule → LIDCReader)
plus a `parse_lidc_xml()` entry point that returns the full reader list
without mutating image volumes. Mask rasterisation lives in
`extract.py` to keep this module dependency-light (only stdlib + numpy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET


# LIDC XML uses a default namespace. Helper to find tags regardless.
def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _find_all(elem: ET.Element, name: str) -> List[ET.Element]:
    """Tag search ignoring XML namespaces."""
    return [c for c in elem.iter() if _strip_ns(c.tag) == name]


def _find_first(elem: ET.Element, name: str) -> Optional[ET.Element]:
    for c in elem.iter():
        if _strip_ns(c.tag) == name:
            return c
    return None


@dataclass
class Characteristics:
    """Nine LIDC nodule attribute scores. 0 indicates 'not reported'."""

    subtlety: int = 0
    internalStructure: int = 0
    calcification: int = 0
    sphericity: int = 0
    margin: int = 0
    lobulation: int = 0
    spiculation: int = 0
    texture: int = 0
    malignancy: int = 0

    @classmethod
    def from_element(cls, ch_elem: Optional[ET.Element]) -> "Characteristics":
        if ch_elem is None:
            return cls()
        kwargs = {}
        for f in (
            "subtlety", "internalStructure", "calcification", "sphericity",
            "margin", "lobulation", "spiculation", "texture", "malignancy",
        ):
            node = _find_first(ch_elem, f)
            try:
                kwargs[f] = int(float(node.text.strip())) if (node is not None and node.text) else 0
            except (ValueError, AttributeError):
                kwargs[f] = 0
        return cls(**kwargs)


@dataclass
class LIDCROI:
    """One contour on one CT slice."""

    image_sop_uid: str
    image_z_position: float          # mm
    inclusion: bool                  # TRUE = solid contour, FALSE = exclusion (hole)
    x_coords: List[float] = field(default_factory=list)  # pixel x (0-indexed, may be sub-pixel)
    y_coords: List[float] = field(default_factory=list)

    @classmethod
    def from_element(cls, roi: ET.Element) -> Optional["LIDCROI"]:
        uid = _find_first(roi, "imageSOP_UID")
        zpos = _find_first(roi, "imageZposition")
        inc = _find_first(roi, "inclusion")
        xs = _find_all(roi, "xCoord")
        ys = _find_all(roi, "yCoord")
        if uid is None or len(xs) == 0 or len(ys) == 0:
            return None
        try:
            z = float(zpos.text) if zpos is not None and zpos.text else 0.0
        except ValueError:
            z = 0.0
        inclusion = True
        if inc is not None and inc.text:
            inclusion = inc.text.strip().upper() == "TRUE"
        return cls(
            image_sop_uid=(uid.text or "").strip(),
            image_z_position=z,
            inclusion=inclusion,
            x_coords=[float(x.text) for x in xs if x.text],
            y_coords=[float(y.text) for y in ys if y.text],
        )


@dataclass
class LIDCNodule:
    """One nodule annotated by one reader (i.e. one <unblindedReadNodule>)."""

    nodule_id: str
    characteristics: Characteristics = field(default_factory=Characteristics)
    rois: List[LIDCROI] = field(default_factory=list)

    @property
    def is_large(self) -> bool:
        """LIDC convention: a nodule is "large" (>3 mm) when characteristics
        are populated AND it has more than one ROI slice."""
        return self.characteristics.malignancy > 0 and len(self.rois) > 1


@dataclass
class NonNoduleMark:
    """A radiologist non-nodule finding (small lesion / vessel / etc. <3 mm).

    Only a single point (locus) is annotated, not a polygon. Useful as a
    *hard* negative class for nodule detection classifiers (Choi 2014 CMPB)
    — these are real radiologist-flagged objects that are NOT nodules.
    """
    non_nodule_id: str
    image_sop_uid: str
    image_z_position: float       # mm
    x_coord: float                # pixel
    y_coord: float

    @classmethod
    def from_element(cls, nn_elem: ET.Element) -> Optional["NonNoduleMark"]:
        nid = _find_first(nn_elem, "nonNoduleID")
        uid = _find_first(nn_elem, "imageSOP_UID")
        zpos = _find_first(nn_elem, "imageZposition")
        locus = _find_first(nn_elem, "locus")
        if uid is None or locus is None:
            return None
        xn = _find_first(locus, "xCoord")
        yn = _find_first(locus, "yCoord")
        if xn is None or yn is None or not xn.text or not yn.text:
            return None
        try:
            z = float(zpos.text) if zpos is not None and zpos.text else 0.0
            x = float(xn.text); y = float(yn.text)
        except (TypeError, ValueError):
            return None
        return cls(
            non_nodule_id=(nid.text or "").strip() if nid is not None else "",
            image_sop_uid=(uid.text or "").strip(),
            image_z_position=z, x_coord=x, y_coord=y,
        )


@dataclass
class LIDCReader:
    """One radiologist's reading session."""

    session_index: int               # 1..4
    annotation_version: str = ""
    nodules: List[LIDCNodule] = field(default_factory=list)
    non_nodules: List[NonNoduleMark] = field(default_factory=list)


def parse_lidc_xml(xml_path: str | Path) -> List[LIDCReader]:
    """Parse an LIDC ``<LidcReadMessage>`` XML into reader/nodule/ROI dataclasses.

    Returns one ``LIDCReader`` per ``<readingSession>``. Non-nodule findings
    (``<nonNodule>``) are skipped. Ordering matches the XML file order, which
    in LIDC is the same as the original "reading session index" (1..4) used
    in the published characterisations.
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sessions = _find_all(root, "readingSession")

    readers: List[LIDCReader] = []
    for idx, session in enumerate(sessions, start=1):
        reader = LIDCReader(session_index=idx)
        ann_ver = _find_first(session, "annotationVersion")
        if ann_ver is not None and ann_ver.text:
            reader.annotation_version = ann_ver.text.strip()
        for nodule_elem in _find_all(session, "unblindedReadNodule"):
            nid_elem = _find_first(nodule_elem, "noduleID")
            nodule_id = (nid_elem.text or "").strip() if nid_elem is not None else ""
            ch_elem = _find_first(nodule_elem, "characteristics")
            ch = Characteristics.from_element(ch_elem)
            rois: List[LIDCROI] = []
            for roi_elem in _find_all(nodule_elem, "roi"):
                roi = LIDCROI.from_element(roi_elem)
                if roi is not None and roi.inclusion:
                    rois.append(roi)
            reader.nodules.append(LIDCNodule(
                nodule_id=nodule_id, characteristics=ch, rois=rois,
            ))
        # Also collect nonNodule findings for this reader (hard negatives)
        for nn_elem in _find_all(session, "nonNodule"):
            nn = NonNoduleMark.from_element(nn_elem)
            if nn is not None:
                reader.non_nodules.append(nn)
        readers.append(reader)
    return readers
