"""LIDC-IDRI annotation parser — XML → per-reader mask NRRD + characteristics CSV.

Python port of the MATLAB pipeline `lung-image-analysis` /
`LungCancerScreeningRadiomics` (Wookjin Choi, 2014 CMPB / 2021 CMPB).
Reads each LIDC patient's DICOM CT series + paired XML annotation
(`<readingSession>` × 4 radiologists, each with `<unblindedReadNodule>`s
carrying per-slice polygon ROIs and 9 characteristic scores) and emits:

* `<pid>_CT.nrrd`                — the CT volume (uses
  `qradiomics.io.dicom.load_dicom_series` for the conversion).
* `<pid>_CT_Phy{1..4}-label.nrrd` — per-reader binary mask, one per
  radiologist that produced annotations.
* `<pid>_nodules.csv`             — long table (one row per
  reader×nodule) with nodule_id, reading_session, characteristics 1-5,
  volume, bounding box, centroid, mean/min/max intensity.

The "Phy" suffix mirrors the original MATLAB naming where bit-encoded
nodule masks `2^(sid-1)` are split per reader. We keep separate label
files instead of one bit-packed uint8 because pyradiomics expects a
single label value per mask.
"""

from .parse_xml import (
    LIDCNodule,
    LIDCReader,
    LIDCROI,
    parse_lidc_xml,
)
from .extract import (
    convert_patient,
    scan_lidc_dir,
)
from .staple import (
    staple_consensus,
    staple_patient,
)

__all__ = [
    "LIDCNodule",
    "LIDCReader",
    "LIDCROI",
    "convert_patient",
    "parse_lidc_xml",
    "scan_lidc_dir",
    "staple_consensus",
    "staple_patient",
]
