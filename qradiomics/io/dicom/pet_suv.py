"""PET DICOM → SUVbw NRRD.

Implements the QIBA vendor-neutral SUV pseudocode
(https://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV)).

Three Units encodings supported:
  * BQML  — raw activity concentration; decay-correct injected dose, then
            SUVbw = raw * (weight_g / decayed_dose_Bq)
  * CNTS  — vendor counts with private SUV factor at tag (0x7053, 0x1000)
  * GML   — already in SUV; pass through (factor 1)

The legacy script at
radiomics_pipelines/HeartToxicity_pipeline/scripts/pet_suv.py is the
behavioural reference; this port fixes two bugs (undefined `flist`
reference, wrong-shape direction concatenation) and exposes a
public API.

Originally authored by Wookjin Choi <wookjin.choi@jefferson.edu>.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union


PathLike = Union[str, Path]

# Fallback defaults when injection metadata is missing — values match the
# legacy script (75 kg patient, 420 MBq FDG, 90 min uptake + 15 min prep,
# FDG half-life 6588 s).
_FALLBACK_WEIGHT_G = 75000.0
_FALLBACK_DOSE_BQ = 420_000_000.0
_FALLBACK_UPTAKE_S = 1.75 * 3600
_FALLBACK_HALFLIFE_S = 6588.0


@dataclass
class SUVResult:
    """Output of read_pet_suv. The `estimated` flag tells callers that one
    or more pieces of metadata were missing and a fallback was used."""
    image: "object"          # sitk.Image in SUVbw units
    raw_image: "object"      # sitk.Image in raw DICOM units (intensity-preserved)
    suv_factor: float
    estimated: bool
    units: str               # "BQML" | "CNTS" | "GML" | other


def _parse_time(s: str) -> datetime.datetime:
    for fmt in ("%H%M%S.%f", "%H%M%S"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    raise ValueError(f"unparseable DICOM time: {s!r}")


def _slice_z(ds) -> Optional[float]:
    if hasattr(ds, "SliceLocation") and ds.SliceLocation is not None:
        try:
            return float(ds.SliceLocation)
        except (TypeError, ValueError):
            pass
    if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
        try:
            return float(ds.ImagePositionPatient[2])
        except (TypeError, ValueError, IndexError):
            pass
    return None


def _read_sorted(dicom_files: List[Path]):
    """Read every DICOM file, sort by z-location, return list of (ds, z)."""
    import pydicom
    rows = []
    for p in dicom_files:
        try:
            ds = pydicom.dcmread(str(p))
        except Exception:
            continue
        z = _slice_z(ds)
        if z is None:
            continue
        rows.append((ds, z))
    rows.sort(key=lambda x: x[1])
    return rows


def _geometry(rows):
    """Compute (spacing, origin, direction) from a sorted slice list."""
    import numpy as np
    ds0 = rows[0][0]
    px, py = (float(v) for v in ds0.PixelSpacing)
    zs = [r[1] for r in rows]
    if len(zs) >= 2:
        pz = float(round(float(np.mean(np.diff(zs))), 6))
    else:
        pz = float(getattr(ds0, "SliceThickness", 1.0))
    spacing = (px, py, pz)
    origin = tuple(float(v) for v in ds0.ImagePositionPatient)
    iop = [float(v) for v in ds0.ImageOrientationPatient]   # 6 floats
    row_dir = np.array(iop[:3])
    col_dir = np.array(iop[3:])
    slice_dir = np.cross(row_dir, col_dir)
    direction = (
        row_dir[0], col_dir[0], slice_dir[0],
        row_dir[1], col_dir[1], slice_dir[1],
        row_dir[2], col_dir[2], slice_dir[2],
    )
    return spacing, origin, direction


def compute_suv_factor(ds, latest_ds=None) -> Tuple[float, bool]:
    """Compute SUVbw scale factor for one DICOM dataset (header-only).

    `latest_ds` is the last-acquired slice in the series and is used as a
    secondary scan-time source when needed.
    Returns (factor, estimated).
    """
    import numpy as np

    # Patient weight
    weight_g = _FALLBACK_WEIGHT_G
    estimated = False
    try:
        w = float(ds.PatientWeight) * 1000.0
        if w > 0:
            weight_g = w
        else:
            estimated = True
    except (AttributeError, TypeError, ValueError):
        estimated = True

    units = getattr(ds, "Units", "")

    if units == "GML":
        return 1.0, estimated

    if units == "CNTS":
        try:
            return float(ds[(0x7053, 0x1000)].value), estimated
        except Exception:
            # Fall back to BQML pathway
            pass

    if units != "BQML" and units != "CNTS":
        # Unknown units → identity, but flag as estimated
        return 1.0, True

    # BQML: derive from radiopharmaceutical metadata
    try:
        seriestime = _parse_time(ds.SeriesTime)
        try:
            aq_first = _parse_time(ds.AcquisitionTime)
        except Exception:
            aq_first = seriestime
        aq_last = aq_first
        if latest_ds is not None:
            try:
                aq_last = _parse_time(latest_ds.AcquisitionTime)
            except Exception:
                pass
        # Convention: scan time is earliest available, clamped by series start
        if seriestime <= aq_first:
            scantime = seriestime
        else:
            try:
                # Vendor-private "scan time" tag, when present
                scantime = ds[(0x0009, 0x100d)].value
                if isinstance(scantime, str):
                    scantime = _parse_time(scantime)
            except Exception:
                scantime = aq_first if aq_first < aq_last else aq_last

        ri = ds.RadiopharmaceuticalInformationSequence[0]
        injection_time = _parse_time(ri.RadiopharmaceuticalStartTime)
        half_life = float(ri.RadionuclideHalfLife)
        injected_dose = float(ri.RadionuclideTotalDose)
        diff = scantime - injection_time
        diff_s = diff.seconds + diff.microseconds / 1e6
        decay = float(np.exp(-np.log(2) * diff_s / half_life))
        decayed_dose = injected_dose * decay
        if decayed_dose <= 0:
            raise ValueError("decayed_dose <= 0")
        return weight_g / decayed_dose, estimated
    except Exception:
        decay = float(np.exp(-np.log(2) * _FALLBACK_UPTAKE_S / _FALLBACK_HALFLIFE_S))
        return weight_g / (_FALLBACK_DOSE_BQ * decay), True


def read_pet_suv(dicom_dir: PathLike) -> SUVResult:
    """Read a PET DICOM series directory and return raw + SUVbw images.

    The image array preserves the DICOM rescale slope/intercept (via
    pydicom's `apply_rescale`); SUV is then `raw * suv_factor`. Both
    images share the geometry parsed from the first slice's
    ImageOrientationPatient + ImagePositionPatient.
    """
    import numpy as np
    from pydicom.pixel_data_handlers import apply_rescale
    import SimpleITK as sitk

    files = sorted(Path(dicom_dir).iterdir())
    rows = _read_sorted(files)
    if not rows:
        raise ValueError(f"No DICOM slices with positional metadata under {dicom_dir}")

    raw_stack = np.stack([apply_rescale(ds.pixel_array, ds) for ds, _ in rows], axis=0)
    spacing, origin, direction = _geometry(rows)

    ds0 = rows[0][0]
    ds_last = rows[-1][0]
    factor, estimated = compute_suv_factor(ds0, latest_ds=ds_last)
    units = getattr(ds0, "Units", "")
    suv_stack = raw_stack.astype(np.float32) * factor

    raw_img = sitk.GetImageFromArray(raw_stack)
    raw_img.SetSpacing(spacing)
    raw_img.SetOrigin(origin)
    raw_img.SetDirection(direction)

    suv_img = sitk.GetImageFromArray(suv_stack)
    suv_img.SetSpacing(spacing)
    suv_img.SetOrigin(origin)
    suv_img.SetDirection(direction)

    return SUVResult(image=suv_img, raw_image=raw_img, suv_factor=factor,
                     estimated=estimated, units=units)
