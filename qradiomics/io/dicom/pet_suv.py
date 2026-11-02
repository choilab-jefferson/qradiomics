"""PET DICOM → Standardized Uptake Value (SUV) converter.

Implements the **IBSI-SUV v3.1 reference standard**
(https://theibsi.github.io/ibsi-suv/, reference implementation
https://github.com/oncoray/suv_computation) together with backwards-compatible
QIBA vendor-neutral fallback estimation
(https://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV)).

Supported SOP Classes:
  * Positron Emission Tomography Image Storage (``1.2.840.10008.5.1.4.1.1.128``)
  * Enhanced PET Image Storage (``1.2.840.10008.5.1.4.1.1.130``)
  * Legacy Converted Enhanced PET Image Storage (``1.2.840.10008.5.1.4.1.1.128.1``)

Supported Input Units & Normalization Types:
  * ``BQML``  — radioactivity concentration (Bq/mL) with ``DecayCorrection``
    in ``{"ADMIN", "START", "NONE"}`` (Classic PET) or ``DecayCorrected`` in
    ``{"YES", "NO"}`` (Enhanced PET), including vendor private scan-start tags
    (Siemens ``(0071,1022)`` and GE ``(0009,100D)``), frame-reference-time
    back-computation, frame-average decay offset ``t_ave``, MBq-to-Bq dose
    auto-correction (``< 10^4``), midnight-crossing correction, and
    date-anonymization recovery.
  * ``GML``   — pre-computed SUV in g/mL with ``SUVType`` in
    ``{"", "BW", "LBM", "LBMJAMES128", "LBMJANMA", "IBW"}``.
  * ``CM2ML`` — pre-computed body-surface-area SUV (``SUVbsa``) in cm^2/mL.
  * ``CNTS``  — Philips vendor counts with private activity scale factor
    ``(7053,1009)`` (preferred, routes through ``BQML``) or private SUVbw
    scale factor ``(7053,1000)``.

Target SUV outputs (``suv_type`` / ``target_suv_type``):
  * ``"BW"`` (default, ``SUVbw`` in g/mL)
  * ``"LBM"`` (``SUVlbm`` Janmahasatian/Hallynck 120/148)
  * ``"LBMJAMES128"`` (``SUVlbm(James128)``)
  * ``"LBMJANMA"`` (``SUVlbm(Janma)``)
  * ``"IBW"`` (``SUVibw``)
  * ``"BSA"`` (``SUVbsa`` Kim/DuBois in cm^2/mL)

Originally authored by Wookjin Choi <wookjin.choi@jefferson.edu>.
"""

from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

PathLike = Union[str, Path]

# Fallback defaults when injection metadata is missing and strict=False —
# values match the legacy script (75 kg patient, 420 MBq FDG, 90 min uptake
# + 15 min prep, FDG half-life 6588 s).
_FALLBACK_WEIGHT_G = 75000.0
_FALLBACK_DOSE_BQ = 420_000_000.0
_FALLBACK_UPTAKE_S = 1.75 * 3600
_FALLBACK_HALFLIFE_S = 6588.0

ENHANCED_PET_SOP_CLASSES = (
    "1.2.840.10008.5.1.4.1.1.130",
    "1.2.840.10008.5.1.4.1.1.128.1",
)

SUPPORTED_SUV_TYPES = ("BW", "LBM", "LBMJAMES128", "LBMJANMA", "IBW", "BSA")


@dataclass
class SUVResult:
    """Output of :func:`read_pet_suv`.

    The ``estimated`` flag tells callers that one or more pieces of metadata
    were missing and a fallback was used (only possible when ``strict=False``).
    """

    image: "object"  # sitk.Image in target SUV units (default SUVbw)
    raw_image: "object"  # sitk.Image in raw rescaled DICOM units
    suv_factor: float
    estimated: bool
    units: str  # "BQML" | "CNTS" | "GML" | "CM2ML" | other
    suv_type: str = "BW"
    sop_class_uid: str = ""
    qc_warnings: Tuple[str, ...] = ()


def _parse_time(s: str) -> datetime.datetime:
    s_clean = str(s).strip()
    if not s_clean:
        raise ValueError("unparseable DICOM time: empty string")
    for fmt in ("%H%M%S.%f", "%H%M%S", "%H%M", "%H"):
        try:
            return datetime.datetime.strptime(s_clean, fmt)
        except (ValueError, TypeError):
            continue
    raise ValueError(f"unparseable DICOM time: {s!r}")


def _parse_dicom_date(s: str) -> datetime.date:
    s_clean = str(s).strip()
    if not s_clean:
        raise ValueError("unparseable DICOM date: empty string")
    for fmt in ("%Y%m%d", "%Y.%m.%d"):
        try:
            return datetime.datetime.strptime(s_clean, fmt).date()
        except (ValueError, TypeError):
            continue
    raise ValueError(f"unparseable DICOM date: {s!r}")


def _parse_dicom_datetime(s: str) -> datetime.datetime:
    s_clean = str(s).strip()
    if not s_clean:
        raise ValueError("unparseable DICOM datetime: empty string")
    m = re.match(r"^(\d{8,14}(?:\.\d+)?)([+-]\d{4})?$", s_clean)
    core = m.group(1) if m else s_clean
    for fmt in (
        "%Y%m%d%H%M%S.%f",
        "%Y%m%d%H%M%S",
        "%Y%m%d%H%M",
        "%Y%m%d%H",
        "%Y%m%d",
    ):
        try:
            return datetime.datetime.strptime(core, fmt)
        except (ValueError, TypeError):
            continue
    raise ValueError(f"unparseable DICOM datetime: {s!r}")


def _combine_da_tm(da_val: object, tm_val: object) -> datetime.datetime:
    d = _parse_dicom_date(str(da_val))
    t = _parse_time(str(tm_val)).time()
    return datetime.datetime.combine(d, t)


def _slice_z(ds) -> Optional[float]:
    """Return physical slice z-position.

    Prioritizes ``ImagePositionPatient[2]`` (Type 1 required DICOM attribute in
    patient coordinates) over ``SliceLocation`` (optional Type 3 attribute that
    can have inverted sign in clinical datasets such as ACRIN 6668 /
    ACRIN-NSCLC-FDG-PET-157).
    """
    if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
        try:
            return float(ds.ImagePositionPatient[2])
        except (TypeError, ValueError, IndexError):
            pass
    if hasattr(ds, "SliceLocation") and ds.SliceLocation is not None:
        try:
            return float(ds.SliceLocation)
        except (TypeError, ValueError):
            pass
    return None


def _read_sorted(dicom_files: List[Path]):
    """Read every DICOM file, sort by z-location, return list of (ds, z)."""
    import pydicom

    rows = []
    for p in dicom_files:
        if not p.is_file():
            continue
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
    iop = [float(v) for v in ds0.ImageOrientationPatient]  # 6 floats
    row_dir = np.array(iop[:3])
    col_dir = np.array(iop[3:])
    slice_dir = np.cross(row_dir, col_dir)
    # The cross product gives the mathematical image-plane normal, but for
    # some patient positions (e.g. HFP) it points opposite to the stacking
    # direction (ascending z-sort). Flip it so the direction matrix agrees
    # with the physical first→last slice vector; otherwise the image is
    # mirrored along the slice axis in downstream viewers.
    if len(rows) > 1:
        pos_first = np.array([float(v) for v in rows[0][0].ImagePositionPatient])
        pos_last = np.array([float(v) for v in rows[-1][0].ImagePositionPatient])
        if np.dot(slice_dir, pos_last - pos_first) < 0:
            slice_dir = -slice_dir
    direction = (
        row_dir[0],
        col_dir[0],
        slice_dir[0],
        row_dir[1],
        col_dir[1],
        slice_dir[1],
        row_dir[2],
        col_dir[2],
        slice_dir[2],
    )
    return spacing, origin, direction


def _get_patient_weight_kg_g(ds) -> Tuple[float, float]:
    """Extract patient weight in (kg, g) per IBSI-SUV v3.1 Section 2.1.1."""
    val = getattr(ds, "PatientWeight", None)
    if val is None or str(val).strip() == "":
        raise ValueError("Missing PatientWeight (0010,1030)")
    w = float(val)
    if w <= 0:
        raise ValueError(f"Non-positive PatientWeight (0010,1030): {w}")
    if w >= 1000.0:
        # Weight erroneously recorded in grams instead of kilograms (DRO_1_1)
        return w / 1000.0, w
    return w, w * 1000.0


def _get_patient_height_cm(ds) -> float:
    """Extract patient height in cm from PatientSize (0010,1020) in meters."""
    val = getattr(ds, "PatientSize", None)
    if val is None or str(val).strip() == "":
        raise ValueError("Missing PatientSize (0010,1020)")
    h = float(val)
    if h <= 0:
        raise ValueError(f"Non-positive PatientSize (0010,1020): {h}")
    return h * 100.0


def _get_patient_sex(ds) -> str:
    """Extract PatientSex (0010,0040) in {'M', 'F', 'O'}."""
    val = getattr(ds, "PatientSex", None)
    if val is None:
        raise ValueError("Missing PatientSex (0010,0040)")
    sex = str(val).strip().upper()
    if sex not in ("M", "F", "O"):
        raise ValueError(f"Invalid PatientSex (0010,0040): {sex!r}")
    return sex


def compute_body_norm_factor_g(ds, suv_type: str = "BW") -> float:
    """Return body-size normalization factor in grams (or cm^2 for BSA).

    Follows IBSI-SUV v3.1 Section 2.1 equations for:
      * ``BW``          : m * 1000 [g]
      * ``LBM``         : Hallynck/James (120/148) * 1000 [g]
      * ``LBMJAMES128`` : James (128/148) * 1000 [g]
      * ``LBMJANMA``    : Janmahasatian (2005) * 1000 [g]
      * ``IBW``         : Ideal body weight * 1000 [g]
      * ``BSA``         : DuBois & DuBois (1916) / Kim (1994) * 10000 [cm^2]
    """
    st = str(suv_type or "BW").strip().upper()
    w_kg, w_g = _get_patient_weight_kg_g(ds)
    if st in ("", "BW"):
        return w_g
    h_cm = _get_patient_height_cm(ds)
    if st == "BSA":
        bsa_m2 = 0.007184 * (h_cm**0.725) * (w_kg**0.425)
        return bsa_m2 * 10000.0
    sex = _get_patient_sex(ds)
    if st == "LBM":
        m_val = 1.10 * w_kg - 120.0 * ((w_kg / h_cm) ** 2)
        f_val = 1.07 * w_kg - 148.0 * ((w_kg / h_cm) ** 2)
    elif st == "LBMJAMES128":
        m_val = 1.10 * w_kg - 128.0 * ((w_kg / h_cm) ** 2)
        f_val = 1.07 * w_kg - 148.0 * ((w_kg / h_cm) ** 2)
    elif st == "LBMJANMA":
        bmi = w_kg / ((h_cm * 1e-2) ** 2)
        m_val = (9270.0 * w_kg) / (6680.0 + 216.0 * bmi)
        f_val = (9270.0 * w_kg) / (8780.0 + 244.0 * bmi)
    elif st == "IBW":
        m_val = 48.0 + 1.06 * (h_cm - 152.0)
        f_val = 45.5 + 0.91 * (h_cm - 152.0)
    else:
        raise ValueError(f"Unsupported SUVType: {st!r}")

    if sex == "M":
        factor_kg = m_val
    elif sex == "F":
        factor_kg = f_val
    else:
        factor_kg = 0.5 * (m_val + f_val)
    return factor_kg * 1000.0


def _t_ave_seconds(frame_duration_ms: float, half_life_s: float) -> float:
    """Compute average frame decay offset ``t_ave`` in seconds (IBSI-SUV Section 3.3.2.2.2)."""
    if frame_duration_ms <= 0:
        raise ValueError(f"Non-positive frame duration (ms): {frame_duration_ms}")
    duration_s = frame_duration_ms / 1000.0
    lam = math.log(2.0) / half_life_s
    return (1.0 / lam) * math.log((lam * duration_s) / (1.0 - math.exp(-lam * duration_s)))


def _extract_dose_and_halflife(ds, need_halflife: bool = True) -> Tuple[float, float, object]:
    """Extract (dose_Bq, half_life_s, ri_dataset) per IBSI-SUV Section 3.3.2.1."""
    if (
        not hasattr(ds, "RadiopharmaceuticalInformationSequence")
        or not ds.RadiopharmaceuticalInformationSequence
    ):
        raise ValueError("Missing RadiopharmaceuticalInformationSequence (0054,0016)")
    ri = ds.RadiopharmaceuticalInformationSequence[0]
    dose_raw = getattr(ri, "RadionuclideTotalDose", None)
    if dose_raw is None or str(dose_raw).strip() == "":
        raise ValueError("Missing RadionuclideTotalDose (0018,1074)")
    dose = float(dose_raw)
    if dose <= 0:
        raise ValueError(f"Non-positive RadionuclideTotalDose (0018,1074): {dose}")
    if dose < 1e4:
        # Stored in MBq instead of Bq (common in ACRIN-NSCLC-FDG-PET; DRO_3_0)
        dose *= 1e6

    hl = 0.0
    if need_halflife:
        hl_raw = getattr(ri, "RadionuclideHalfLife", None)
        if hl_raw is None or str(hl_raw).strip() == "":
            raise ValueError("Missing RadionuclideHalfLife (0018,1075)")
        hl = float(hl_raw)
        if hl <= 0:
            raise ValueError(f"Non-positive RadionuclideHalfLife (0018,1075): {hl}")
    return dose, hl, ri


def _resolve_admin_datetime(
    ri,
    t_ref: datetime.datetime,
    half_life: float,
    allow_start_time: bool = True,
) -> datetime.datetime:
    """Resolve radiopharmaceutical administration datetime (IBSI-SUV Figure 3.6 / 4.4)."""
    dt_raw = getattr(ri, "RadiopharmaceuticalStartDateTime", None)
    tm_raw = getattr(ri, "RadiopharmaceuticalStartTime", None)
    has_dt = dt_raw is not None and str(dt_raw).strip() != ""
    has_tm = tm_raw is not None and str(tm_raw).strip() != ""

    if has_dt:
        t_adm = _parse_dicom_datetime(str(dt_raw))
        diff_s = (t_ref - t_adm).total_seconds()
        if -3600.0 <= diff_s < 2.0 * half_life:
            return t_adm
        if half_life >= 41400.0:
            raise ValueError(
                "Uptake duration out of bounds for long-lived radionuclide "
                f"(diff_s={diff_s}, half_life={half_life})"
            )
        t_adm_combined = datetime.datetime.combine(t_ref.date(), t_adm.time())
        if (t_ref - t_adm_combined).total_seconds() < -3600.0:
            t_adm_combined -= datetime.timedelta(days=1)
        return t_adm_combined

    if allow_start_time and has_tm:
        if half_life >= 41400.0:
            raise ValueError(
                "RadiopharmaceuticalStartDateTime (0018,1078) is required for "
                f"long-lived radionuclide (half_life={half_life} >= 41400 s)"
            )
        t_adm_time = _parse_time(str(tm_raw)).time()
        t_adm_combined = datetime.datetime.combine(t_ref.date(), t_adm_time)
        if (t_ref - t_adm_combined).total_seconds() < -3600.0:
            t_adm_combined -= datetime.timedelta(days=1)
        return t_adm_combined

    raise ValueError(
        "Missing RadiopharmaceuticalStartDateTime (0018,1078) and "
        "RadiopharmaceuticalStartTime (0018,1072)"
    )


def _compute_ibsi_suv_factor_strict(
    ds,
    latest_ds=None,
    *,
    target_suv_type: str = "BW",
    allow_synthetic_minimal: bool = False,
) -> float:
    """Compute exact IBSI-SUV v3.1 multiplicative factor (from rescaled pixel units to target SUV)."""
    target_norm_g = compute_body_norm_factor_g(ds, target_suv_type)
    _, w_g = _get_patient_weight_kg_g(ds)
    bw_to_target = target_norm_g / w_g

    units = str(getattr(ds, "Units", "") or "").strip().upper()
    suv_type = str(getattr(ds, "SUVType", "") or "").strip().upper()
    mfg = str(getattr(ds, "Manufacturer", "") or "").strip().upper()

    if units == "GML":
        if suv_type in ("", "BW"):
            return 1.0 * bw_to_target
        if suv_type in ("LBM", "LBMJAMES128", "LBMJANMA", "IBW"):
            src_norm_g = compute_body_norm_factor_g(ds, suv_type)
            return (w_g / src_norm_g) * bw_to_target
        raise ValueError(f"Unsupported SUVType {suv_type!r} for Units=GML")

    if units == "CM2ML":
        if suv_type not in ("", "BSA"):
            raise ValueError(f"Unsupported SUVType {suv_type!r} for Units=CM2ML")
        src_norm_g = compute_body_norm_factor_g(ds, "BSA")
        return (w_g / src_norm_g) * bw_to_target

    cnts_scale = 1.0
    if units == "CNTS":
        if mfg and "PHILIPS" not in mfg:
            raise ValueError(f"Units=CNTS only supported for Philips (got Manufacturer={mfg!r})")
        act_elem = ds.get((0x7053, 0x1009))
        suv_elem = ds.get((0x7053, 0x1000))
        act_f = (
            float(act_elem.value)
            if (act_elem is not None and str(act_elem.value).strip() != "")
            else 0.0
        )
        suv_f = (
            float(suv_elem.value)
            if (suv_elem is not None and str(suv_elem.value).strip() != "")
            else 0.0
        )
        if act_f > 0.0:
            cnts_scale = act_f
            units = "BQML"
        elif suv_f > 0.0 and suv_type in ("", "BW"):
            return suv_f * bw_to_target
        elif allow_synthetic_minimal:
            # In non-strict mode, allow CNTS without vendor tag to try BQML pathway
            units = "BQML"
        else:
            raise ValueError("Missing valid Philips private scale factors for Units=CNTS")

    if units != "BQML":
        raise ValueError(f"Unsupported Units: {units!r}")

    decay_corr = str(getattr(ds, "DecayCorrection", "") or "").strip().upper()
    if not decay_corr and allow_synthetic_minimal:
        decay_corr = "START"
    if decay_corr not in ("ADMIN", "START", "NONE"):
        raise ValueError(f"Invalid DecayCorrection (0054,1102): {decay_corr!r}")

    if decay_corr == "ADMIN":
        dose, _, _ = _extract_dose_and_halflife(ds, need_halflife=False)
        return cnts_scale * (w_g / dose) * bw_to_target

    dose, half_life, ri = _extract_dose_and_halflife(ds, need_halflife=True)

    has_acq_date = (
        getattr(ds, "AcquisitionDate", None) is not None
        and str(getattr(ds, "AcquisitionDate", "")).strip() != ""
    )

    if not has_acq_date and allow_synthetic_minimal:
        # Minimal synthetic dataset path (time-only, used in unit tests without dates)
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
        if seriestime <= aq_first:
            scantime = seriestime
        else:
            try:
                scantime = ds[(0x0009, 0x100D)].value
                if isinstance(scantime, str):
                    scantime = _parse_time(scantime)
            except Exception:
                scantime = aq_first if aq_first < aq_last else aq_last
        injection_time = _parse_time(ri.RadiopharmaceuticalStartTime)
        diff_s = (scantime - injection_time).total_seconds()
        if diff_s < -3600.0:
            diff_s += 86400.0
        decayed_dose = dose * math.exp(-math.log(2.0) * diff_s / half_life)
        if decayed_dose <= 0:
            raise ValueError("decayed_dose <= 0")
        return cnts_scale * (w_g / decayed_dose) * bw_to_target

    if decay_corr == "START":
        t_ref = None
        if "SIEMENS" in mfg and (0x0071, 0x1022) in ds:
            val = ds[(0x0071, 0x1022)].value
            if isinstance(val, bytes):
                val = val.decode("ascii", errors="ignore")
            if val is not None and str(val).strip() != "":
                t_ref = _parse_dicom_datetime(str(val))
        elif "GE" in mfg and (0x0009, 0x100D) in ds:
            val = ds[(0x0009, 0x100D)].value
            if isinstance(val, bytes):
                val = val.decode("ascii", errors="ignore")
            if val is not None and str(val).strip() != "":
                t_ref = _parse_dicom_datetime(str(val))

        if t_ref is None:
            t_acq = _combine_da_tm(ds.AcquisitionDate, ds.AcquisitionTime)
            t_ser = None
            try:
                t_ser = _combine_da_tm(ds.SeriesDate, ds.SeriesTime)
            except Exception:
                t_ser = None
            if t_ser is not None and abs((t_acq - t_ser).total_seconds()) <= 1.0:
                t_ref = t_acq
            elif "GE" in mfg:
                frt_raw = getattr(ds, "FrameReferenceTime", None)
                if frt_raw is None or str(frt_raw).strip() == "":
                    if allow_synthetic_minimal and t_ser is not None and t_ser <= t_acq:
                        t_ref = t_ser
                    else:
                        raise ValueError("Missing FrameReferenceTime (0054,1300)")
                else:
                    frt = float(frt_raw)
                    if frt < 0:
                        raise ValueError(f"Negative FrameReferenceTime (0054,1300): {frt}")
                    t_ref = t_acq - datetime.timedelta(seconds=frt / 1000.0)
            else:
                frt_raw = getattr(ds, "FrameReferenceTime", None)
                if frt_raw is None or str(frt_raw).strip() == "":
                    if allow_synthetic_minimal and t_ser is not None and t_ser <= t_acq:
                        t_ref = t_ser
                    else:
                        raise ValueError("Missing FrameReferenceTime (0054,1300)")
                else:
                    frt = float(frt_raw)
                    if frt < 0:
                        raise ValueError(f"Negative FrameReferenceTime (0054,1300): {frt}")
                    afd_raw = getattr(ds, "ActualFrameDuration", None)
                    if afd_raw is None or str(afd_raw).strip() == "":
                        raise ValueError("Missing ActualFrameDuration (0018,1242)")
                    t_ave = _t_ave_seconds(float(afd_raw), half_life)
                    t_ref = t_acq + datetime.timedelta(seconds=t_ave - frt / 1000.0)
    else:  # NONE
        t_acq = _combine_da_tm(ds.AcquisitionDate, ds.AcquisitionTime)
        afd_raw = getattr(ds, "ActualFrameDuration", None)
        if afd_raw is None or str(afd_raw).strip() == "":
            raise ValueError("Missing ActualFrameDuration (0018,1242) for DecayCorrection=NONE")
        t_ave = _t_ave_seconds(float(afd_raw), half_life)
        t_ref = t_acq + datetime.timedelta(seconds=t_ave)

    t_adm = _resolve_admin_datetime(ri, t_ref, half_life, allow_start_time=True)
    lam = math.log(2.0) / half_life
    diff_s = (t_ref - t_adm).total_seconds()
    decayed_dose = dose * math.exp(-lam * diff_s)
    if decayed_dose <= 0:
        raise ValueError("decayed_dose <= 0")
    return cnts_scale * (w_g / decayed_dose) * bw_to_target


def compute_suv_factor(
    ds,
    latest_ds=None,
    *,
    target_suv_type: str = "BW",
    strict: bool = False,
) -> Tuple[float, bool]:
    """Compute SUV scale factor for one DICOM dataset (header-only).

    When ``strict=True``, strictly enforces the IBSI-SUV v3.1 standard and
    raises :class:`ValueError` on any missing or inconsistent DICOM tag.
    When ``strict=False`` (default), uses the IBSI-SUV v3.1 calculation when
    metadata are valid, and falls back to QIBA estimation (``estimated=True``)
    when patient weight or radiopharmaceutical tags are missing.

    Returns ``(factor, estimated)``.
    """
    if strict:
        factor = _compute_ibsi_suv_factor_strict(
            ds,
            latest_ds=latest_ds,
            target_suv_type=target_suv_type,
            allow_synthetic_minimal=False,
        )
        return factor, False

    try:
        factor = _compute_ibsi_suv_factor_strict(
            ds,
            latest_ds=latest_ds,
            target_suv_type=target_suv_type,
            allow_synthetic_minimal=True,
        )
        return factor, False
    except Exception:
        pass

    # Fallback pathway (strict=False only)
    weight_g = _FALLBACK_WEIGHT_G
    estimated = False
    try:
        _, weight_g = _get_patient_weight_kg_g(ds)
    except Exception:
        estimated = True

    units = str(getattr(ds, "Units", "") or "").strip().upper()
    if units == "GML":
        return 1.0, estimated

    if units == "CNTS":
        try:
            val = float(ds[(0x7053, 0x1000)].value)
            if val > 0:
                return val, estimated
        except Exception:
            pass

    if units not in ("BQML", "CNTS"):
        return 1.0, True

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
        if seriestime <= aq_first:
            scantime = seriestime
        else:
            try:
                scantime = ds[(0x0009, 0x100D)].value
                if isinstance(scantime, str):
                    scantime = _parse_time(scantime)
            except Exception:
                scantime = aq_first if aq_first < aq_last else aq_last

        dose, half_life, ri = _extract_dose_and_halflife(ds, need_halflife=True)
        injection_time = _parse_time(ri.RadiopharmaceuticalStartTime)
        diff_s = (scantime - injection_time).total_seconds()
        decay = float(math.exp(-math.log(2.0) * diff_s / half_life))
        decayed_dose = dose * decay
        if decayed_dose <= 0:
            raise ValueError("decayed_dose <= 0")
        return weight_g / decayed_dose, True
    except Exception:
        decay = float(math.exp(-math.log(2.0) * _FALLBACK_UPTAKE_S / _FALLBACK_HALFLIFE_S))
        return weight_g / (_FALLBACK_DOSE_BQ * decay), True


def _fg_get(pf, sf, seq_name: str):
    if hasattr(pf, seq_name) and getattr(pf, seq_name):
        return getattr(pf, seq_name)
    if sf is not None and hasattr(sf, seq_name) and getattr(sf, seq_name):
        return getattr(sf, seq_name)
    return None


def _parse_ucum_unit(code_val: str) -> Tuple[str, str]:
    c = str(code_val or "").strip()
    cl = c.lower()
    if cl == "bq/ml":
        return "BQML", ""
    if cl in ("g/ml{suvbw}", "g/ml"):
        return "GML", "BW"
    if "lbm(james128)" in cl or "lbmjames128" in cl:
        return "GML", "LBMJAMES128"
    if "lbm(janma)" in cl or "lbmjanma" in cl:
        return "GML", "LBMJANMA"
    if "suvlbm" in cl:
        return "GML", "LBM"
    if "suvibw" in cl:
        return "GML", "IBW"
    if "cm2/ml" in cl:
        return "CM2ML", "BSA"
    return "", ""


def _read_enhanced_pet_suv(
    ds,
    *,
    suv_type: str = "BW",
    strict: bool = False,
) -> SUVResult:
    """Read an Enhanced PET multi-frame DICOM dataset and return SUVResult."""
    import numpy as np
    import SimpleITK as sitk

    arr = ds.pixel_array.astype(np.float64)
    pfs = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if not pfs:
        raise ValueError("Enhanced PET dataset missing PerFrameFunctionalGroupsSequence")
    sf = (
        ds.SharedFunctionalGroupsSequence[0]
        if (hasattr(ds, "SharedFunctionalGroupsSequence") and ds.SharedFunctionalGroupsSequence)
        else None
    )
    n_frames = len(pfs)
    raw_frames = np.zeros_like(arr, dtype=np.float64)
    suv_frames = np.zeros_like(arr, dtype=np.float64)
    factors: List[float] = []
    detected_units = ""

    target_norm_g = compute_body_norm_factor_g(ds, suv_type)
    _, w_g = _get_patient_weight_kg_g(ds)
    bw_to_target = target_norm_g / w_g

    for i in range(n_frames):
        pf = pfs[i]
        rw_seq = _fg_get(pf, sf, "RealWorldValueMappingSequence")
        chosen = None
        if rw_seq:
            candidates = []
            for item in rw_seq:
                slope = getattr(item, "RealWorldValueSlope", None)
                intercept = getattr(item, "RealWorldValueIntercept", None)
                mucs = getattr(item, "MeasurementUnitsCodeSequence", None)
                if slope is None or intercept is None or not mucs:
                    continue
                code_item = mucs[0]
                code_val = (
                    getattr(code_item, "CodeValue", None)
                    or getattr(code_item, "LongCodeValue", None)
                    or getattr(code_item, "URNCodeValue", None)
                    or ""
                )
                u_type, s_type = _parse_ucum_unit(str(code_val))
                if u_type:
                    prio = (
                        0
                        if (u_type == "GML" and s_type == "BW")
                        else (1 if u_type in ("GML", "CM2ML") else 2)
                    )
                    candidates.append((prio, float(slope), float(intercept), u_type, s_type))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                chosen = candidates[0]

        if chosen is None:
            pvt_seq = _fg_get(pf, sf, "PixelValueTransformationSequence")
            if pvt_seq:
                pvt = pvt_seq[0]
                rt = str(getattr(pvt, "RescaleType", "") or "").strip().upper()
                if rt in ("BQML", "GML", "CM2ML"):
                    s_type = "BW" if rt == "GML" else ("BSA" if rt == "CM2ML" else "")
                    chosen = (
                        2,
                        float(pvt.RescaleSlope),
                        float(pvt.RescaleIntercept),
                        rt,
                        s_type,
                    )

        if chosen is None:
            raise ValueError(
                "No valid RealWorldValueMappingSequence or PixelValueTransformationSequence"
            )

        _, slope, intercept, units, frame_suv_type = chosen
        if not detected_units:
            detected_units = units
        u = arr[i] * slope + intercept
        raw_frames[i] = u

        if units == "GML":
            if frame_suv_type in ("", "BW"):
                factor = 1.0 * bw_to_target
            else:
                src_norm_g = compute_body_norm_factor_g(ds, frame_suv_type)
                factor = (w_g / src_norm_g) * bw_to_target
            suv_frames[i] = u * factor
            factors.append(factor)
            continue

        if units == "CM2ML":
            src_norm_g = compute_body_norm_factor_g(ds, "BSA")
            factor = (w_g / src_norm_g) * bw_to_target
            suv_frames[i] = u * factor
            factors.append(factor)
            continue

        # BQML
        dose, half_life, ri = _extract_dose_and_halflife(ds, need_halflife=True)
        decay_corr = str(getattr(ds, "DecayCorrected", "") or "").strip().upper()
        if decay_corr not in ("YES", "NO"):
            raise ValueError(f"Invalid DecayCorrected (0018,9758): {decay_corr!r}")
        if decay_corr == "YES":
            dcdt = getattr(ds, "DecayCorrectionDateTime", None)
            if dcdt is None or str(dcdt).strip() == "":
                raise ValueError("Missing DecayCorrectionDateTime (0018,9701)")
            t_ref = _parse_dicom_datetime(str(dcdt))
        else:
            fc_seq = _fg_get(pf, sf, "FrameContentSequence")
            if not fc_seq:
                raise ValueError("Missing FrameContentSequence (0020,9111)")
            fc = fc_seq[0]
            frdt = getattr(fc, "FrameReferenceDateTime", None)
            if frdt is not None and str(frdt).strip() != "":
                t_ref = _parse_dicom_datetime(str(frdt))
            else:
                fadt = getattr(fc, "FrameAcquisitionDateTime", None)
                fad = getattr(fc, "FrameAcquisitionDuration", None)
                if fadt is None or fad is None or float(fad) <= 0:
                    raise ValueError(
                        "Missing FrameAcquisitionDateTime (0018,9074) or "
                        "FrameAcquisitionDuration (0018,9220)"
                    )
                t_ave = _t_ave_seconds(float(fad), half_life)
                t_ref = _parse_dicom_datetime(str(fadt)) + datetime.timedelta(seconds=t_ave)

        t_adm = _resolve_admin_datetime(ri, t_ref, half_life, allow_start_time=False)
        lam = math.log(2.0) / half_life
        diff_s = (t_ref - t_adm).total_seconds()
        decayed_dose = dose * math.exp(-lam * diff_s)
        factor = (w_g / decayed_dose) * bw_to_target
        suv_frames[i] = u * factor
        factors.append(factor)

    # Sort frames by ImagePositionPatient[2] and construct SimpleITK geometry
    ipp_list = []
    for i in range(n_frames):
        pf = pfs[i]
        pps = _fg_get(pf, sf, "PlanePositionSequence")
        if not pps:
            raise ValueError("Missing PlanePositionSequence (0020,9113)")
        ipp = [float(v) for v in pps[0].ImagePositionPatient]
        ipp_list.append(ipp)

    z_vals = [ipp[2] for ipp in ipp_list]
    order = np.argsort(z_vals)
    raw_stack = raw_frames[order].astype(np.float32)
    suv_stack = suv_frames[order].astype(np.float32)
    sorted_ipp = [ipp_list[idx] for idx in order]
    sorted_zs = [z_vals[idx] for idx in order]

    pms = _fg_get(pfs[order[0]], sf, "PixelMeasuresSequence")
    if pms and hasattr(pms[0], "PixelSpacing"):
        px, py = (float(v) for v in pms[0].PixelSpacing)
        st_default = float(getattr(pms[0], "SliceThickness", 1.0))
    else:
        px, py, st_default = 1.0, 1.0, 1.0
    pz = float(round(float(np.mean(np.diff(sorted_zs))), 6)) if len(sorted_zs) >= 2 else st_default
    spacing = (px, py, pz)
    origin = tuple(sorted_ipp[0])

    pos_seq = _fg_get(pfs[order[0]], sf, "PlaneOrientationSequence")
    if pos_seq and hasattr(pos_seq[0], "ImageOrientationPatient"):
        iop = [float(v) for v in pos_seq[0].ImageOrientationPatient]
    else:
        iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    row_dir = np.array(iop[:3])
    col_dir = np.array(iop[3:])
    slice_dir = np.cross(row_dir, col_dir)
    if len(sorted_ipp) > 1:
        pos_first = np.array(sorted_ipp[0])
        pos_last = np.array(sorted_ipp[-1])
        if np.dot(slice_dir, pos_last - pos_first) < 0:
            slice_dir = -slice_dir
    direction = (
        float(row_dir[0]),
        float(col_dir[0]),
        float(slice_dir[0]),
        float(row_dir[1]),
        float(col_dir[1]),
        float(slice_dir[1]),
        float(row_dir[2]),
        float(col_dir[2]),
        float(slice_dir[2]),
    )

    raw_img = sitk.GetImageFromArray(raw_stack)
    raw_img.SetSpacing(spacing)
    raw_img.SetOrigin(origin)
    raw_img.SetDirection(direction)

    suv_img = sitk.GetImageFromArray(suv_stack)
    suv_img.SetSpacing(spacing)
    suv_img.SetOrigin(origin)
    suv_img.SetDirection(direction)

    sop_uid = str(getattr(ds, "SOPClassUID", "") or "")
    mean_factor = float(np.mean(factors)) if factors else 1.0
    return SUVResult(
        image=suv_img,
        raw_image=raw_img,
        suv_factor=mean_factor,
        estimated=False,
        units=detected_units,
        suv_type=str(suv_type).upper(),
        sop_class_uid=sop_uid,
        qc_warnings=(),
    )


def read_pet_suv(
    dicom_dir: PathLike,
    *,
    suv_type: str = "BW",
    strict: bool = False,
) -> SUVResult:
    """Read a PET DICOM series directory (or multi-frame file) and return raw + SUV images.

    Supports both Classic Positron Emission Tomography Image Storage
    (``1.2.840.10008.5.1.4.1.1.128``) and Enhanced PET Image Storage
    (``1.2.840.10008.5.1.4.1.1.130`` / ``1.2.840.10008.5.1.4.1.1.128.1``)
    following the IBSI-SUV v3.1 reference standard.

    Parameters
    ----------
    dicom_dir : PathLike
        Directory containing PET DICOM slices or a single Enhanced PET DICOM file.
    suv_type : str, default "BW"
        Target SUV normalization type: ``"BW"``, ``"LBM"``, ``"LBMJAMES128"``,
        ``"LBMJANMA"``, ``"IBW"``, or ``"BSA"``.
    strict : bool, default False
        When ``True``, strictly enforces IBSI-SUV v3.1 metadata validation and
        raises :class:`ValueError` if any required tag is missing or invalid.
        When ``False``, logs warnings in ``result.qc_warnings``, sets
        ``result.estimated = True``, and falls back to QIBA default parameters.
    """
    import numpy as np
    import pydicom
    import SimpleITK as sitk

    try:
        from pydicom.pixels import apply_rescale
    except ImportError:
        from pydicom.pixel_data_handlers import apply_rescale

    p_in = Path(dicom_dir)
    if p_in.is_file():
        files = [p_in]
    else:
        files = sorted(p for p in p_in.iterdir() if p.is_file())

    if not files:
        raise ValueError(f"No DICOM files found under {dicom_dir}")

    # Check if the series is Enhanced PET Image Storage
    for p in files:
        try:
            ds_probe = pydicom.dcmread(str(p))
        except Exception:
            continue
        sop_uid = str(getattr(ds_probe, "SOPClassUID", "") or "")
        if sop_uid in ENHANCED_PET_SOP_CLASSES or hasattr(
            ds_probe, "PerFrameFunctionalGroupsSequence"
        ):
            return _read_enhanced_pet_suv(ds_probe, suv_type=suv_type, strict=strict)
        break

    rows = _read_sorted(files)
    if not rows:
        raise ValueError(f"No DICOM slices with positional metadata under {dicom_dir}")

    ds0 = rows[0][0]
    ds_last = rows[-1][0]
    sop_uid = str(getattr(ds0, "SOPClassUID", "") or "")
    units = str(getattr(ds0, "Units", "") or "")

    raw_slices = []
    suv_slices = []
    factors: List[float] = []
    estimated_any = False
    warnings_set = []

    for ds, _ in rows:
        if strict:
            slope_raw = getattr(ds, "RescaleSlope", None)
            intercept_raw = getattr(ds, "RescaleIntercept", None)
            if (
                slope_raw is None
                or str(slope_raw).strip() == ""
                or intercept_raw is None
                or str(intercept_raw).strip() == ""
            ):
                raise ValueError("Missing RescaleSlope (0028,1053) or RescaleIntercept (0028,1052)")
            raw_slice = ds.pixel_array.astype(np.float64) * float(slope_raw) + float(intercept_raw)
            factor_i, est_i = compute_suv_factor(
                ds, latest_ds=ds_last, target_suv_type=suv_type, strict=True
            )
        else:
            raw_slice = apply_rescale(ds.pixel_array, ds).astype(np.float64)
            try:
                factor_i = _compute_ibsi_suv_factor_strict(
                    ds,
                    latest_ds=ds_last,
                    target_suv_type=suv_type,
                    allow_synthetic_minimal=True,
                )
                est_i = False
            except Exception as exc:
                msg = str(exc)
                if msg not in warnings_set:
                    warnings_set.append(msg)
                factor_i, est_i = compute_suv_factor(
                    ds, latest_ds=ds_last, target_suv_type=suv_type, strict=False
                )

        raw_slices.append(raw_slice.astype(np.float32))
        suv_slices.append((raw_slice * factor_i).astype(np.float32))
        factors.append(factor_i)
        if est_i:
            estimated_any = True

    raw_stack = np.stack(raw_slices, axis=0)
    suv_stack = np.stack(suv_slices, axis=0)
    spacing, origin, direction = _geometry(rows)

    raw_img = sitk.GetImageFromArray(raw_stack)
    raw_img.SetSpacing(spacing)
    raw_img.SetOrigin(origin)
    raw_img.SetDirection(direction)

    suv_img = sitk.GetImageFromArray(suv_stack)
    suv_img.SetSpacing(spacing)
    suv_img.SetOrigin(origin)
    suv_img.SetDirection(direction)

    mean_factor = (
        float(factors[0])
        if len(set(round(f, 12) for f in factors)) == 1
        else float(np.mean(factors))
    )

    return SUVResult(
        image=suv_img,
        raw_image=raw_img,
        suv_factor=mean_factor,
        estimated=estimated_any,
        units=units,
        suv_type=str(suv_type).upper(),
        sop_class_uid=sop_uid,
        qc_warnings=tuple(warnings_set),
    )
