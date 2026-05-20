#!/usr/bin/env bash
# =============================================================================
# NSCLC-Radiomics (LUNG1) — survival analysis
# =============================================================================
# Cohort  : https://www.cancerimagingarchive.net/collection/nsclc-radiomics/
# Paper   : Aerts HJWL et al., Nature Communications 2014.
# Pipeline: TCIA-shipped NRRD (or convert from DICOM) → qr extract
#           with the bundled `nsclc-survival` pattern (1130 features per
#           patient with Original + LoG + Wavelet + Square + SquareRoot +
#           Logarithm × 7 feature classes) → merge with the clinical CSV
#           on patient_id → univariate Cox PH on overall survival.
# =============================================================================
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the LUNG1 TCIA directory}"
CLINICAL_CSV="${CLINICAL_CSV:?set CLINICAL_CSV to the LUNG1 clinical CSV}"
OUT="${OUT:-./runs/lung1}"

mkdir -p "$OUT"

# ─── 1. Build manifest from the NRRD tree ────────────────────────────────────
# If your tree contains DICOM instead of NRRD, run qr convert first
# (see nsclc_cetuximab_survival.sh for the DICOM path).
qr convert manifest-from-dir \
  -d "$DATASET_ROOT" \
  --image-glob '*_CT.nrrd' \
  --mask-glob '*_CT_manual_gtv-1-label.nrrd' \
  --modality CT \
  -o "$OUT/manifest.csv"

# ─── 2. Extract radiomics features ───────────────────────────────────────────
qr extract \
  -m "$OUT/manifest.csv" \
  -p nsclc-survival \
  -o "$OUT/features.csv"
# → ~1130 features per patient

# ─── 3. Merge with clinical OS data on patient_id ────────────────────────────
qr results merge \
  -f "$OUT/features.csv" \
  -c "$CLINICAL_CSV" \
  --clinical-id-col patient_id \
  --time-col OS_days \
  --event-col OS_event \
  -o "$OUT/analysis_ready.csv"

# ─── 4. Univariate Cox PH on every radiomics feature ─────────────────────────
qr analyze survival \
  -i "$OUT/analysis_ready.csv" \
  --outcome OS_months --event OS_event \
  -o "$OUT/cox_results.csv" \
  --top-n 20

echo ""
echo "Done. Outputs in $OUT/:"
echo "  manifest.csv          patients × {patient_id, modality, image_path, mask_path}"
echo "  features.csv          patients × ~1130 PyRadiomics features"
echo "  analysis_ready.csv    merged features + OS_months + OS_event"
echo "  cox_results.csv       univariate Cox PH (feature, HR, CI, p) ranked by p"
