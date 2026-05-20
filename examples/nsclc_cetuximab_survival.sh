#!/usr/bin/env bash
# =============================================================================
# NSCLC-Cetuximab (RTOG-0617 cetuximab arm) — survival analysis
# =============================================================================
# Cohort  : https://www.cancerimagingarchive.net/collection/nsclc-cetuximab/
# Pipeline: Per-patient DICOM CT series → qr convert dicom-series → NRRD
#           Per-patient RTSTRUCT (PTV by default) → qr convert rtstruct →
#           label NRRD with the CT geometry → manifest → qr extract → merge
#           with the clinical CSV (patid + survival_months + survival_status
#           columns) → univariate Cox PH on overall survival.
# =============================================================================
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the NSCLC-Cetuximab TCIA directory}"
CLINICAL_CSV="${CLINICAL_CSV:?set CLINICAL_CSV to the cetuximab clinical CSV}"
OUT="${OUT:-./runs/nsclc_cetuximab}"
ROI="${ROI:-PTV}"   # CTV / PTV / GTV — varies across the cohort; PTV is the most consistent

NRRD_DIR="$OUT/nrrd"
mkdir -p "$NRRD_DIR"

# ─── 1. Convert per-patient DICOM CT + RTSTRUCT to NRRD ──────────────────────
for PAT_DIR in "$DATASET_ROOT"/*/; do
    PAT=$(basename "$PAT_DIR")
    # TCIA layout: <DATASET_ROOT>/<patient>/<date>/<study_dir>/{CT files,RTSeries/RS.*}
    CT_DIR=$(find "$PAT_DIR" -maxdepth 3 -type d -name '*_CT_*' | head -1)
    RS=$(find "$CT_DIR/RTSeries" -name 'RS.*' 2>/dev/null | head -1)

    OUT_CT="$NRRD_DIR/${PAT}_CT.nrrd"
    OUT_MASK="$NRRD_DIR/${PAT}_${ROI}-label.nrrd"

    [ -z "$CT_DIR" ] && { echo "$PAT: no CT directory — skipping"; continue; }
    [ -z "$RS"     ] && { echo "$PAT: no RTSTRUCT — skipping"; continue; }

    if [ ! -f "$OUT_CT" ]; then
        qr convert dicom-series -i "$CT_DIR" -o "$OUT_CT"
    fi
    if [ ! -f "$OUT_MASK" ]; then
        qr convert rtstruct -d "$CT_DIR" -r "$RS" --roi "$ROI" -o "$OUT_MASK" \
            || echo "$PAT: ROI '$ROI' not found — skipping"
    fi
done

# ─── 2. Build manifest from the freshly-converted NRRDs ──────────────────────
# Flat layout (filename prefix = patient_id) → emit manifest by hand.
{
    echo "patient_id,modality,image_path,mask_path"
    for ct in "$NRRD_DIR"/*_CT.nrrd; do
        pat=$(basename "$ct" _CT.nrrd)
        mask="$NRRD_DIR/${pat}_${ROI}-label.nrrd"
        [ -f "$mask" ] && echo "$pat,CT,$ct,$mask"
    done
} > "$OUT/manifest.csv"

# ─── 3. Extract radiomics features ───────────────────────────────────────────
qr extract \
  -m "$OUT/manifest.csv" \
  -p nsclc-survival \
  -o "$OUT/features.csv"

# ─── 4. Merge with clinical OS data ──────────────────────────────────────────
qr results merge \
  -f "$OUT/features.csv" \
  -c "$CLINICAL_CSV" \
  --clinical-id-col patid \
  --time-col survival_months \
  --event-col survival_status \
  -o "$OUT/analysis_ready.csv"

# ─── 5. Univariate Cox PH ────────────────────────────────────────────────────
qr analyze survival \
  -i "$OUT/analysis_ready.csv" \
  --outcome OS_months --event OS_event \
  -o "$OUT/cox_results.csv" \
  --top-n 20

echo ""
echo "Done. Outputs in $OUT/:"
echo "  nrrd/                 per-patient *_CT.nrrd and *_${ROI}-label.nrrd"
echo "  manifest.csv          patients × {patient_id, modality, image_path, mask_path}"
echo "  features.csv          patients × ~1130 features"
echo "  analysis_ready.csv    features + OS_months + OS_event"
echo "  cox_results.csv       Cox PH per feature, ranked by p"
