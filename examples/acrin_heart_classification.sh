#!/usr/bin/env bash
# =============================================================================
# ACRIN-NSCLC-FDG-PET — Heart-ROI radiomics classification
# =============================================================================
# Cohort  : https://www.cancerimagingarchive.net/collection/acrin-nsclc-fdg-pet/
# Pipeline: Per-patient preCT DICOM → qr convert dicom-series → NRRD.
#           Per-patient RTSTRUCT (auto or manual) → qr convert rtstruct
#           --roi Heart (case-insensitive: matches 'Heart' or 'heart')
#           → label NRRD aligned with the CT geometry → manifest → qr extract
#           with `ct-default` (1409 features per patient) → merge with the
#           supplied clinical CSV → univariate logistic regression
#           classification on a binary outcome.
# =============================================================================
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the ACRIN TCIA directory}"
CLINICAL_CSV="${CLINICAL_CSV:?set CLINICAL_CSV to the ACRIN clinical CSV}"
OUTCOME="${OUTCOME:-fdg_uptake_binary}"
OUT="${OUT:-./runs/acrin}"

NRRD_DIR="$OUT/nrrd"
mkdir -p "$NRRD_DIR"

# ─── 1. Per-patient DICOM CT + RTSTRUCT → NRRD ───────────────────────────────
for PAT_DIR in "$DATASET_ROOT"/ACRIN*/; do
    PAT=$(basename "$PAT_DIR")
    CT_DIR="$PAT_DIR/$PAT/preCT"
    [ -d "$CT_DIR" ] || { echo "$PAT: no preCT directory — skipping"; continue; }
    RS=$(ls "$CT_DIR"/RS.manual.* 2>/dev/null | head -1)
    [ -z "$RS" ] && RS=$(ls "$CT_DIR"/RS.auto.* 2>/dev/null | head -1)
    [ -z "$RS" ] && { echo "$PAT: no RTSTRUCT — skipping"; continue; }

    OUT_CT="$NRRD_DIR/${PAT}_preCT.nrrd"
    OUT_MASK="$NRRD_DIR/${PAT}_Heart-label.nrrd"

    [ -f "$OUT_CT"   ] || qr convert dicom-series -i "$CT_DIR" -o "$OUT_CT"
    [ -f "$OUT_MASK" ] || qr convert rtstruct -d "$CT_DIR" -r "$RS" --roi Heart -o "$OUT_MASK" \
        || echo "$PAT: Heart ROI not found — skipping"
done

# ─── 2. Manifest ─────────────────────────────────────────────────────────────
{
    echo "patient_id,modality,image_path,mask_path"
    for ct in "$NRRD_DIR"/*_preCT.nrrd; do
        pat=$(basename "$ct" _preCT.nrrd)
        mask="$NRRD_DIR/${pat}_Heart-label.nrrd"
        [ -f "$mask" ] && echo "$pat,CT,$ct,$mask"
    done
} > "$OUT/manifest.csv"

# ─── 3. Extract ──────────────────────────────────────────────────────────────
qr extract \
  -m "$OUT/manifest.csv" \
  -p ct-default \
  -o "$OUT/features.csv"
# → ~1409 features per patient

# ─── 4. Merge with clinical (clinical CSV must contain patient_id + $OUTCOME) ─
qr results merge \
  -f "$OUT/features.csv" \
  -c "$CLINICAL_CSV" \
  --clinical-id-col patient_id \
  --time-col OS_days \
  --event-col OS_event \
  -o "$OUT/analysis_ready.csv" || echo "(merge fell back: no OS columns — using features.csv directly)"

INPUT_FOR_CLASSIFY="$OUT/analysis_ready.csv"
[ -f "$INPUT_FOR_CLASSIFY" ] || INPUT_FOR_CLASSIFY="$OUT/features.csv"

# ─── 5. Univariate logistic regression classification ────────────────────────
qr analyze classify \
  -i "$INPUT_FOR_CLASSIFY" \
  --outcome "$OUTCOME" \
  -o "$OUT/classify_results.csv" \
  --top-n 20

echo ""
echo "Done. Outputs in $OUT/:"
echo "  nrrd/                 *_preCT.nrrd + *_Heart-label.nrrd"
echo "  features.csv          patients × ~1409 features"
echo "  classify_results.csv  univariate logistic regression, ranked by p"
