#!/usr/bin/env bash
# =============================================================================
# IBSI digital phantom — radiomics reproducibility validation
# =============================================================================
# Cohort  : https://github.com/theibsi/data_sets
#           "IBSI 1 chapter 2 digital phantom" + "IBSI 1 CT radiomics phantom"
# Pipeline: Clone the IBSI data_sets repo, build a manifest pointing at the
#           bundled NRRD/NIfTI volumes + masks, and run qr extract with the
#           ct-default pattern. Optional: compare PyRadiomics output against
#           the IBSI reference table for each feature class.
# =============================================================================
set -euo pipefail

IBSI_ROOT="${IBSI_ROOT:?set IBSI_ROOT to the cloned theibsi/data_sets directory}"
OUT="${OUT:-./runs/ibsi}"

mkdir -p "$OUT"

# Pick a phantom subset; the digital phantom is single-case + has known
# closed-form feature values.
SUBSET="${SUBSET:-ibsi_1_digital_phantom}"
IMAGE="$IBSI_ROOT/$SUBSET/image/phantom.nrrd"
MASK="$IBSI_ROOT/$SUBSET/mask/mask.nrrd"

[ -f "$IMAGE" ] || { echo "Image not found: $IMAGE" >&2; exit 1; }
[ -f "$MASK"  ] || { echo "Mask not found: $MASK"   >&2; exit 1; }

# ─── 1. Manifest ─────────────────────────────────────────────────────────────
cat > "$OUT/manifest.csv" <<EOF
patient_id,modality,image_path,mask_path
$SUBSET,CT,$IMAGE,$MASK
EOF

# ─── 2. Extract ──────────────────────────────────────────────────────────────
qr extract -m "$OUT/manifest.csv" -p ct-default -o "$OUT/features.csv"

echo ""
echo "Done. Outputs in $OUT/:"
echo "  features.csv          IBSI digital phantom features"
echo ""
echo "Compare against the IBSI reference values in:"
echo "  $IBSI_ROOT/${SUBSET}/reference/"
