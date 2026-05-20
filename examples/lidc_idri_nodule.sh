#!/usr/bin/env bash
# =============================================================================
# LIDC-IDRI — lung nodule radiomics + AHSN shape descriptor
# =============================================================================
# Cohort  : https://www.cancerimagingarchive.net/collection/lidc-idri/
# Pipeline: Per-patient DICOM CT series → qr convert dicom-series → NRRD.
#           LIDC-IDRI ships nodule annotations as per-reader XML (not
#           RTSTRUCT). Producing a single binary label NRRD requires
#           per-patient XML parsing; this script expects a label NRRD
#           per patient under <LABEL_DIR>/<patient_id>_nodule-label.nrrd
#           (any tool that voxelises the XML annotations works — for
#           example pylidc or a STAPLE consensus of the four readers).
#
#           features come from `ct-default` (1409 features per nodule)
#           plus, optionally, the AHSN shape descriptor (180-dim,
#           qradiomics.shape, 2014 CMPB) as a sidecar feature column.
# =============================================================================
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT to the LIDC-IDRI DICOM root}"
LABEL_DIR="${LABEL_DIR:?set LABEL_DIR to the directory holding <patient_id>_nodule-label.nrrd}"
CLINICAL_CSV="${CLINICAL_CSV:-}"  # optional — e.g. malignancy score CSV
OUT="${OUT:-./runs/lidc}"

NRRD_DIR="$OUT/nrrd"
mkdir -p "$NRRD_DIR"

# ─── 1. DICOM → NRRD per patient ─────────────────────────────────────────────
for PAT_DIR in "$DATASET_ROOT"/LIDC-IDRI-*/; do
    PAT=$(basename "$PAT_DIR")
    # LIDC nests CT under <patient>/<study>/<series>/
    SERIES_DIR=$(find "$PAT_DIR" -maxdepth 3 -type d -name '0.000000-*' \
                 -o -maxdepth 3 -type d ! -name '.*' | head -1)
    [ -z "$SERIES_DIR" ] && continue

    OUT_CT="$NRRD_DIR/${PAT}_CT.nrrd"
    [ -f "$OUT_CT" ] || qr convert dicom-series -i "$SERIES_DIR" -o "$OUT_CT"
done

# ─── 2. Manifest pairing CT NRRD ↔ pre-built nodule label NRRD ───────────────
{
    echo "patient_id,modality,image_path,mask_path"
    for ct in "$NRRD_DIR"/*_CT.nrrd; do
        pat=$(basename "$ct" _CT.nrrd)
        mask="$LABEL_DIR/${pat}_nodule-label.nrrd"
        [ -f "$mask" ] && echo "$pat,CT,$ct,$mask"
    done
} > "$OUT/manifest.csv"

# ─── 3. PyRadiomics features ─────────────────────────────────────────────────
qr extract -m "$OUT/manifest.csv" -p ct-default -o "$OUT/features.csv"

# ─── 4. (Optional) AHSN 180-dim shape descriptor ─────────────────────────────
# Available in qradiomics.shape (CMPB 2014). Uncomment to compute alongside
# the PyRadiomics features. The AHSN block is a 33³ ROI cropped around the
# nodule centroid; see qradiomics/shape/ for the API.
#
#   python - <<'PY'
#   import csv, numpy as np, SimpleITK as sitk
#   from qradiomics.shape import ahsn, AHSNConfig
#   cfg = AHSNConfig()
#   rows = []
#   with open("$OUT/manifest.csv") as f:
#       for r in csv.DictReader(f):
#           img = sitk.GetArrayFromImage(sitk.ReadImage(r["image_path"]))
#           msk = sitk.GetArrayFromImage(sitk.ReadImage(r["mask_path"]))
#           # crop a 33³ block around the mask centroid (omitted)
#           # desc = ahsn(block, cfg=cfg)
#           rows.append({"patient_id": r["patient_id"], "ahsn_dim": cfg.dim})
#   with open("$OUT/ahsn.csv", "w", newline="") as f:
#       w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
#   PY

# ─── 5. (Optional) classification if clinical CSV is supplied ────────────────
if [ -n "$CLINICAL_CSV" ]; then
    qr results merge \
      -f "$OUT/features.csv" -c "$CLINICAL_CSV" \
      --clinical-id-col patient_id \
      --time-col OS_days --event-col malignancy \
      -o "$OUT/analysis_ready.csv" || true

    qr analyze classify \
      -i "$OUT/analysis_ready.csv" --outcome malignancy \
      -o "$OUT/classify_results.csv" --top-n 20
fi

echo ""
echo "Done. Outputs in $OUT/:"
echo "  nrrd/                 *_CT.nrrd"
echo "  manifest.csv          patient_id + image + mask"
echo "  features.csv          patients × ~1409 features"
echo "  classify_results.csv  (if CLINICAL_CSV was set)"
