#!/usr/bin/env bash
# =============================================================================
# run_cohort_nf.sh — Nextflow-driven full-cohort pipeline (TCIA collection)
# =============================================================================
# Thin wrapper over the generic Nextflow backend of `qr workflow run`:
#
#   qr workflow plan -t tcia_to_ml ...      → $OUT/plan.json
#   qr workflow run  $OUT/plan.json --executor nextflow
#
# The plan runs `tcia series/download/manifest` once (QR_PRE), then
# `convert from-manifest --case` + `extract --case` once per patient
# (QR_CASE, one Nextflow task each, cached for -resume), then
# `results merge` + `ml train/evaluate` once (QR_POST). No bash-side
# parallelism (GNU parallel / xargs) is involved any more.
#
# Usage:
#   ./run_cohort_nf.sh <collection> <roi> <pattern> [task] [outcome]
#
# Environment:
#   OUT          — output dir (default runs/<collection>)
#   CLINICAL     — clinical CSV (default clinical/<collection>.csv)
#   MAX_SERIES   — limit TCIA download (default 0 = all)
#   ENGINE       — extraction engine(s) (default pyradiomics)
#   NF_PROFILE   — Nextflow profile (default local; slurm)
#   NF_RUN_DIR   — Nextflow run dir (default $OUT/nf_run)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 3 ]; then
    echo "usage: $0 <collection> <roi> <pattern> [task] [outcome]" >&2
    exit 2
fi
COLLECTION="$1"; ROI="$2"; PATTERN="$3"
TASK="${4:-survival}"; OUTCOME="${5:-OS_event}"
OUT="${OUT:-runs/$COLLECTION}"
CLINICAL="${CLINICAL:-clinical/${COLLECTION}.csv}"
NF_PROFILE="${NF_PROFILE:-local}"
NF_RUN_DIR="${NF_RUN_DIR:-$OUT/nf_run}"

source ./_qr_resolve.sh

mkdir -p "$OUT"
qr workflow plan -t tcia_to_ml \
    --collection "$COLLECTION" --roi "$ROI" --pattern "$PATTERN" \
    --task "$TASK" --outcome "$OUTCOME" \
    --clinical "$CLINICAL" \
    --max-series "${MAX_SERIES:-0}" --engine "${ENGINE:-pyradiomics}" \
    --outdir "$OUT" --output "$OUT/plan.json"

qr workflow run "$OUT/plan.json" --executor nextflow \
    --profile "$NF_PROFILE" --run-dir "$NF_RUN_DIR"
