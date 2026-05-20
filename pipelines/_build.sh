#!/usr/bin/env bash
# Regenerate every pipeline bundle (plan.json + main.nf + prefect_flow.py)
# from a single qr workflow plan invocation per cohort.
#
# Run from the qradiomics-public repo root:
#   pip install -e .
#   bash pipelines/_build.sh
#
# This is idempotent — re-running overwrites the generated files.
set -euo pipefail
cd "$(dirname "$0")"

QR="${QR:-qr}"
ROOT=$(pwd)

build_one() {
    local DIR="$1"; shift
    local TEMPLATE="$1"; shift
    local OUTDIR="runs/${DIR}"

    echo "== $DIR =="
    mkdir -p "$DIR/clinical"
    [ -f "$DIR/clinical/clinical.csv" ] || \
        echo "patient_id,OS_days,OS_event,age,sex,stage" > "$DIR/clinical/clinical.csv"

    "$QR" workflow plan -t "$TEMPLATE" \
        --clinical "$DIR/clinical/clinical.csv" \
        --outdir "$OUTDIR" \
        --output "$DIR/plan.json" \
        "$@"

    "$QR" workflow scaffold -p "$DIR/plan.json" -e nextflow -o "$DIR/main.nf"
    "$QR" workflow scaffold -p "$DIR/plan.json" -e prefect  -o "$DIR/prefect_flow.py"
    cp ../examples/nextflow/nextflow.config "$DIR/nextflow.config"
}

build_one lung1 tcia_to_ml \
    --collection NSCLC-Radiomics --roi GTV-1 \
    --pattern nsclc-survival --task survival

build_one nsclc_cetuximab tcia_to_ml \
    --collection NSCLC-Cetuximab --roi PTV \
    --pattern nsclc-survival --task survival

build_one lidc_idri tcia_to_ml \
    --collection LIDC-IDRI --roi Nodule \
    --pattern ct-default --task classify --outcome malignancy

build_one acrin_heart tcia_to_ml \
    --collection ACRIN-NSCLC-FDG-PET --roi Heart \
    --pattern ct-default --task classify --outcome fdg_uptake_binary

echo ""
echo "Built bundles:"
for d in lung1 nsclc_cetuximab lidc_idri acrin_heart; do
    echo "  $d/{plan.json,main.nf,prefect_flow.py,nextflow.config}"
done
