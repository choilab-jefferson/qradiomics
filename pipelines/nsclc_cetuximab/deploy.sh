#!/usr/bin/env bash
# One-command deploy + run for the nsclc_cetuximab pipeline.
# Edit clinical/clinical.csv with your real outcome file before running.
set -euo pipefail
cd "$(dirname "$0")"

# Canonical lab tracking endpoints (MLflow, Prefect). Caller can override.
# shellcheck source=../_env.sh
. ../_env.sh

# Best-effort install: try full extras (dev), fall back to minimal (public).
pip install --upgrade --quiet 'qradiomics[rtstruct,prefect,tracking]' 2>/dev/null \
    || pip install --upgrade --quiet 'qradiomics[rtstruct,prefect]' 2>/dev/null \
    || pip install --upgrade --quiet 'qradiomics[rtstruct]' 2>/dev/null \
    || pip install --upgrade --quiet 'qradiomics'

EXECUTOR="${EXECUTOR:-nextflow}"

if [ "$EXECUTOR" = "nextflow" ]; then
    if ! command -v nextflow >/dev/null 2>&1; then
        echo "Fetching Nextflow ..."
        curl -fsSL https://get.nextflow.io | bash
        export PATH="$PWD:$PATH"
    fi
    nextflow run main.nf -profile local -resume
elif [ "$EXECUTOR" = "prefect" ]; then
    python prefect_flow.py
else
    qr workflow run plan.json --executor "$EXECUTOR"
fi
