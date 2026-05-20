#!/usr/bin/env bash
# One-command deploy + run for the lung1 pipeline.
# Edit clinical/clinical.csv with your real outcome file before running.
set -euo pipefail
cd "$(dirname "$0")"

pip install --upgrade --quiet qradiomics[rtstruct,prefect]

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
