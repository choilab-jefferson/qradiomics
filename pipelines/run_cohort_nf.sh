#!/usr/bin/env bash
# =============================================================================
# run_cohort_nf.sh — Nextflow-driven full-cohort pipeline
# =============================================================================
# Drop-in replacement for run_cohort.sh when you want Nextflow to own the
# parallelism. Stages 0–3 (catalog → fetch → series → patient) stay in
# bash (they're already parallel via GNU parallel + qr tcia download -j).
# Stage 4 (features) and 5–6 are dispatched as Nextflow processes so each
# patient gets its own work directory, cache, and resume.
#
# Why Nextflow for stage 4? PyRadiomics extraction is CPU-bound (~10-30s
# per patient with the nsclc-survival pattern). Running it as one
# multi-process pool fan-outs cleanly across cores; Nextflow adds cache,
# resume on failure, and HPC scheduler support.
#
# Usage:
#   ./run_cohort_nf.sh <collection> <roi> <pattern> [task] [outcome]
#
# Environment (in addition to run_cohort.sh's):
#   NF_PROFILE   — Nextflow profile (default local; slurm | docker | k8s)
#   NF_WORK      — Nextflow work directory (default $OUT/.nf-work)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# Reuse the bash pipeline for stages 0–3 (catalog/fetch/series/patient).
# Then take over with Nextflow for stages 4–6.
STAGES_BASH="${STAGES_BASH:-catalog,fetch,series,patient}"
STAGES="$STAGES_BASH" ./run_cohort.sh "$@" || exit $?

COLLECTION="$1"; ROI="$2"; PATTERN="$3"
TASK="${4:-survival}"; OUTCOME="${5:-OS_event}"
OUT="${OUT:-runs/$COLLECTION}"
CLINICAL="${CLINICAL:-clinical/${COLLECTION}.csv}"
NF_PROFILE="${NF_PROFILE:-local}"
NF_WORK="${NF_WORK:-$OUT/.nf-work}"

if ! command -v nextflow >/dev/null 2>&1; then
    echo ""
    echo "Nextflow not on PATH. Install:"
    echo "  curl -fsSL https://get.nextflow.io | bash"
    echo "  sudo mv nextflow /usr/local/bin/"
    exit 1
fi

# shellcheck source=./_qr_resolve.sh
source ./_qr_resolve.sh

MANIFEST="$OUT/manifest.csv"
[ -f "$MANIFEST" ] || { echo "$MANIFEST missing — bash stages 0–3 did not produce a manifest"; exit 1; }

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Nextflow stages: features → merge → modeling"
echo "  Profile        : $NF_PROFILE"
echo "  Work dir       : $NF_WORK"
echo "  Manifest       : $MANIFEST"
echo "════════════════════════════════════════════════════════════════════"

# Build a minimal qradiomics-features.nf adjacent to the manifest.
# Each patient row → one extract task. Final task collects per-patient
# CSVs into features.csv, then merges + trains.
NF_FILE="$OUT/qradiomics-features.nf"
cat > "$NF_FILE" <<EOF
nextflow.enable.dsl = 2

params.manifest  = "$MANIFEST"
params.pattern   = "$PATTERN"
params.out       = "$OUT"
params.clinical  = "$CLINICAL"
params.task      = "$TASK"
params.outcome   = "$OUTCOME"

// QR command resolution (matches _qr_resolve.sh)
QR = "${QR}"

process extract_one {
    tag "\$pid"
    publishDir "\${params.out}/.nf-features", mode: 'copy', overwrite: true
    cpus 1

    input:
    tuple val(pid), val(image), val(mask)

    output:
    path "\${pid}.features.csv"

    script:
    """
    cat > one.csv <<MAN
patient_id,modality,image_path,mask_path
\${pid},CT,\${image},\${mask}
MAN
    ${QR} extract -m one.csv -p \${params.pattern} -o \${pid}.features.csv --jobs 1
    """
}

process gather {
    publishDir params.out, mode: 'copy', overwrite: true
    input:
    path csvs

    output:
    path "features.csv"

    script:
    """
    files=( ${csvs} )
    head -1 \\\${files[0]} > features.csv
    for f in "\\\${files[@]}"; do
        tail -n +2 "\\\$f" >> features.csv
    done
    """
}

process merge_clinical {
    publishDir params.out, mode: 'copy', overwrite: true
    input:
    path features

    output:
    path "analysis_ready.csv"

    script:
    if (file(params.clinical).exists())
        """
        ${QR} results merge -f ${features} -c ${params.clinical} \\
            --clinical-id-col patient_id --time-col OS_days --event-col OS_event \\
            -o analysis_ready.csv
        """
    else
        """
        cp ${features} analysis_ready.csv
        """
}

process train_and_evaluate {
    publishDir params.out, mode: 'copy', overwrite: true
    input:
    path ar

    output:
    path "model.pkl"
    path "cv_metrics.json"
    path "evaluation.json"

    script:
    def time_opt = params.task == "survival" ? "--time-col OS_months" : ""
    """
    ${QR} ml train -i ${ar} --task ${params.task} --outcome ${params.outcome} ${time_opt} \\
        --model model.pkl --metrics cv_metrics.json
    ${QR} ml evaluate -i ${ar} --model model.pkl \\
        --task ${params.task} --outcome ${params.outcome} ${time_opt} \\
        --report evaluation.json
    """
}

workflow {
    rows = Channel
        .fromPath(params.manifest)
        .splitCsv(header: true)
        .map { row -> tuple(row.patient_id, row.image_path, row.mask_path) }

    extract_one(rows) | collect | gather | merge_clinical | train_and_evaluate
}
EOF

cd "$OUT"
nextflow run "$(basename "$NF_FILE")" -profile "$NF_PROFILE" -work-dir "$NF_WORK" -resume
