#!/usr/bin/env bash
# =============================================================================
# Common post-preprocessing workflow.
#
# Input contract (per-cohort root directory):
#     $OUT/manifest.csv           patient_id,image_path,mask_path
#     $OUT/clinical/clinical.csv  patient_id, ... outcome columns ...   (optional)
#
# Stages (skip individual ones via STAGES=features,modeling etc.):
#     register  — qr register     → $OUT/register/*.nrrd  (Scenario C only)
#     hu_correct — qr hu-correct  → $OUT/hu_corrected/*.nrrd  (CBCT only)
#     features  — qr extract     → $OUT/features.csv
#     delta     — qr delta        → $OUT/delta_features.csv  (longitudinal only)
#     merge     — qr results merge → $OUT/analysis_ready.csv
#     modeling  — qr ml train + evaluate → $OUT/{model.pkl,cv_metrics.json,evaluation.json}
#
# Usage:
#     run_workflow.sh <OUT> <TASK> <OUTCOME> \
#         [--time-col COL] [--event-col COL] [--pattern PAT] [--jobs N] \
#         [--stages a,b,c]
#
#     TASK = survival | classify | features-only
#
# Idempotent: re-running with the same arguments skips already-completed stages.
# =============================================================================
set -uo pipefail

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^#//'
    exit "${1:-1}"
}

[ $# -lt 3 ] && usage 1
OUT="$1"; TASK="$2"; OUTCOME="$3"
shift 3

TIME_COL=""
EVENT_COL=""
PATTERN="ct-default"
N_PARALLEL="${N_PARALLEL:-$(($(nproc) / 2))}"
STAGES="features,merge,modeling"
DELTA_PAIRS=""           # comma-separated 'name=A-B' specs
TREND="0"                # 1 to enable trend column emission

while [ $# -gt 0 ]; do
    case "$1" in
        --time-col)    TIME_COL="$2"; shift 2 ;;
        --event-col)   EVENT_COL="$2"; shift 2 ;;
        --pattern)     PATTERN="$2"; shift 2 ;;
        --jobs)        N_PARALLEL="$2"; shift 2 ;;
        --stages)      STAGES="$2"; shift 2 ;;
        --delta-pair)  DELTA_PAIRS="$DELTA_PAIRS,$2"; shift 2 ;;
        --with-trend)  TREND="1"; shift ;;
        -h|--help)     usage 0 ;;
        *) echo "✘ unknown arg: $1"; exit 1 ;;
    esac
done

# ─── locate qr ──────────────────────────────────────────────────────────────
WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$WORKFLOW_DIR/../_qr_resolve.sh"

stage_on() {
    [[ ",$STAGES," == *",$1,"* ]]
}

MANIFEST="$OUT/manifest.csv"
# manifest is required for any image-touching stage (register, hu_correct,
# preprocess, features). Pure CSV stages (delta, merge, modeling) can run
# from features.csv alone — detect that case so users can ship CSV-only
# transforms without a stub manifest.
needs_manifest=0
for s in register hu_correct preprocess features; do
    case ",$STAGES," in *",$s,"*) needs_manifest=1; break;; esac
done
N=0
if [ -f "$MANIFEST" ]; then
    N=$(($(wc -l < "$MANIFEST") - 1))
elif [ "$needs_manifest" = "1" ]; then
    echo "✘ missing manifest: $MANIFEST (required for stages: register, hu_correct, preprocess, features)"
    exit 2
fi
if [ "$needs_manifest" = "1" ] && [ "$N" -lt 1 ]; then
    echo "✘ empty manifest at $MANIFEST (no image/mask pairs found for this variant)"
    exit 2
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  common workflow"
echo "  OUT     : $OUT"
echo "  task    : $TASK   outcome: $OUTCOME"
echo "  pattern : $PATTERN    jobs: $N_PARALLEL"
echo "  stages  : $STAGES"
if [ "$needs_manifest" = "1" ]; then
    echo "  manifest: $N patients"
fi
echo "════════════════════════════════════════════════════════════════════"

# ─── register (Scenario C) ─────────────────────────────────────────────────
# Driven by a register_plan.csv ($OUT/register_plan.csv) with columns
#   patient_id,fixed,moving[,mask,output_image,output_mask,output_transform]
# Skipped silently when the plan file is absent.
REGISTER_PLAN="$OUT/register_plan.csv"
if stage_on register; then
    if [ -f "$REGISTER_PLAN" ]; then
        echo "[register] running plan $REGISTER_PLAN"
        python3 - "$REGISTER_PLAN" <<'PY'
import csv, subprocess, sys
plan = sys.argv[1]
with open(plan) as f:
    for i, row in enumerate(csv.DictReader(f), 1):
        if not row.get("output_image"):
            continue
        cmd = ["qr", "register",
               "--fixed", row["fixed"], "--moving", row["moving"],
               "--output-image", row["output_image"]]
        if row.get("mask"):
            cmd += ["--mask", row["mask"]]
            if row.get("output_mask"):
                cmd += ["--output-mask", row["output_mask"]]
        if row.get("output_transform"):
            cmd += ["--output-transform", row["output_transform"]]
        print(f"  [register {i}] {row.get('patient_id','?')}", flush=True)
        subprocess.run(cmd, check=True)
PY
    else
        echo "[register] no $REGISTER_PLAN — skipping"
    fi
fi

# ─── hu_correct (CBCT scatter compensation) ─────────────────────────────────
HU_PLAN="$OUT/hu_plan.csv"
if stage_on hu_correct; then
    if [ -f "$HU_PLAN" ]; then
        echo "[hu_correct] running plan $HU_PLAN"
        python3 - "$HU_PLAN" <<'PY'
import csv, subprocess, sys
plan = sys.argv[1]
with open(plan) as f:
    for i, row in enumerate(csv.DictReader(f), 1):
        cmd = ["qr", "hu-correct",
               "-i", row["input"], "-r", row["reference"], "-o", row["output"]]
        print(f"  [hu_correct {i}] {row.get('patient_id','?')}", flush=True)
        subprocess.run(cmd, check=True)
PY
    else
        echo "[hu_correct] no $HU_PLAN — skipping"
    fi
fi

# ─── features ───────────────────────────────────────────────────────────────
FEATURES="$OUT/features.csv"
if stage_on features; then
    if [ -f "$FEATURES" ] && [ "$(($(wc -l < "$FEATURES") - 1))" -ge "$N" ]; then
        echo "[features] cached → $FEATURES"
    else
        echo "[features] qr extract (jobs=$N_PARALLEL)"
        qr extract -m "$MANIFEST" -p "$PATTERN" -o "$FEATURES" --jobs "$N_PARALLEL"
    fi
fi

# ─── delta + trend (longitudinal only) ──────────────────────────────────────
DELTA_FEATURES="$OUT/delta_features.csv"
if stage_on delta; then
    # Only run if we actually have delta pairs or trend requested.
    if [ -n "$DELTA_PAIRS" ] || [ "$TREND" = "1" ]; then
        # Need a timepoint column in features.csv. The current `qr extract`
        # does NOT propagate manifest's `timepoint` column into features.csv,
        # so users must merge it in beforehand (or this stage will UsageError).
        cmd=(qr delta -f "$FEATURES" -o "$DELTA_FEATURES")
        IFS=',' read -ra pairs <<< "${DELTA_PAIRS#,}"
        for p in "${pairs[@]}"; do
            [ -n "$p" ] && cmd+=(--pair "$p")
        done
        [ "$TREND" = "1" ] && cmd+=(--with-trend)
        echo "[delta] ${cmd[*]}"
        "${cmd[@]}"
    else
        echo "[delta] no --delta-pair / --with-trend supplied — skipping"
    fi
fi

# features-only short-circuit
if [ "$TASK" = "features-only" ]; then
    echo "✓ features-only mode — done"
    exit 0
fi

# ─── merge ──────────────────────────────────────────────────────────────────
ANALYSIS_READY="$OUT/analysis_ready.csv"
CLINICAL="$OUT/clinical/clinical.csv"
if stage_on merge; then
    if [ ! -f "$CLINICAL" ]; then
        echo "✘ no clinical CSV at $CLINICAL — cannot run merge/modeling"
        exit 3
    fi
    MERGE_OPTS=()
    [ -n "$TIME_COL" ]  && MERGE_OPTS+=(--time-col  "$TIME_COL")
    [ -n "$EVENT_COL" ] && MERGE_OPTS+=(--event-col "$EVENT_COL")
    echo "[merge] qr results merge"
    qr results merge \
        -f "$FEATURES" -c "$CLINICAL" \
        --clinical-id-col patient_id \
        "${MERGE_OPTS[@]}" \
        -o "$ANALYSIS_READY"
fi

# ─── modeling ───────────────────────────────────────────────────────────────
if stage_on modeling; then
    [ -f "$ANALYSIS_READY" ] || { echo "✘ no $ANALYSIS_READY — merge failed"; exit 4; }
    TIME_OPT=()
    if [ "$TASK" = "survival" ] && [ -n "$TIME_COL" ]; then
        # qr ml train uses --time-col only for survival; lifelines wants months.
        # qr results merge already converted OS_days→OS_months, so pass that.
        TIME_OPT+=(--time-col OS_months)
    fi
    echo "[modeling] qr ml train  (task=$TASK, outcome=$OUTCOME)"
    qr ml train -i "$ANALYSIS_READY" \
        --task "$TASK" --outcome "$OUTCOME" "${TIME_OPT[@]}" \
        --model "$OUT/model.pkl" --metrics "$OUT/cv_metrics.json"
    echo "[modeling] qr ml evaluate"
    qr ml evaluate -i "$ANALYSIS_READY" \
        --model "$OUT/model.pkl" \
        --task "$TASK" --outcome "$OUTCOME" "${TIME_OPT[@]}" \
        --report "$OUT/evaluation.json"
fi

echo "✓ done: $OUT"
