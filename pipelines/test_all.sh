#!/usr/bin/env bash
# =============================================================================
# test_all.sh — run every configured TCIA cohort end-to-end
# =============================================================================
# Default mode is "smoke" (MAX_PATIENTS=10). For full-cohort production
# runs, use ./run_all.sh (which is just this script with MAX_PATIENTS=0).
#
# Output:
#   $OUT_ROOT/<bundle>/...     per-cohort tree (idempotent, cached)
#   $OUT_ROOT/<bundle>.log     full log (auto-tailed live to terminal)
#   $OUT_ROOT/report.md        cross-cohort summary
#
# Usage:
#   ./test_all.sh                       # smoke, all cohorts (10 patients each)
#   MAX_PATIENTS=0 ./test_all.sh        # full cohort, all collections
#   ./test_all.sh lung1 acrin_heart     # specific subset
#
# Environment:
#   OUT_ROOT     — output root (default: runs/)
#   N_PARALLEL   — workers per cohort (default: nproc / 2)
#   MAX_PATIENTS — limit per cohort (0 = unlimited)
#   COHORT_PAR   — cohorts in parallel (default: 1)
#   STAGES       — stages to run, comma-separated (default: 0,1,2,3,4,5,6)
#   QUIET        — set to 1 to suppress live log streaming
#   CONTINUE_ON_FAIL — set to 0 to abort after the first cohort failure
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

OUT_ROOT="${OUT_ROOT:-runs}"
N_PARALLEL="${N_PARALLEL:-$(($(nproc) / 2))}"
MAX_PATIENTS="${MAX_PATIENTS:-10}"
COHORT_PAR="${COHORT_PAR:-1}"
# Empty = let run_cohort.sh fill in the canonical full sequence (currently
# catalog/fetch/series/patient/preprocess/features/shape/merge/modeling;
# anonymize is opt-in). Override e.g. STAGES=features,merge,modeling to
# rebuild only the modeling tail.
STAGES="${STAGES:-}"
QUIET="${QUIET:-0}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"

# ─── Cohort table ────────────────────────────────────────────────────────────
# bundle           collection           roi    pattern         task     outcome
declare -a COHORTS=(
    "lung1            NSCLC-Radiomics      GTV-1   nsclc-survival   survival  OS_event"
    # nsclc_cetuximab excluded: TCIA no longer hosts a "NSCLC-Cetuximab"
    #   collection; the curl /getCollectionValues endpoint returns no match.
    # acrin_heart excluded: ACRIN-NSCLC-FDG-PET on TCIA has 0 RTSTRUCT / SEG
    #   objects, so no Heart mask can be derived. Use pipelines/acrin_local
    #   (pre-converted NRRDs) for the ACRIN cohort instead.
    # LIDC-IDRI excluded: XML-only annotations need pylidc preprocessing.
)

# ─── Filter by argv subset if provided ───────────────────────────────────────
if [ $# -gt 0 ]; then
    SELECTED=()
    for arg in "$@"; do
        for row in "${COHORTS[@]}"; do
            name=$(echo "$row" | awk '{print $1}')
            [ "$name" = "$arg" ] && SELECTED+=("$row")
        done
    done
    COHORTS=("${SELECTED[@]}")
fi

START=$(date +%s)
REPORT="$OUT_ROOT/report.md"
mkdir -p "$OUT_ROOT"

# ─── Header ──────────────────────────────────────────────────────────────────
N_COHORTS=${#COHORTS[@]}
echo "════════════════════════════════════════════════════════════════════"
echo "  qradiomics test_all"
echo "  cohorts      : $N_COHORTS  ($(printf '%s ' "${COHORTS[@]%% *}"))"
echo "  MAX_PATIENTS : $MAX_PATIENTS  ($([ "$MAX_PATIENTS" = "0" ] && echo unlimited || echo "limit"))"
echo "  N_PARALLEL   : $N_PARALLEL (per cohort)"
echo "  COHORT_PAR   : $COHORT_PAR  (cohorts in parallel)"
echo "  STAGES       : ${STAGES:-(all)}"
echo "  output       : $OUT_ROOT"
echo ""
echo "  Live status:  python3 pipelines/monitor.py $OUT_ROOT"
echo "════════════════════════════════════════════════════════════════════"

# ─── Per-cohort runner ───────────────────────────────────────────────────────
run_one() {
    local row="$1"
    read -r bundle collection roi pattern task outcome <<<"$row"
    # Use a timestamped log file per invocation so an orphan `tail -f` left
    # behind by an earlier interrupted run never re-emits new content.
    local TS
    TS=$(date +%Y%m%d-%H%M%S)
    local LOG="$OUT_ROOT/$bundle.$TS.log"
    # Maintain a stable symlink for monitors (run_cohort.log → latest).
    ln -sfn "$(basename "$LOG")" "$OUT_ROOT/$bundle.log"

    echo ""
    echo "►►► $bundle  ($collection) → $LOG"

    : > "$LOG"
    local rc=0

    # Use tee to write to LOG and stream to terminal in one shot. No
    # background tail process → no orphan risk. PIPESTATUS captures the
    # actual run_cohort.sh exit code from the pipeline head.
    if [ "$QUIET" = "0" ]; then
        OUT="$OUT_ROOT/$bundle" \
        CLINICAL="$bundle/clinical/clinical.csv" \
        N_PARALLEL="$N_PARALLEL" \
        MAX_PATIENTS="$MAX_PATIENTS" \
        STAGES="$STAGES" \
        ./run_cohort.sh "$collection" "$roi" "$pattern" "$task" "$outcome" 2>&1 \
            | stdbuf -oL grep -vE "ConvergenceWarning|RuntimeWarning|frozen runpy|^GLCM|symmetrical|warnings|^INFO:|Warning|InsecureRequestWarning|^$" \
            | stdbuf -oL sed "s/^/[$bundle] /" \
            | tee "$LOG"
        rc=${PIPESTATUS[0]}
    else
        OUT="$OUT_ROOT/$bundle" \
        CLINICAL="$bundle/clinical/clinical.csv" \
        N_PARALLEL="$N_PARALLEL" \
        MAX_PATIENTS="$MAX_PATIENTS" \
        STAGES="$STAGES" \
        ./run_cohort.sh "$collection" "$roi" "$pattern" "$task" "$outcome" \
            >"$LOG" 2>&1 || rc=$?
    fi

    if [ "$rc" -eq 0 ]; then
        echo "  ✓ $bundle done  ($(date -d@$(( $(date +%s) - START )) -u +%H:%M:%S))"
    else
        echo "  ✗ $bundle FAILED (exit $rc)  — see $LOG"
        [ "$CONTINUE_ON_FAIL" = "0" ] && exit "$rc"
    fi
    return 0
}
export -f run_one
export OUT_ROOT N_PARALLEL MAX_PATIENTS STAGES QUIET START CONTINUE_ON_FAIL

# ─── Dispatch ────────────────────────────────────────────────────────────────
if [ "$COHORT_PAR" -gt 1 ] && command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${COHORTS[@]}" | parallel -j "$COHORT_PAR" run_one
else
    for row in "${COHORTS[@]}"; do
        run_one "$row" || true
    done
fi

ELAPSED=$(( $(date +%s) - START ))

# ─── Aggregate Markdown report ───────────────────────────────────────────────
COHORT_ROWS_FILE="$OUT_ROOT/.cohorts.txt"
printf '%s\n' "${COHORTS[@]}" > "$COHORT_ROWS_FILE"

python3 - "$OUT_ROOT" "$COHORT_ROWS_FILE" "$REPORT" "$ELAPSED" "$MAX_PATIENTS" <<'PY'
import csv, json, sys
from pathlib import Path

out_root = Path(sys.argv[1])
cohorts = [line.strip() for line in open(sys.argv[2]) if line.strip()]
report_path = sys.argv[3]
elapsed = int(sys.argv[4])
max_patients = sys.argv[5]

rows = []
for row in cohorts:
    parts = row.split()
    bundle, collection, roi, pattern, task, outcome = parts[:6]
    out = out_root / bundle
    n = 0
    if (out / "manifest.csv").exists():
        with open(out / "manifest.csv") as f:
            n = max(0, sum(1 for _ in f) - 1)
    feat_count = 0
    if (out / "features.csv").exists():
        with open(out / "features.csv") as f:
            feat_count = max(0, len(f.readline().split(",")) - 1)
    cv_str, eval_str, note = "—", "—", ""
    mp = out / "cv_metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        if m.get("task") == "survival" and m.get("cv_c_index_mean") is not None:
            cv_str = f"c-index = {m['cv_c_index_mean']:.3f} ± {m['cv_c_index_std']:.3f}"
        elif m.get("task") == "classify" and m.get("cv_auc_mean") is not None:
            cv_str = f"AUC = {m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f}"
        if m.get("note"):
            note = m["note"]
    ep = out / "evaluation.json"
    if ep.exists():
        e = json.loads(ep.read_text())
        if e.get("task") == "survival":
            eval_str = f"c-index = {e.get('c_index', float('nan')):.3f}"
        else:
            eval_str = f"AUC = {e.get('auc', float('nan')):.3f}"
    rows.append({
        "bundle": bundle, "collection": collection, "task": task,
        "n": n, "features": feat_count,
        "cv": cv_str, "eval": eval_str, "note": note,
    })

lines = [
    "# qradiomics test_all — summary", "",
    f"- elapsed: {elapsed}s",
    f"- MAX_PATIENTS: {max_patients}",
    "",
    "| bundle | collection | n | features | task | CV | held-out | note |",
    "|---|---|--:|--:|---|---|---|---|",
]
for r in rows:
    lines.append(
        f"| {r['bundle']} | {r['collection']} | {r['n']} | {r['features']} | "
        f"{r['task']} | {r['cv']} | {r['eval']} | {r['note']} |"
    )
Path(report_path).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "elapsed: ${ELAPSED}s ($(date -d@"$ELAPSED" -u +%H:%M:%S))   report: $REPORT"
echo "════════════════════════════════════════════════════════════════════"
