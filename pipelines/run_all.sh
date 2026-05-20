#!/usr/bin/env bash
# =============================================================================
# run_all.sh — full-scale (no patient limit) processing of every cohort
# =============================================================================
# Production sibling of test_all.sh — same cohort table, but processes
# the entire collection per cohort (MAX_PATIENTS=0). Stages remain
# idempotent: re-runs only redo what's missing.
#
# Usage:
#   ./run_all.sh                       # all cohorts, full scale
#   ./run_all.sh lung1 acrin_heart     # specific subset
#
# Environment:
#   OUT_ROOT     — output root (default: runs/)
#   N_PARALLEL   — workers per cohort (default: nproc / 2)
#   COHORT_PAR   — cohorts in parallel (default: 1; needs GNU parallel)
#   STAGES       — limit to a subset of stages (0..6); default: all
# =============================================================================
set -euo pipefail
exec env MAX_PATIENTS=0 ./test_all.sh "$@"
