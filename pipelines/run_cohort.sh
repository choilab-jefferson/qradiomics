#!/usr/bin/env bash
# =============================================================================
# run_cohort.sh — full-cohort TCIA → ML pipeline with hierarchical caching
# =============================================================================
# Stages are first-class NAMED units; the numbers in headers are just the
# canonical position. New steps can be added by appending a name to
# PIPELINE_STAGES below and writing its block. STAGES env accepts names
# (preferred) or positions:
#
#   STAGES=catalog,fetch,series     # by name (preferred)
#   STAGES=0,1,2                     # by position (backward compat)
#   STAGES=features,merge,modeling   # rebuild just the modeling tail
#
# Canonical stage list (shipped):
#
#   catalog    tcia series  → <OUT>/{ct,rt}_series.csv
#   fetch      tcia download (CT + RT for the SAME patients)
#                          → <OUT>/dicom/<patient>/<study>/<series>/
#   series     DICOM → NRRD             (per-series, parallel, cached)
#                          → <OUT>/series/<patient>/<patient>__<UID>__CT.nrrd
#                          → <OUT>/series/<patient>/<patient>__<UID>__<ROI>-label.nrrd
#   patient    pick best CT + label  → <OUT>/patient/<patient>.json
#                          + <OUT>/manifest.csv
#   features   qr extract              → <OUT>/features.csv
#   merge      qr results merge        → <OUT>/analysis_ready.csv
#   modeling   qr ml train + evaluate  → <OUT>/{model.pkl,cv_metrics.json,evaluation.json}
#
# Every stage is idempotent: re-running re-uses outputs already on disk.
# Live status is written to <OUT>/.status/state.json (rendered by monitor.py).
#
# Usage:
#   ./run_cohort.sh <collection> <roi> <pattern> [task] [outcome]
#
# Env (all optional):
#   OUT          — output root (default: runs/<collection>)
#   CLINICAL     — clinical CSV path
#   N_PARALLEL   — series-conversion workers (default: $(nproc))
#   MAX_PATIENTS — limit to N patients (0 = unlimited)
#   STAGES       — stages to run (names or positions, default = all)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# Resolve qr (handles installed shim, dev fallback, sub-bash workers)
# shellcheck source=./_qr_resolve.sh
source ./_qr_resolve.sh

COLLECTION="${1:?collection name required (e.g. NSCLC-Radiomics)}"
ROI="${2:?ROI name required (e.g. GTV-1)}"
PATTERN="${3:?pattern id required (e.g. nsclc-survival)}"
TASK="${4:-survival}"
OUTCOME="${5:-OS_event}"

OUT="${OUT:-runs/$COLLECTION}"
CLINICAL="${CLINICAL:-clinical/${COLLECTION}.csv}"
N_PARALLEL="${N_PARALLEL:-$(nproc)}"
MAX_PATIENTS="${MAX_PATIENTS:-0}"
STAGES="${STAGES:-}"

mkdir -p "$OUT"/{dicom,series,patient,preprocess,.status}

# ─── Stages as first-class named units ───────────────────────────────────────
# This pipeline ships with nine canonical stages, but the architecture is
# name-based — STAGES can be a comma-separated list of names (or 0-based
# positions for backward compatibility). New steps can be added by simply
# appending a name to PIPELINE_STAGES and defining its block below.
#
#   catalog     tcia series catalog
#   fetch       tcia download
#   anonymize   strip PHI from DICOM (institutional cohorts; off by default
#               — TCIA data is already deidentified)
#   series      DICOM → NRRD per series
#   patient     manifest aggregation
#   preprocess  crop to ROI + (optional) isotropic resample
#   features    PyRadiomics intensity/texture features
#   shape       AHSN (2014 CMPB) + spiculation (2021 CMPB) descriptors
#   merge       join features with clinical CSV
#   modeling    qr ml train + evaluate
PIPELINE_STAGES=(catalog fetch anonymize series patient preprocess features shape merge modeling)
NUM_STAGES=${#PIPELINE_STAGES[@]}

# Normalise STAGES env. By default skip 'anonymize' (TCIA cohorts are already
# deidentified; institutional cohorts must opt in explicitly to avoid silently
# anonymising data in place).
if [ -z "${STAGES:-}" ]; then
    _default=()
    for s in "${PIPELINE_STAGES[@]}"; do
        [ "$s" = "anonymize" ] && continue
        _default+=("$s")
    done
    STAGES=$(IFS=,; echo "${_default[*]}")
fi

stage_on() {
    # Accepts a stage NAME (canonical) — `STAGES=catalog,fetch` works.
    # Also accepts the numeric index for backward compatibility — `STAGES=0,1` works.
    local want="$1"
    case ",$STAGES," in *",$want,"*) return 0;; esac
    # Try numeric lookup
    local i=0
    for s in "${PIPELINE_STAGES[@]}"; do
        if [ "$s" = "$want" ]; then
            case ",$STAGES," in *",$i,"*) return 0;; esac
        fi
        i=$((i + 1))
    done
    return 1
}

# Stage banner — displays "[N/T] name" using the canonical position
stage_banner() {
    local name="$1"; local i=0
    for s in "${PIPELINE_STAGES[@]}"; do
        if [ "$s" = "$name" ]; then
            echo ""
            echo "[$i/$NUM_STAGES] $name"
            return
        fi
        i=$((i + 1))
    done
    echo ""
    echo "[?/?] $name"
}

cache_say() { echo "    [cached] $*"; }

# ─── Stage state tracking (rendered by pipelines/monitor.py) ─────────────────
STATE_FILE="$OUT/.status/state.json"
_init_state() {
    python3 - "$STATE_FILE" "$COLLECTION" "$ROI" "${PIPELINE_STAGES[@]}" <<'PY'
import json, sys
sf, coll, roi = sys.argv[1], sys.argv[2], sys.argv[3]
stages = sys.argv[4:]
state = {
    "collection": coll, "roi": roi,
    "stages": {n: {"state": "pending"} for n in stages},
}
open(sf, "w").write(json.dumps(state, indent=2))
PY
}
_mark() {  # _mark <stage> <state>
    python3 - "$STATE_FILE" "$1" "$2" <<'PY'
import json, sys, time
sf, stage, st = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.load(open(sf))
entry = data.setdefault("stages", {}).setdefault(stage, {})
entry["state"] = st
if st == "running":
    entry["started_at"] = time.time()
elif st in ("done", "failed"):
    entry["finished_at"] = time.time()
open(sf, "w").write(json.dumps(data, indent=2))
PY
}
trap 'rc=$?; [ "$rc" -ne 0 ] && _mark "${CURRENT_STAGE:-unknown}" failed' EXIT
[ -f "$STATE_FILE" ] || _init_state

echo "════════════════════════════════════════════════════════════════════"
echo "  Collection : $COLLECTION"
echo "  ROI        : $ROI"
echo "  Pattern    : $PATTERN"
echo "  Task       : $TASK   (outcome=$OUTCOME)"
echo "  OUT        : $OUT"
echo "  Workers    : $N_PARALLEL"
echo "  Stages     : $STAGES"
echo "  Max patients: ${MAX_PATIENTS} (0 = unlimited)"
echo "  qr resolved: $QR"
echo "════════════════════════════════════════════════════════════════════"

# ============================================================================
# Stage 0 — catalog
# ============================================================================
CT_SERIES="$OUT/ct_series.csv"
RT_SERIES="$OUT/rt_series.csv"
_has_data() {
    # Catalog CSVs always have a header row. Treat them as cached only if
    # they hold at least one data row (i.e. wc -l ≥ 2).
    [ -f "$1" ] && [ "$(wc -l < "$1")" -ge 2 ]
}

if stage_on catalog; then
    echo ""
    stage_banner catalog
    CURRENT_STAGE=catalog; _mark catalog running
    if _has_data "$CT_SERIES"; then
        cache_say "$CT_SERIES ($(($(wc -l < "$CT_SERIES") - 1)) CT series)"
    else
        qr tcia series --collection "$COLLECTION" --modality CT --output "$CT_SERIES"
    fi
    if _has_data "$RT_SERIES"; then
        cache_say "$RT_SERIES ($(($(wc -l < "$RT_SERIES") - 1)) RT series)"
    else
        qr tcia series --collection "$COLLECTION" --modality RTSTRUCT --output "$RT_SERIES" || true
    fi
    _mark catalog done
fi

# Patient list (cohort intersection)
PATIENT_LIST="$OUT/patients.txt"
if stage_on catalog; then
    python3 - "$CT_SERIES" "$RT_SERIES" "$PATIENT_LIST" "$MAX_PATIENTS" <<'PY'
import csv, sys
ct, rt, out, maxp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def pids(p):
    try:
        with open(p) as f:
            return {r["PatientID"] for r in csv.DictReader(f)}
    except FileNotFoundError:
        return set()
ct_p, rt_p = pids(ct), pids(rt)
both = sorted(ct_p & rt_p) if rt_p else sorted(ct_p)
if maxp > 0: both = both[:maxp]
with open(out, "w") as f:
    f.write("\n".join(both) + "\n")
print(f"  selected {len(both)} patients with " + ("CT+RT" if rt_p else "CT only"))
PY
fi
NUM_PATIENTS=$(wc -l < "$PATIENT_LIST" 2>/dev/null || echo 0)

# ============================================================================
# Stage 1 — fetch (CT + RT for the same patients)
# ============================================================================
if stage_on fetch; then
    echo ""
    stage_banner fetch; echo "  ($NUM_PATIENTS patients)"
    CURRENT_STAGE=fetch; _mark fetch running
    python3 - "$CT_SERIES" "$RT_SERIES" "$PATIENT_LIST" <<'PY'
import csv, sys
ct, rt, pf = sys.argv[1], sys.argv[2], sys.argv[3]
keep = set(open(pf).read().split())
for src in (ct, rt):
    try:
        with open(src) as fr:
            r = csv.DictReader(fr); rows = [x for x in r if x["PatientID"] in keep]
        with open(src.replace(".csv","_selected.csv"), "w", newline="") as fw:
            w = csv.DictWriter(fw, fieldnames=r.fieldnames); w.writeheader(); w.writerows(rows)
    except FileNotFoundError:
        pass
PY
    # I/O-bound: many concurrent HTTP downloads. Cap at $N_PARALLEL but at
    # least 8 to keep TCIA pipelined.
    DL_WORKERS=$N_PARALLEL
    [ "$DL_WORKERS" -lt 8 ] && DL_WORKERS=8
    qr tcia download --manifest "${CT_SERIES%.csv}_selected.csv" \
        --output "$OUT/dicom" --workers "$DL_WORKERS"
    [ -f "${RT_SERIES%.csv}_selected.csv" ] && \
        qr tcia download --manifest "${RT_SERIES%.csv}_selected.csv" \
            --output "$OUT/dicom" --workers "$DL_WORKERS"
    _mark fetch done
fi

# ============================================================================
# Stage — anonymize (institutional cohorts only; OFF by default).
# Strips PHI tags from every DICOM under $OUT/dicom in place. Run BEFORE
# `series` so the NRRDs carry the anonymised metadata. Set
#     STAGES=catalog,fetch,anonymize,series,patient,preprocess,features,shape,merge,modeling
# or include `anonymize` in the STAGES list to enable.
# ============================================================================
ANON_FLAG_FILE="$OUT/.status/anonymized.json"
if stage_on anonymize; then
    echo ""
    stage_banner anonymize
    CURRENT_STAGE=anonymize; _mark anonymize running
    if [ -f "$ANON_FLAG_FILE" ]; then
        cache_say "$ANON_FLAG_FILE (already anonymised)"
    else
        qr anonymize -i "$OUT/dicom" \
            --replace-pid --pid-salt "${ANON_SALT:-qradiomics-$COLLECTION}" \
            --mapping "$OUT/pid_map.csv" \
            --jobs "$N_PARALLEL"
        echo "{\"anonymized_at\": $(date +%s), \"salt_label\": \"${ANON_SALT:-default}\"}" \
            > "$ANON_FLAG_FILE"
    fi
    _mark anonymize done
fi

# ============================================================================
# Stage — per-series DICOM → NRRD (parallel, idempotent per series UID)
# ============================================================================
if stage_on series; then
    echo ""
    stage_banner series; echo "  (DICOM → NRRD per series; $N_PARALLEL workers)"
    CURRENT_STAGE=series; _mark series running
    SERIES_WORK="$OUT/.series_dirs.txt"
    find "$OUT/dicom" -mindepth 3 -maxdepth 3 -type d > "$SERIES_WORK"

    # Per-series worker. Sources the qr resolver so sub-bash gets `qr` too.
    # RTSTRUCT pairing uses the structure-set's ReferencedSeriesInstanceUID
    # (DICOM-correct) rather than a directory-sibling heuristic, so cohorts
    # that store CT and RT in different studies (ACRIN-NSCLC-FDG-PET, etc.)
    # work without tweaks.
    PIPELINES_DIR="$(pwd)"
    PROC="$OUT/.proc_series.sh"
    ERR_LOG="$OUT/.series_errors.log"
    : > "$ERR_LOG"
    cat > "$PROC" <<EOF
#!/usr/bin/env bash
set -e
source "$PIPELINES_DIR/_qr_resolve.sh"

SERIES_DIR="\$1"
OUT_ROOT="$OUT"
ROI="$ROI"
ERR_LOG="$ERR_LOG"

PAT=\$(echo "\$SERIES_DIR" | awk -F/ '{print \$(NF-2)}')
SERIES_UID=\$(basename "\$SERIES_DIR")
mkdir -p "\$OUT_ROOT/series/\$PAT"
# Accept both .dcm-suffixed and extensionless DICOM (TCIA quirk). Exclude LICENSE files.
N_DCM=\$(find "\$SERIES_DIR" -maxdepth 1 -type f \\( -name '*.dcm' -o ! -name '*.*' \\) 2>/dev/null | grep -v -i "LICENSE" | wc -l)
[ "\$N_DCM" -eq 0 ] && N_DCM=\$(ls "\$SERIES_DIR"/*.dcm 2>/dev/null | wc -l)

_record_err() {
    # \$1=phase \$2=stderr
    printf '[%s] %s :: %s\\n%s\\n' "\$(date +%H:%M:%S)" "\$1" "\$SERIES_DIR" "\$2" >> "\$ERR_LOG"
}

if [ "\$N_DCM" -gt 5 ]; then
    # CT-like series → convert if not cached
    OUT_NRRD="\$OUT_ROOT/series/\$PAT/\${PAT}__\${SERIES_UID}__CT.nrrd"
    if [ ! -f "\$OUT_NRRD" ]; then
        if ! err=\$(qr convert dicom-series -i "\$SERIES_DIR" -o "\$OUT_NRRD" 2>&1); then
            _record_err "convert" "\$err"
        fi
    fi
elif [ "\$N_DCM" -ge 1 ]; then
    # RTSTRUCT (1 dcm) — locate the referenced CT series anywhere under the
    # patient subtree (not just the same study) by reading the
    # ReferencedSeriesInstanceUID from the structure set.
    RS=\$(find "\$SERIES_DIR" -maxdepth 1 -type f \\( -name '*.dcm' -o ! -name '*.*' \\) | grep -v -i "LICENSE" | head -1)
    [ -z "\$RS" ] && RS=\$(ls "\$SERIES_DIR"/*.dcm 2>/dev/null | head -1)
    PATIENT_DIR=\$(dirname "\$(dirname "\$SERIES_DIR")")
    REF_UID=\$(python3 - "\$RS" <<PY
import sys, pydicom
try:
    ds = pydicom.dcmread(sys.argv[1], stop_before_pixels=True, force=True)
    ref = ds.ReferencedFrameOfReferenceSequence[0] \\
            .RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
    print(ref.SeriesInstanceUID)
except Exception:
    pass
PY
)
    if [ -n "\$REF_UID" ] && [ -d "\$PATIENT_DIR/\${REF_UID%/}" -o -d "\$(find "\$PATIENT_DIR" -maxdepth 2 -type d -name "\$REF_UID" -print -quit)" ]; then
        CT_DIR=\$(find "\$PATIENT_DIR" -maxdepth 2 -type d -name "\$REF_UID" -print -quit)
    else
        # Fallback: pick the largest sibling under the same study (legacy heuristic)
        STUDY_DIR=\$(dirname "\$SERIES_DIR")
        CT_DIR=\$(find "\$STUDY_DIR" -mindepth 1 -maxdepth 1 -type d ! -path "\$SERIES_DIR" 2>/dev/null \
            | while read d; do cnt=\$(ls "\$d"/*.dcm 2>/dev/null | wc -l); [ "\$cnt" -gt 5 ] && echo "\$cnt \$d"; done \
            | sort -rn | head -1 | awk '{print \$2}')
    fi
    if [ -z "\$CT_DIR" ]; then
        _record_err "rtstruct-pair" "no CT match for RS=\$RS (REF_UID=\$REF_UID)"
        exit 0
    fi
    OUT_MASK="\$OUT_ROOT/series/\$PAT/\${PAT}__\${SERIES_UID}__\${ROI}-label.nrrd"
    if [ ! -f "\$OUT_MASK" ]; then
        if ! err=\$(qr convert rtstruct -d "\$CT_DIR" -r "\$RS" --roi "\$ROI" -o "\$OUT_MASK" 2>&1); then
            _record_err "rtstruct" "\$err"
        fi
    fi
fi
EOF
    chmod +x "$PROC"

    N_SERIES=$(wc -l < "$SERIES_WORK")
    echo "  $N_SERIES series queued"
    if command -v parallel >/dev/null 2>&1; then
        parallel -j "$N_PARALLEL" --bar "$PROC" :::: "$SERIES_WORK" || true
    else
        cat "$SERIES_WORK" | xargs -P "$N_PARALLEL" -I {} "$PROC" {} || true
    fi
    N_NRRD=$(find "$OUT/series" -type f -name '*.nrrd' 2>/dev/null | wc -l)
    N_ERR=$(grep -c '^\[' "$ERR_LOG" 2>/dev/null || echo 0)
    echo "  series done: $N_NRRD nrrd produced, $N_ERR errors"
    if [ "$N_ERR" -gt 0 ]; then
        echo "  first 5 errors (full log: $ERR_LOG):"
        head -20 "$ERR_LOG" | sed 's/^/    /'
    fi
    _mark series done
fi

# ============================================================================
# Stage 3 — patient aggregation (best CT + matching label per patient)
# ============================================================================
MANIFEST="$OUT/manifest.csv"
if stage_on patient; then
    echo ""
    stage_banner patient
    CURRENT_STAGE=patient; _mark patient running
    python3 - "$OUT" "$ROI" "$MANIFEST" <<'PY'
import json, csv, sys
from pathlib import Path
out = Path(sys.argv[1]); roi = sys.argv[2]; manifest = sys.argv[3]
rows = []
for pat_dir in sorted((out/"series").glob("*")):
    if not pat_dir.is_dir():
        continue
    pat = pat_dir.name
    cts = sorted(pat_dir.glob(f"{pat}__*__CT.nrrd"), key=lambda p: p.stat().st_size, reverse=True)
    masks = sorted(pat_dir.glob(f"{pat}__*__{roi}-label.nrrd"),
                   key=lambda p: p.stat().st_size, reverse=True)
    (out/"patient").mkdir(parents=True, exist_ok=True)
    if not cts or not masks:
        (out/"patient"/f"{pat}.json").write_text(json.dumps({
            "patient_id": pat, "status": "skipped",
            "reason": ("no_ct" if not cts else "no_mask"),
            "n_ct": len(cts), "n_mask": len(masks)}, indent=2))
        continue
    ct, mask = cts[0], masks[0]
    rows.append({"patient_id": pat, "modality": "CT",
                 "image_path": str(ct.resolve()), "mask_path": str(mask.resolve())})
    (out/"patient"/f"{pat}.json").write_text(json.dumps({
        "patient_id": pat, "ct": ct.name, "mask": mask.name,
        "n_ct_candidates": len(cts), "n_mask_candidates": len(masks)}, indent=2))
with open(manifest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["patient_id","modality","image_path","mask_path"])
    w.writeheader(); w.writerows(rows)
print(f"  {len(rows)} patients with CT+{roi} mask")
PY
    _mark patient done
fi

# ============================================================================
# Stage — preprocess (crop to ROI + optional isotropic resample)
# ============================================================================
PRE_DIR="$OUT/preprocess"
PRE_MANIFEST="$OUT/manifest_preprocessed.csv"
PRE_RESAMPLE="${PRE_RESAMPLE:-1.0}"   # set to "" to disable resampling
if stage_on preprocess; then
    echo ""
    stage_banner preprocess
    CURRENT_STAGE=preprocess; _mark preprocess running
    if [ -s "$PRE_MANIFEST" ] && [ "$PRE_MANIFEST" -nt "$MANIFEST" ]; then
        cache_say "$PRE_MANIFEST"
    else
        # 20mm pad around the mask bounding box; optional isotropic resample
        RESAMPLE_OPT=""
        [ -n "$PRE_RESAMPLE" ] && RESAMPLE_OPT="--resample $PRE_RESAMPLE"
        qr preprocess -m "$MANIFEST" -o "$PRE_DIR" \
            --roi-name "$ROI" --pad-mm 20 $RESAMPLE_OPT \
            --jobs "$N_PARALLEL" --out-manifest "$PRE_MANIFEST"
    fi
    _mark preprocess done
fi

# ============================================================================
# Stage — features (PyRadiomics intensity/texture features)
# ============================================================================
FEATURES="$OUT/features.csv"
# Use the preprocessed manifest if preprocess ran (smaller volumes → much
# faster PyRadiomics extraction).
FEATURES_INPUT_MANIFEST="$MANIFEST"
[ -s "$PRE_MANIFEST" ] && FEATURES_INPUT_MANIFEST="$PRE_MANIFEST"

if stage_on features; then
    echo ""
    stage_banner features
    CURRENT_STAGE=features; _mark features running
    if [ -s "$FEATURES" ] && [ "$FEATURES" -nt "$FEATURES_INPUT_MANIFEST" ]; then
        cache_say "$FEATURES"
    else
        qr extract -m "$FEATURES_INPUT_MANIFEST" -p "$PATTERN" -o "$FEATURES" --jobs "$N_PARALLEL"
    fi
    _mark features done
fi

# ============================================================================
# Stage — shape (AHSN + spiculation descriptors via qradiomics.shape)
# ============================================================================
SHAPE_FEATURES="$OUT/shape_features.csv"
SHAPE_SKIP="${SHAPE_SKIP:-0}"
if stage_on shape && [ "$SHAPE_SKIP" != "1" ]; then
    echo ""
    stage_banner shape
    CURRENT_STAGE=shape; _mark shape running
    if [ -s "$SHAPE_FEATURES" ] && [ "$SHAPE_FEATURES" -nt "$FEATURES_INPUT_MANIFEST" ]; then
        cache_say "$SHAPE_FEATURES"
    else
        qr shape extract -m "$FEATURES_INPUT_MANIFEST" -o "$SHAPE_FEATURES" \
            --jobs "$N_PARALLEL" || true   # shape is optional; don't abort cohort
    fi
    _mark shape done
fi

# ============================================================================
# Stage — merge clinical
# ============================================================================
ANALYSIS_READY="$OUT/analysis_ready.csv"
if stage_on merge; then
    echo ""
    stage_banner merge
    CURRENT_STAGE=merge; _mark merge running
    if [ -f "$CLINICAL" ]; then
        qr results merge -f "$FEATURES" -c "$CLINICAL" \
            --clinical-id-col patient_id --time-col OS_days --event-col OS_event \
            -o "$ANALYSIS_READY" || cp "$FEATURES" "$ANALYSIS_READY"
    else
        echo "  no clinical CSV at $CLINICAL → using features.csv as analysis_ready"
        cp "$FEATURES" "$ANALYSIS_READY"
    fi
    _mark merge done
fi

# ============================================================================
# Stage — modeling (qr ml train + evaluate)
# ============================================================================
if stage_on modeling; then
    echo ""
    stage_banner modeling
    CURRENT_STAGE=modeling; _mark modeling running
    TIME_OPT=""
    [ "$TASK" = "survival" ] && TIME_OPT="--time-col OS_months"
    N_ROWS=$(($(wc -l < "$ANALYSIS_READY" 2>/dev/null || echo 1) - 1))
    N_COLS=$(head -1 "$ANALYSIS_READY" 2>/dev/null | awk -F, '{print NF}')
    echo "  task=$TASK  outcome=$OUTCOME  rows=$N_ROWS  cols=$N_COLS"
    echo "  → training (qr ml train)"
    qr ml train -i "$ANALYSIS_READY" --task "$TASK" --outcome "$OUTCOME" $TIME_OPT \
        --model "$OUT/model.pkl" --metrics "$OUT/cv_metrics.json"
    echo "  → evaluating (qr ml evaluate)"
    qr ml evaluate -i "$ANALYSIS_READY" --model "$OUT/model.pkl" \
        --task "$TASK" --outcome "$OUTCOME" $TIME_OPT \
        --report "$OUT/evaluation.json"
    _mark modeling done
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "DONE: $COLLECTION → $OUT/"
echo "  catalog : ct_series.csv ($(($(wc -l < "$CT_SERIES" 2>/dev/null || echo 1) - 1))) + rt_series.csv"
echo "  dicom   : $(find "$OUT/dicom" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | wc -l) series"
echo "  series  : $(find "$OUT/series" -name '*.nrrd' 2>/dev/null | wc -l) NRRDs"
echo "  patient : $(ls "$OUT/patient"/*.json 2>/dev/null | wc -l) patient summaries"
echo "  manifest: $(($(wc -l < "$MANIFEST" 2>/dev/null || echo 1) - 1)) usable patients"
echo "  outputs : features.csv + analysis_ready.csv + model.pkl + cv_metrics.json + evaluation.json"
echo "════════════════════════════════════════════════════════════════════"
