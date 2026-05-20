#!/usr/bin/env bash
# LIDC-IDRI end-to-end reproducibility pipeline
#
# Stages:
#   1. (assumed) `qr tcia download --collection LIDC-IDRI ...` → DICOM + XML
#   2. `qr lidc convert-cohort` → per-patient CT NRRD + per-reader masks + nodules.csv
#   3. STAPLE consensus mask per patient (optional, all-readers consensus)
#   4. extract_features.py → per-(reader,nodule) radiomics + spiculation CSV
#
# Reproducibility targets:
#   - Choi 2014 CMPB AHSN (nodule vs non-nodule detection)
#   - Choi 2021 CMPB spiculation features (Np / Na / Nl / s1 / s2)
#   - LIDC malignancy classification (radiomics 1409 + spiculation)
#
# Env override:
#   LIDC_SRC=/data/users/wxc151/LIDC-IDRI         # raw TCIA tree
#   LIDC_OUT=/data/users/wxc151/LIDC-IDRI-out     # converted patients
#   FEATURES_CSV=$LIDC_OUT/features.csv           # final feature table
#   JOBS=16                                       # parallel workers
#   LIMIT=                                        # smoke-test limit

set -euo pipefail
LIDC_SRC=${LIDC_SRC:-/data/users/wxc151/LIDC-IDRI}
LIDC_OUT=${LIDC_OUT:-/data/users/wxc151/LIDC-IDRI-out}
FEATURES_CSV=${FEATURES_CSV:-$LIDC_OUT/features.csv}
JOBS=${JOBS:-16}
LIMIT=${LIMIT:-}

PUB=/home/wxc151/gitRepos/qradiomics-public
HERE=$(dirname "$0")

mkdir -p "$LIDC_OUT"

echo "═══════════════════════════════════════════════════════════════"
echo "  LIDC-IDRI reproducibility pipeline"
echo "  src: $LIDC_SRC"
echo "  out: $LIDC_OUT"
echo "  jobs: $JOBS  ${LIMIT:+limit=$LIMIT}"
echo "═══════════════════════════════════════════════════════════════"

# Stage 2: XML → NRRD (CT + per-reader masks)
echo "[2/4] qr lidc convert-cohort"
PYTHONPATH="$PUB" python3 -m qradiomics.cli.main lidc convert-cohort \
    --src "$LIDC_SRC" --out "$LIDC_OUT" --jobs "$JOBS" ${LIMIT:+--limit "$LIMIT"}

# Stage 3: STAPLE consensus per patient (optional — best-effort)
echo "[3/4] STAPLE consensus per patient"
PYTHONPATH="$PUB" python3 -c "
from pathlib import Path
from qradiomics.io.lidc import staple_patient
out = Path('$LIDC_OUT')
ok = fail = 0
for pat in sorted(p for p in out.iterdir() if p.is_dir()):
    try:
        r = staple_patient(pat, pat.name)
        if r['n_readers'] > 1:
            ok += 1
            print(f\"  ✓ {pat.name}: {r['n_readers']} readers → {r['consensus_voxels']} consensus voxels\")
    except Exception as e:
        fail += 1
        print(f\"  ✘ {pat.name}: {type(e).__name__}: {e}\")
print(f'STAPLE: {ok} ok / {fail} fail')
"

# Stage 4: feature extraction (atomic radiomics + spiculation)
echo "[4/4] feature extraction → $FEATURES_CSV"
PYTHONPATH="$PUB" python3 "$HERE/extract_features.py" \
    --lidc-out "$LIDC_OUT" \
    --lidc-src "$LIDC_SRC" \
    --out "$FEATURES_CSV" \
    --jobs "$JOBS" \
    ${LIMIT:+--limit "$LIMIT"}

echo "✓ done → $FEATURES_CSV"
