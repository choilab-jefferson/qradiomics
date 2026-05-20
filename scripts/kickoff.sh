#!/usr/bin/env bash
# qradiomics kick-off — one-shot clone + install + smoke test.
#
# Designed for two entry points:
#
#   (a) Run from outside the repo (no clone yet):
#         curl -sSL https://raw.githubusercontent.com/choilab-jefferson/qradiomics/main/scripts/kickoff.sh | bash
#
#   (b) Run from inside an existing clone:
#         bash scripts/kickoff.sh
#         bash scripts/kickoff.sh --skip-clone
#
# Outcome: a Python environment with `qradiomics` installed, `qr info`
# printing the install mode, and the smoke test suite green. The script
# never installs system packages, never touches PHI, and never pushes
# anywhere. It is safe to re-run.
#
# Env overrides:
#   QR_REPO_URL          git URL to clone; default: public GitHub mirror
#   QR_REPO_DIR          where to clone; default: ./qradiomics
#   QR_BRANCH            branch to check out; default: main
#   QR_PYTHON            python binary to use; default: python3
#   QR_SKIP_SMOKE=1      skip pytest; just install + qr info
#   QR_VENV              path to virtualenv to create; default: <repo>/.venv
#                        Set QR_VENV=- to install in the current environment.

set -euo pipefail

DEFAULT_URL="https://github.com/choilab-jefferson/qradiomics.git"

REPO_URL="${QR_REPO_URL:-${1:-$DEFAULT_URL}}"
REPO_DIR="${QR_REPO_DIR:-qradiomics}"
BRANCH="${QR_BRANCH:-}"
PY="${QR_PYTHON:-python3}"
VENV="${QR_VENV:-}"
SKIP_CLONE=0

for a in "$@"; do
  case "$a" in
    --skip-clone) SKIP_CLONE=1 ;;
  esac
done

log()  { printf '\033[1;36m[kickoff]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n'   "$*"; }
fail() { printf '\033[1;31m  ✗\033[0m %s\n' "$*"; exit 1; }

# ── 0. Detect whether we are already inside a qradiomics clone ──────────────
if [ "$SKIP_CLONE" -eq 0 ] && [ -f "pyproject.toml" ] && grep -q '^name = "qradiomics"' pyproject.toml 2>/dev/null; then
  log "detected existing qradiomics clone at $PWD; skipping clone"
  SKIP_CLONE=1
  REPO_DIR="$PWD"
fi

# ── 1. Clone ────────────────────────────────────────────────────────────────
if [ "$SKIP_CLONE" -eq 0 ]; then
  if [ -e "$REPO_DIR" ]; then
    log "$REPO_DIR already exists — reusing"
  else
    log "cloning $REPO_URL → $REPO_DIR"
    if [ -n "$BRANCH" ]; then
      git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$REPO_DIR" \
        || fail "git clone failed"
    else
      git clone --depth 1 "$REPO_URL" "$REPO_DIR" \
        || fail "git clone failed"
    fi
  fi
  cd "$REPO_DIR"
  ok "in $(pwd)"
else
  ok "in $(pwd)"
fi

# ── 2. Virtualenv (optional) ────────────────────────────────────────────────
if [ -z "$VENV" ]; then
  VENV=".venv"
fi
if [ "$VENV" = "-" ]; then
  log "installing into the current Python environment (no venv)"
else
  if [ ! -d "$VENV" ]; then
    log "creating virtualenv at $VENV"
    "$PY" -m venv "$VENV" || fail "venv creation failed"
  fi
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  PY="$(command -v python)"
  ok "venv active: $(command -v python)"
fi

# ── 3. Install ──────────────────────────────────────────────────────────────
log "pip install -e ."
$PY -m pip install --upgrade pip setuptools wheel >/dev/null

# pyradiomics on PyPI (3.1.0) only ships wheels up to cp39. For Python 3.10+
# the sdist build fails silently inside pip's isolated build env (missing
# numpy/cython), pip falls back to a pre-3.1.0 candidate, and the install
# errors out as "No matching distribution found for pyradiomics>=3.1.0".
# Install pyradiomics from the upstream git first on Python 3.10+ — that's
# also what dev installs already use (3.1.1.dev) and what qradiomics is
# actually built against.
PYMINOR=$($PY -c 'import sys; print(sys.version_info[1])')
if [ "${PYMINOR:-0}" -ge 10 ]; then
  log "pyradiomics: installing from git (PyPI wheels only cover cp37-cp39)"
  $PY -m pip install --quiet \
    "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git" \
    || fail "git install of pyradiomics failed"
fi

$PY -m pip install --quiet -e . || fail "pip install -e . failed"
ok "qradiomics installed"

# ── 4. qr info ──────────────────────────────────────────────────────────────
log "qr info"
if ! command -v qr >/dev/null 2>&1; then
  fail "qr CLI not found on PATH after install"
fi
qr info 2>&1 | sed 's/^/    /'
ok "qr CLI ready"

# ── 5. Smoke test ───────────────────────────────────────────────────────────
# Two-part smoke: a fast pytest subset (~5 s, no PyRadiomics) + a real
# end-to-end CLI run on synthetic NRRD (~15 s, exercises extract → merge
# → analyze). The full pytest suite ships under tests/ and can be run
# manually with `pytest tests/` when you want the heavy parity checks.
if [ "${QR_SKIP_SMOKE:-0}" = "1" ]; then
  log "QR_SKIP_SMOKE=1 — skipping smoke"
else
  if ! $PY -c "import pytest" 2>/dev/null; then
    log "installing pytest for smoke test"
    $PY -m pip install --quiet pytest || fail "pytest install failed"
  fi

  FAST_TESTS="tests/test_data_model.py tests/test_pet_suv.py \
              tests/test_delta.py tests/test_results_merge.py \
              tests/test_lidc.py"
  log "fast unit tests: pytest ${FAST_TESTS// /, }"
  if $PY -m pytest $FAST_TESTS -q --no-header --tb=line; then
    ok "fast unit tests green"
  else
    fail "fast unit tests failed — see output above"
  fi

  log "end-to-end smoke (synthetic NRRD → qr extract → merge → analyze)"
  if $PY scripts/smoke.py; then
    ok "end-to-end smoke green"
  else
    fail "end-to-end smoke failed — see output above"
  fi
fi

log "kick-off complete."
echo ""
echo "Next:"
echo "  qr --help               browse the CLI"
echo "  qr info                 show install info"
echo "  qr phi-check <csv>      gate clinical CSVs before staging"
echo ""
echo "Operations reference: AGENTS.md (conventions, layout, don'ts, citations)."
