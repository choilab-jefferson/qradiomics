# Sourced by pipeline scripts to resolve a working `qr` command.
#
# Resolution order:
#   1. If an env-provided $QR is already callable, use it.
#   2. If the installed `qr` shim already has the v0.9 'workflow' command, use it.
#   3. Otherwise fall back to `python3 -m qradiomics.cli.main` with PYTHONPATH
#      pointing at the qradiomics-public repo root (this script's parent's parent).
#
# Exports: $QR (the actual command), PYTHONPATH (if fallback was used).
# Defines a shell function `qr` that delegates to $QR so existing scripts
# work unchanged.
#
# Also sources _env.sh so MLFLOW_TRACKING_URI / PREFECT_API_URL get the
# canonical lab defaults before any qr / Prefect / MLflow client runs.

# shellcheck source=./_env.sh
. "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

if [ -z "${QR:-}" ]; then
    if command -v qr >/dev/null 2>&1 && command qr workflow templates >/dev/null 2>&1; then
        QR=$(command -v qr)
    else
        _qr_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
        export PYTHONPATH="${_qr_repo_root}${PYTHONPATH:+:$PYTHONPATH}"
        QR="python3 -m qradiomics.cli.main"
    fi
fi
export QR

# Shell function: makes `qr ...` work in any bash subshell that sources this
# file. `export -f` is bash-only, so guard it.
qr() {
    $QR "$@"
}
if [ -n "${BASH_VERSION:-}" ]; then
    export -f qr
fi
