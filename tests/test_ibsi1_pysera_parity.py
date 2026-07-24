"""IBSI-1 digital-phantom feature parity regression test (PySERA engine).

Sibling of test_ibsi1_phantom_parity.py — pins the pysera extractor against
the same published IBSI-1 chapter-2 digital-phantom reference values (config
"A": no interpolation / resegmentation / normalization, fixed bin size 1,
full 3D).

The phantom + reference table live outside the repository (a local checkout
of the `mirp` test fixtures, path set via $IBSI1_MIRP_DATA), so the test
skips cleanly when the data is absent.

Achieved parity at authoring time: 117/119 mapped features within tolerance
(98.3%). The two persistent failures (cm_inv_diff_norm_3D_avg,
cm_inv_diff_mom_norm_3D_avg) are a documented pysera implementation
deviation (IDN/IDMN normalise by Ng-1 instead of Ng), not a config bug. We
assert a conservative floor of 95%, mirroring the PyRadiomics sibling test.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PIPELINE = Path(__file__).resolve().parents[1] / "pipelines" / "ibsi1"
_PARITY_PY = _PIPELINE / "run_ibsi1_pysera_parity.py"

# Minimum fraction of mapped features required within tolerance. Set below
# the achieved 0.983 so the documented IDN/IDMN deviation does not flip the
# gate.
MIN_PASS_FRACTION = 0.95


def _load_parity_module():
    spec = importlib.util.spec_from_file_location("ibsi1_pysera_parity", _PARITY_PY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ibsi1_pysera_parity"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def parity():
    if not _PARITY_PY.is_file():
        pytest.skip("ibsi1 pysera parity helper not present")
    mod = _load_parity_module()
    if not mod.data_available():
        pytest.skip("IBSI-1 phantom data not available (set $IBSI1_MIRP_DATA)")
    try:
        import pysera  # noqa: F401
    except ImportError:
        pytest.skip("pysera not installed")
    return mod.evaluate_pysera()


def test_ibsi1_pysera_parity(parity):
    n_mapped = parity["n_mapped"]
    n_pass = parity["n_pass"]
    frac = parity["pct_within_tolerance"] / 100.0

    assert n_mapped >= 110, f"expected >=110 mapped IBSI tags, got {n_mapped}"

    failures = [r for r in parity["rows"] if not r[5]]
    fail_repr = "\n".join(
        f"  {tag}: ours={ours} ref={ref} tol={tol} |d|={ad:.4g}"
        for tag, ours, ref, tol, ad, _ in failures
    )
    assert frac >= MIN_PASS_FRACTION, (
        f"IBSI-1 pysera phantom parity {n_pass}/{n_mapped} ({frac:.1%}) "
        f"below floor {MIN_PASS_FRACTION:.0%}.\nFailures:\n{fail_repr}"
    )
