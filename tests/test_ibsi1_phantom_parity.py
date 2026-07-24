"""IBSI-1 digital-phantom feature parity regression test.

Pins the qradiomics PyRadiomics extractor against the published IBSI-1
chapter-2 digital-phantom reference values (config "A": no interpolation /
resegmentation / normalization, fixed integer discretisation, full 3D).

The phantom + reference table live outside the repository (a local checkout
of the `mirp` test fixtures under ~/gitRepos/mirp/test/data, path set via
$IBSI1_MIRP_DATA), so the test skips cleanly when the data is absent.

Achieved parity at authoring time: 97/99 mapped features within tolerance
(98.0%). The two persistent failures (morph_pca_min_axis, morph_pca_least_axis)
are a known PyRadiomics-vs-IBSI deviation in principal-axis length convention,
not a config bug. We assert a conservative floor of 95% so the check pins
compliance without being brittle to that documented deviation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PIPELINE = Path(__file__).resolve().parents[1] / "pipelines" / "ibsi1"
_PARITY_PY = _PIPELINE / "run_ibsi1_parity.py"

# Minimum fraction of mapped features required within tolerance. Set below the
# achieved 0.98 so the documented PCA-axis deviation does not flip the gate.
MIN_PASS_FRACTION = 0.95


def _load_parity_module():
    spec = importlib.util.spec_from_file_location("ibsi1_parity", _PARITY_PY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ibsi1_parity"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def parity():
    if not _PARITY_PY.is_file():
        pytest.skip("ibsi1 parity helper not present")
    mod = _load_parity_module()
    if not mod.data_available():
        pytest.skip("IBSI-1 phantom data not available (set $IBSI1_MIRP_DATA)")
    try:
        import radiomics  # noqa: F401
    except ImportError:
        pytest.skip("PyRadiomics (radiomics) not installed")
    return mod.evaluate()


def test_ibsi1_phantom_parity(parity):
    n_mapped = parity["n_mapped"]
    n_pass = parity["n_pass"]
    frac = parity["pct_within_tolerance"] / 100.0

    assert n_mapped >= 90, f"expected >=90 mapped IBSI tags, got {n_mapped}"

    failures = [r for r in parity["rows"] if not r[5]]
    fail_repr = "\n".join(
        f"  {tag}: ours={ours} ref={ref} tol={tol} |d|={ad:.4g}"
        for tag, ours, ref, tol, ad, _ in failures
    )
    assert frac >= MIN_PASS_FRACTION, (
        f"IBSI-1 phantom parity {n_pass}/{n_mapped} ({frac:.1%}) "
        f"below floor {MIN_PASS_FRACTION:.0%}.\nFailures:\n{fail_repr}"
    )
