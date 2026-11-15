"""rtools is the default extraction engine; pyradiomics is explicit opt-in (#28)."""

import csv
import importlib

import pytest
from click.testing import CliRunner

from qradiomics import PatternLoader
from qradiomics.extractor import PYRADIOMICS_INSTALL_HINT
from qradiomics.extractor_registry import DEFAULT_ENGINE, resolve_engines
from qradiomics.pattern_loader import FeatureExtractionConfig
from qradiomics.workflow import template_nrrd_survival

extract_mod = importlib.import_module("qradiomics.cli.commands.extract")

KW = dict(cohort_root="/c", clinical_csv="/c.csv")


def _manifest(tmp_path):
    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "modality", "image_path", "mask_path"])
        w.writerow(["P1", "CT", "img.nrrd", "mask.nrrd"])
    return path


@pytest.fixture
def captured(monkeypatch):
    """Replace the engine runner; record which engines `qr extract` resolved."""
    seen = {}

    def fake_run(engines, job_id, manifest_path, job_dir, extraction_settings, jobs):
        seen["engines"] = list(engines)
        (job_dir / "features.csv").write_text("patient_id,f\nP1,1\n")
        return {"status": "extracted", "patients_processed": 1, "feature_count": 1}

    monkeypatch.setattr(extract_mod, "run_multi_extraction", fake_run)
    return seen


def _invoke(tmp_path, *args):
    return CliRunner().invoke(
        extract_mod.extract, ["-m", str(_manifest(tmp_path)), "-o", str(tmp_path / "f.csv"), *args]
    )


def test_registry_default_is_rtools():
    assert DEFAULT_ENGINE == "rtools"
    assert resolve_engines(None) == ["rtools"]
    assert resolve_engines("", None) == ["rtools"]


def test_pattern_without_extractor_resolves_to_rtools():
    cfg = FeatureExtractionConfig()
    assert cfg.engines == ("rtools",)
    assert FeatureExtractionConfig(extractor=[]).engines == ("rtools",)


def test_shipped_patterns_default_to_rtools_and_keep_pyradiomics_settings():
    loader = PatternLoader()
    ids = [p["pattern_id"] for p in loader.list_patterns()]
    assert set(ids) >= {"ct-default", "pet-default", "nsclc-survival",
                        "standard-radiomics", "survival-analysis"}
    for pid in ids:
        fx = loader.get_pattern(pid).feature_extraction
        assert fx.engine_spec == "rtools", pid
        assert fx.config_ref, pid  # pyradiomics config kept for --engine pyradiomics
    nsclc = loader.get_pattern("nsclc-survival").feature_extraction
    assert {"LoG", "Wavelet"} <= set(nsclc.image_types)


def test_pyradiomics_params_files_still_shipped():
    from importlib import resources

    root = resources.files("qradiomics.data") / "pyradiomics"
    names = {p.name for p in root.iterdir()}
    assert {"ct_default.yaml", "pet_default.yaml", "nsclc_ct.yaml"} <= names


def test_pattern_settings_reach_pyradiomics_engine(tmp_path, captured, monkeypatch):
    """With --engine pyradiomics the pattern's filtered image types are passed through."""
    seen = {}

    def fake_run(engines, job_id, manifest_path, job_dir, extraction_settings, jobs):
        seen["engines"] = list(engines)
        seen["settings"] = dict(extraction_settings)
        (job_dir / "features.csv").write_text("patient_id,f\nP1,1\n")
        return {"status": "extracted", "patients_processed": 1, "feature_count": 1}

    monkeypatch.setattr(extract_mod, "run_multi_extraction", fake_run)
    monkeypatch.setattr(extract_mod, "pyradiomics_available", lambda: True)
    res = _invoke(tmp_path, "-p", "nsclc-survival", "--engine", "pyradiomics")
    assert res.exit_code == 0, res.output
    assert seen["engines"] == ["pyradiomics"]
    assert {"LoG", "Wavelet"} <= set(seen["settings"]["image_types"])


def test_shipped_pattern_runs_on_plain_install_with_ignored_settings_note(
    tmp_path, captured, monkeypatch
):
    monkeypatch.setattr(extract_mod, "pyradiomics_available", lambda: False)
    try:
        runner = CliRunner(mix_stderr=False)  # click < 8.2
    except TypeError:  # click >= 8.2 always separates stderr
        runner = CliRunner()
    res = runner.invoke(
        extract_mod.extract,
        ["-m", str(_manifest(tmp_path)), "-p", "nsclc-survival", "-o", str(tmp_path / "f.csv")],
    )
    assert res.exit_code == 0, res.output
    assert captured["engines"] == ["rtools"]
    assert "Note: engine rtools" in res.stderr and "--engine pyradiomics" in res.stderr
    assert "LoG" in res.stderr


def test_extract_default_engine_is_rtools_without_pyradiomics(tmp_path, captured, monkeypatch):
    # Works on a plain install: pyradiomics absent must not matter by default.
    monkeypatch.setattr(extract_mod, "pyradiomics_available", lambda: False)
    res = _invoke(tmp_path)
    assert res.exit_code == 0, res.output
    assert captured["engines"] == ["rtools"]


def test_extract_explicit_pyradiomics_when_available(tmp_path, captured, monkeypatch):
    monkeypatch.setattr(extract_mod, "pyradiomics_available", lambda: True)
    res = _invoke(tmp_path, "--engine", "pyradiomics")
    assert res.exit_code == 0, res.output
    assert captured["engines"] == ["pyradiomics"]


@pytest.mark.parametrize("args", [["--engine", "pyradiomics"], ["-p", "nsclc-survival", "--engine", "pyradiomics"]])
def test_explicit_pyradiomics_missing_error_explains_install(tmp_path, captured, monkeypatch, args):
    monkeypatch.setattr(extract_mod, "pyradiomics_available", lambda: False)
    res = _invoke(tmp_path, *args)
    assert res.exit_code == 1
    assert "engines" not in captured  # failed before any extraction ran
    assert "git+https://github.com/AIM-Harvard/pyradiomics.git" in res.output
    assert "--engine rtools" in res.output


def test_pattern_with_engine_override_uses_rtools_without_pyradiomics(tmp_path, captured, monkeypatch):
    monkeypatch.setattr(extract_mod, "pyradiomics_available", lambda: False)
    res = _invoke(tmp_path, "-p", "ct-default", "--engine", "rtools")
    assert res.exit_code == 0, res.output
    assert captured["engines"] == ["rtools"]


def test_install_hint_mentions_git_url_and_rtools():
    assert "git+https://github.com/AIM-Harvard/pyradiomics.git" in PYRADIOMICS_INSTALL_HINT
    assert "--engine rtools" in PYRADIOMICS_INSTALL_HINT


def test_workflow_plan_engine_follows_pattern():
    def eng(**kw):
        return template_nrrd_survival(**KW, **kw).vars["engine"]

    assert eng() == "rtools"  # default pattern nsclc-survival -> rtools
    assert eng(pattern="no-such-pattern") == "rtools"  # nothing configured -> default
    assert eng(engine="pyradiomics") == "pyradiomics"  # explicit wins


def test_plan_json_files_default_to_rtools():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    plans = sorted(root.glob("pipelines/*/plan.json")) + sorted(root.glob("nf-modules/cohorts/*.json"))
    assert len(plans) >= 4  # the public tree ships pipelines/*/plan.json only (no nf-modules/)
    for path in plans:
        assert json.loads(path.read_text())["vars"]["engine"] == "rtools", path.name
