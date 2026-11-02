"""The shipped data files must be resolvable from an installed package, not just a checkout.

Regression guard for issue #15: ``qradiomics/data/`` was absent from built wheels entirely,
because the package-data globs matched only files directly under ``qradiomics/``. Consumers
resolving the per-modality PyRadiomics params via ``importlib.resources`` got nothing, and it
was invisible in development because an editable install reads the source tree.
"""

from importlib.resources import files

import pytest
import yaml

#: (filename, binWidth, normalize). CT is Hounsfield units, PET is SUV.
PARAMS = [
    ("ct_default.yaml", 25, False),
    ("ct_original_only.yaml", 25, False),
    ("pet_default.yaml", 0.25, False),
    ("pet_original_only.yaml", 0.25, False),
]


def _params_dir():
    return files("qradiomics") / "data" / "pyradiomics"


@pytest.mark.parametrize("name,bin_width,normalize", PARAMS)
def test_params_file_resolves_and_carries_its_settings(name, bin_width, normalize):
    path = _params_dir() / name
    assert path.is_file(), f"{name} is not resolvable from the installed package"

    cfg = yaml.safe_load(path.read_text())
    setting = cfg.get("setting") or {}
    assert setting["binWidth"] == bin_width
    assert setting["normalize"] is normalize, (
        "normalize must stay false at these bin widths -- normalizing discards the HU/SUV "
        "calibration the bin width is chosen against"
    )


@pytest.mark.parametrize("name", ["ct_original_only.yaml", "pet_original_only.yaml"])
def test_original_only_variants_enable_only_the_original_image_type(name):
    cfg = yaml.safe_load((_params_dir() / name).read_text())
    assert sorted(cfg["imageType"]) == [
        "Original"
    ], f"{name} is the light variant; enabling the filter bank defeats its purpose"


def test_every_modality_has_a_light_variant():
    """CT had one and PET did not, which forced full filter-bank extraction on PET."""
    d = _params_dir()
    for modality in ("ct", "pet"):
        assert (d / f"{modality}_default.yaml").is_file()
        assert (d / f"{modality}_original_only.yaml").is_file()


def test_schema_and_templates_also_ship():
    """The same glob bug hid these; they are equally part of the package's contract."""
    data = files("qradiomics") / "data"
    assert (data / "schema" / "pattern_schema.yaml").is_file()
    assert (data / "templates" / "standard_radiomics.yaml").is_file()


def _settings_blocks(cfg):
    """PyRadiomics params keep settings under ``setting``; patterns nest them one level down."""
    if not isinstance(cfg, dict):
        return
    if isinstance(cfg.get("setting"), dict):
        yield "setting", cfg["setting"]
    extraction = cfg.get("feature_extraction")
    if isinstance(extraction, dict) and isinstance(extraction.get("settings"), dict):
        yield "feature_extraction.settings", extraction["settings"]


def _bundled_configs():
    data = files("qradiomics") / "data"
    for subdir in ("pyradiomics", "templates"):
        for entry in (data / subdir).iterdir():
            if entry.name.endswith((".yaml", ".yml")):
                yield f"{subdir}/{entry.name}", yaml.safe_load(entry.read_text())


def test_no_bundled_config_normalizes_at_a_small_bin_width():
    """``normalize: true`` z-scores intensities to std ~1, so a binWidth of 25 buckets the whole
    ROI into ~1 bin and every texture feature degenerates (firstorder_Entropy ~= 0). This pairing
    silently invalidated a real analysis here; the JeffLuRa canonical fix (02e4e3af, 2026-03-18)
    removed it. CT is Hounsfield units and PET is SUV -- both are already calibrated.
    """
    offenders = []
    for name, cfg in _bundled_configs():
        for where, settings in _settings_blocks(cfg):
            if settings.get("normalize") is not True:
                continue
            bin_width = settings.get("binWidth")
            widths = bin_width.values() if isinstance(bin_width, dict) else [bin_width]
            if any(isinstance(w, (int, float)) and w <= 25 for w in widths):
                offenders.append(f"{name} ({where}): binWidth={bin_width}")
    assert not offenders, "normalize:true paired with a small binWidth in " + ", ".join(offenders)


def test_pet_bin_width_is_the_suv_value_everywhere():
    """0.5 was the pre-2026-03-18 PET bin width; the canonical SUV setting is 0.25."""
    for name, cfg in _bundled_configs():
        for where, settings in _settings_blocks(cfg):
            bin_width = settings.get("binWidth")
            per_modality = bin_width if isinstance(bin_width, dict) else {}
            for modality, width in per_modality.items():
                if "PET" in modality.upper() or "PT" == modality.upper():
                    assert width == 0.25, f"{name} ({where}): PET binWidth={width}, expected 0.25"
            if "pet" in name.lower() and not per_modality:
                assert bin_width == 0.25, f"{name} ({where}): PET binWidth={bin_width}"


def test_geometry_tolerance_is_a_number_not_a_string():
    """YAML 1.1 resolves ``1e-3`` as a *string* -- it needs ``1.0e-3`` to become a float. Consumers
    that read these files directly (PatternLoader.get_pyradiomics_config, pattern ``settings``)
    would otherwise hand SimpleITK a str and get a TypeError on the tolerance setter.
    """
    for name, cfg in _bundled_configs():
        for where, settings in _settings_blocks(cfg):
            if "geometryTolerance" in settings:
                value = settings["geometryTolerance"]
                assert isinstance(value, float), f"{name} ({where}): {value!r} is not a float"


def test_every_modality_has_a_pattern_template():
    """CT had a pattern template and PET did not, so PET extraction bypassed patterns entirely."""
    templates = files("qradiomics") / "data" / "templates"
    for modality in ("ct", "pet"):
        assert (templates / f"{modality}_default.yaml").is_file()
