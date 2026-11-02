"""Tests for feature fusion (#13) and multi-engine workflow plans (#14)."""

import pandas as pd
import pytest
from click.testing import CliRunner

from qradiomics.analytics.fusion import fuse_feature_tables
from qradiomics.cli.commands.results import results
from qradiomics.pattern_loader import FeatureExtractionConfig
from qradiomics.workflow import WorkflowRunner, build_workflow_plan, template_dicom_to_ml


@pytest.fixture
def engines():
    a = pd.DataFrame({"patient_id": ["p1", "p2"], "pyradiomics_f": [1.0, 2.0]})
    b = pd.DataFrame({"patient_id": ["p2", "p3"], "pysera_f": [3.0, 4.0]})
    return a, b


def test_fuse_internal_engines_outer_and_inner(engines):
    a, b = engines
    out = fuse_feature_tables([a, b])
    assert sorted(out["patient_id"]) == ["p1", "p2", "p3"]
    assert {"pyradiomics_f", "pysera_f"} <= set(out.columns)
    inner = fuse_feature_tables([a, b], how="inner")
    assert list(inner["patient_id"]) == ["p2"]


def test_fuse_external_csv_and_label_prefix(tmp_path, engines):
    a, _ = engines
    ext = tmp_path / "ext.csv"
    pd.DataFrame({"patient_id": ["p1", "p2"], "f": [9, 8]}).to_csv(ext, index=False)
    out = fuse_feature_tables([a], [("ext", ext)])
    assert list(out.columns) == ["patient_id", "pyradiomics_f", "ext_f"]


def test_fuse_collision_and_missing_id_raise(engines):
    a, _ = engines
    with pytest.raises(ValueError, match="collision"):
        fuse_feature_tables([a, a])
    with pytest.raises(ValueError, match="missing id"):
        fuse_feature_tables([pd.DataFrame({"x": [1]})])
    with pytest.raises(ValueError, match="at least one"):
        fuse_feature_tables([])


def _timed():
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2"],
            "timepoint": ["pre", "post", "pre"],
            "pyradiomics_f": [1.0, 2.0, 3.0],
        }
    )


def test_fuse_long_timepoint_broadcasts_static_external():
    clin = pd.DataFrame({"patient_id": ["p1", "p2"], "age": [60, 70]})
    out = fuse_feature_tables([_timed()], [clin], timepoint_col="timepoint")
    assert len(out) == 3
    assert out.loc[out["patient_id"] == "p1", "age"].tolist() == [60, 60]


def test_fuse_long_timepoint_joins_on_both_keys():
    other = _timed().rename(columns={"pyradiomics_f": "pysera_f"})
    out = fuse_feature_tables([_timed(), other], timepoint_col="timepoint")
    assert len(out) == 3 and {"pyradiomics_f", "pysera_f"} <= set(out.columns)


def test_fuse_wide_timepoint_prefix():
    clin = pd.DataFrame({"patient_id": ["p1", "p2"], "age": [60, 70]})
    out = fuse_feature_tables([_timed()], [clin], timepoint_col="timepoint", wide=True)
    assert len(out) == 2
    assert {"pre__pyradiomics_f", "post__pyradiomics_f", "age"} <= set(out.columns)
    p1 = out[out["patient_id"] == "p1"].iloc[0]
    assert p1["post__pyradiomics_f"] == 2.0


def test_fuse_duplicate_keys_raise():
    dup = pd.DataFrame({"patient_id": ["p1", "p1"], "f": [1, 2]})
    with pytest.raises(ValueError, match="duplicate"):
        fuse_feature_tables([dup])


def test_cli_fuse(tmp_path, engines):
    a, b = engines
    pa, pb, out = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "o" / "fused.csv"
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)
    res = CliRunner().invoke(results, ["fuse", "-f", f"{pa},{pb}", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert len(pd.read_csv(out)) == 3
    bad = CliRunner().invoke(results, ["fuse", "-f", str(tmp_path / "nope.csv"), "-o", str(out)])
    assert bad.exit_code == 1


def test_engines_property_backward_compatible():
    assert FeatureExtractionConfig().engines == ("pyradiomics",)
    assert FeatureExtractionConfig(extractor="pysera").engines == ("pysera",)
    cfg = FeatureExtractionConfig(extractor=["pyradiomics", "pysera", "pyradiomics"])
    assert cfg.engines == ("pyradiomics", "pysera")
    assert FeatureExtractionConfig(extractor="a, b").engine_spec == "a,b"


KW = dict(cohort_root="/c", clinical_csv="/c.csv")


def _extract_cmd(plan, step_id):
    step = next(s for s in plan.steps if s.id == step_id)
    return WorkflowRunner(plan, dry_run=True)._build_command(step)


def test_plan_multi_engine_single_step():
    cfg = FeatureExtractionConfig(extractor=["pyradiomics", "pysera"])
    plan = build_workflow_plan(cfg, **KW)
    cmd = _extract_cmd(plan, "extract")
    assert cmd[cmd.index("--engine") + 1] == "pyradiomics,pysera"


def test_plan_default_engine_flag():
    plan = template_dicom_to_ml(**KW)
    cmd = _extract_cmd(plan, "extract")
    assert cmd[cmd.index("--engine") + 1] == "pyradiomics"


def test_plan_per_engine_steps_and_fuse():
    cfg = FeatureExtractionConfig(extractor="pyradiomics,pysera")
    plan = build_workflow_plan(cfg, per_engine=True, **KW)
    ids = [s.id for s in plan.steps]
    assert "extract" not in ids
    assert ids.index("extract_pyradiomics") < ids.index("extract_pysera") < ids.index("fuse")
    cmd = _extract_cmd(plan, "extract_pysera")
    assert cmd[cmd.index("--engine") + 1] == "pysera"
    fuse = _extract_cmd(plan, "fuse")
    assert "pyradiomics=" in fuse[fuse.index("--features") + 1]
    assert plan.steps[ids.index("merge")].args["features"] == "{outdir}/features.csv"


def test_plan_unknown_template():
    with pytest.raises(ValueError):
        build_workflow_plan(FeatureExtractionConfig(), template="nope", **KW)
