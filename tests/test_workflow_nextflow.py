"""Nextflow backend of `qr workflow run`: plan -> params translation and runner argv."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from qradiomics import workflow as wf
from qradiomics.cli.commands.workflow import workflow
from tests.test_pairs import make_pairs


def _dicom_plan(tmp_path):
    return wf.template_dicom_to_ml(
        cohort_root=str(tmp_path / "coh"), clinical_csv="clin.csv", outdir=str(tmp_path / "out")
    )


def test_extract_level_per_template():
    """Option A: extract fans out per case where a case-level manifest exists
    (nrrd_survival, tcia_to_ml); DICOM-dir templates keep a cohort-level
    manifest step between convert and extract, so extract stays cohort-level."""
    kw = {"cohort_root": "c", "clinical_csv": "x.csv"}
    for fn, kwargs, per_case in (
        (wf.template_nrrd_survival, kw, True),
        (wf.template_tcia_to_ml, {"collection": "C", "clinical_csv": "x.csv"}, True),
        (wf.template_dicom_to_ml, kw, False),
        (wf.template_dicom_survival, kw, False),
    ):
        steps = {s.id: s for s in fn(**kwargs).steps}
        assert steps["extract"].per_patient is per_case, fn.__name__
        assert steps["merge"].per_patient is False
        if per_case:
            assert steps["extract"].args["case"] == "{patient_id}"
            assert steps["merge"].args["features"] == wf.CASE_FEATURES_GLOB


def test_library_plan_split_pre_case_post(tmp_path):
    plan = _dicom_plan(tmp_path)
    params = wf.plan_to_nextflow_params(plan, ["P1", "P2"], str(tmp_path / "out"))
    assert {"plan_name", "fingerprint", "cwd", "outdir", "pre", "cases", "post"} <= set(params)
    assert params["has_pairs"] is False and params["on_error"] == "finish"
    assert any("manifest-from-dir" in c for c in params["pre"])
    assert [c["id"] for c in params["cases"]] == ["P1", "P2"]
    # {patient_id} substituted per case, no placeholder left anywhere
    c1 = "\n".join(params["cases"][0]["cmds"])
    assert f"{tmp_path}/coh/P1/CT" in c1 and "P1_CT.nrrd" in c1 and "P2" not in c1
    assert "{patient_id}" not in json.dumps(params)
    assert "convert" in c1 and "dicom-series" in c1 and "rtstruct" in c1
    post = "\n".join(params["post"])
    assert " extract" in post and "merge" in post
    assert all(
        "extract" not in cmd for c in params["cases"] for cmd in c["cmds"]
    ), "cohort-level extract must run once, in post"
    json.dumps(params)  # serializable


def test_plan_without_per_patient_steps_is_all_pre(tmp_path):
    plan = wf.template_nrrd_survival(cohort_root="c", clinical_csv="x.csv", outdir=str(tmp_path))
    for st in plan.steps:
        st.per_patient = False
    params = wf.plan_to_nextflow_params(plan, [], str(tmp_path))
    assert params["cases"] == [] and params["post"] == []
    assert len(params["pre"]) >= 4


def test_cohort_step_between_per_patient_steps_rejected(tmp_path):
    plan = _dicom_plan(tmp_path)
    plan.steps[1].per_patient = True
    plan.steps[2].per_patient = False
    plan.steps[3].per_patient = True
    with pytest.raises(ValueError, match="between per-patient"):
        wf.plan_to_nextflow_params(plan, ["P1"], str(tmp_path))


def test_per_patient_plan_needs_cases(tmp_path):
    with pytest.raises(ValueError, match="case list is empty"):
        wf.plan_to_nextflow_params(_dicom_plan(tmp_path), [], str(tmp_path))


def test_read_case_ids(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text("patient_id,image_path\nB,x\nA,y\nB,z\n")
    assert wf.read_case_ids(str(m)) == ["B", "A"]


def test_runner_builds_nextflow_argv(tmp_path, monkeypatch):
    monkeypatch.setattr(wf.shutil, "which", lambda n: "/usr/bin/nextflow")
    calls = {}

    class P:
        returncode = 0

    def fake_run(argv, **kw):
        calls["argv"], calls["kw"] = argv, kw
        return P()

    monkeypatch.setattr(wf.subprocess, "run", fake_run)
    monkeypatch.setattr(wf, "_cohort_outputs", lambda plan: [])
    run_dir = tmp_path / "run"
    (tmp_path / "coh" / "P1").mkdir(parents=True)
    res = wf.WorkflowRunner(
        _dicom_plan(tmp_path), executor="nextflow", run_dir=str(run_dir), profile="slurm"
    ).run()
    assert res[0]["returncode"] == 0
    assert calls["argv"] == [
        "nextflow", "run", str(wf.nextflow_main_path()),
        "-params-file", str(run_dir / "params.json"),
        "-work-dir", str(run_dir / "work"),
        "-profile", "slurm", "-resume",
    ]  # fmt: skip
    assert calls["kw"]["cwd"] == run_dir
    params = json.loads((run_dir / "params.json").read_text())
    assert [c["id"] for c in params["cases"]] == ["P1"]
    assert wf.nextflow_main_path().is_file()


def test_no_resume_drops_flag(tmp_path):
    argv = wf.nextflow_argv(tmp_path / "p.json", tmp_path, resume=False)
    assert "-resume" not in argv


def test_missing_binary_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(wf.shutil, "which", lambda n: None)
    with pytest.raises(wf.NextflowNotFoundError, match="nextflow not on PATH"):
        wf.run_plan_nextflow(_dicom_plan(tmp_path))


def test_cli_default_executor_inline_and_nextflow_option():
    r = CliRunner().invoke(workflow, ["run", "--help"])
    assert r.exit_code == 0
    assert "[default: inline]" in r.output and "nextflow" in r.output


def test_cli_missing_nextflow_is_click_error(tmp_path, monkeypatch):
    monkeypatch.setattr(wf.shutil, "which", lambda n: None)
    pj = tmp_path / "plan.json"
    wf.save_plan(wf.template_nrrd_survival(cohort_root="c", clinical_csv="x.csv"), pj)
    r = CliRunner().invoke(workflow, ["run", str(pj), "-e", "nextflow"])
    assert r.exit_code != 0 and "nextflow not on PATH" in r.output


# ─── fix round: resolver, fingerprints, case list, quoting ───────────────────

import re
import shutil


def _lib_plans(tmp_path):
    kw = {
        "cohort_root": str(tmp_path / "coh"),
        "clinical_csv": "clin.csv",
        "outdir": str(tmp_path / "out"),
        "pair_manifest": "pairs.parquet",
    }
    import inspect

    plans = []
    for fn in wf.LIBRARY.values():
        sig = inspect.signature(fn).parameters
        extra = {"collection": "C"} if "collection" in sig else {}
        plans.append(fn(**{k: v for k, v in kw.items() if k in sig}, **extra))
    return plans


def test_all_library_templates_render_without_placeholders(tmp_path):
    for plan in _lib_plans(tmp_path):
        pairs = make_pairs(1, ("manual",)).assign(case_id="P1")
        params = wf.plan_to_nextflow_params(plan, ["P1"], str(tmp_path / "out"), pairs=pairs)
        blob = json.dumps({k: params[k] for k in ("pre", "cases", "post")})
        assert not re.search(r"\{[A-Za-z_]+\}", blob), (
            plan.name,
            re.findall(r"\{[A-Za-z_]+\}", blob),
        )


def test_nested_vars_resolve():
    assert wf._resolve("{a}/x", {"a": "{b}/y", "b": "/z"}) == "/z/y/x"


def test_params_have_fingerprints(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text("patient_id\nP1\n")
    plan = _dicom_plan(tmp_path)
    p1 = wf.plan_to_nextflow_params(plan, ["P1"], str(tmp_path), manifest=str(m))
    assert p1["fingerprint"] and all(c["fp"] for c in p1["cases"])
    m.write_text("patient_id\nP1\nP2\n")
    p2 = wf.plan_to_nextflow_params(plan, ["P1"], str(tmp_path), manifest=str(m))
    assert p1["fingerprint"] != p2["fingerprint"]
    assert p1["cases"][0]["fp"] == p2["cases"][0]["fp"]  # same commands, same case fp
    plan.steps[1].args["rtstruct"] = "other/RS.dcm" if "rtstruct" in plan.steps[1].args else "x"
    p3 = wf.plan_to_nextflow_params(plan, ["P1"], str(tmp_path), manifest=str(m))
    assert p3["cases"][0]["fp"] != p2["cases"][0]["fp"]


def test_nextflow_process_inputs_declare_fingerprint():
    src = wf.nextflow_main_path().read_text()
    assert src.count("fingerprint") >= 6 and "toSortedList()" in src


def test_resume_disabled_when_cohort_output_missing(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(wf.shutil, "which", lambda n: "/usr/bin/nextflow")
    seen = {}

    class P:
        returncode = 0

    monkeypatch.setattr(
        wf.subprocess, "run", lambda argv, **kw: seen.setdefault("argv", argv) and P()
    )
    (tmp_path / "coh" / "P1").mkdir(parents=True)
    wf.run_plan_nextflow(_dicom_plan(tmp_path), run_dir=str(tmp_path / "run"))
    assert "-resume" not in seen["argv"]
    assert "running without -resume" in caplog.text


def test_inline_and_nextflow_share_manifest_case_list(tmp_path):
    (tmp_path / "coh" / "D1").mkdir(parents=True)
    m = tmp_path / "m.csv"
    m.write_text("patient_id\nM1\nM2\n")
    plan = _dicom_plan(tmp_path)
    r = wf.WorkflowRunner(plan, dry_run=True, manifest=str(m)).run()
    assert sorted({x["patient_id"] for x in r if "patient_id" in x}) == ["M1", "M2"]
    r = wf.WorkflowRunner(plan, dry_run=True).run()
    assert {x["patient_id"] for x in r if "patient_id" in x} == {"D1"}


def test_case_id_validation_and_exclusion(tmp_path):
    coh = tmp_path / "coh"
    (coh / "P1").mkdir(parents=True)
    (coh / "nf_run").mkdir()
    plan = _dicom_plan(tmp_path)
    assert wf.resolve_cases(plan, exclude=[coh / "nf_run"]) == ["P1"]
    (coh / "Patient 01").mkdir()
    with pytest.raises(ValueError, match="unsafe case id"):
        wf.resolve_cases(plan)
    m = tmp_path / "m.csv"
    m.write_text("patient_id\nbad;id\n")
    with pytest.raises(ValueError, match="unsafe case id"):
        wf.resolve_cases(plan, manifest=str(m))


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="nextflow not installed")
def test_cwd_with_space_and_quote_runs(tmp_path, monkeypatch):
    cwd = tmp_path / "my dir's here"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    plan = wf.WorkflowPlan(
        name="q",
        vars={"cohort_root": str(tmp_path / "coh"), "outdir": str(tmp_path / "o")},
        steps=[wf.WorkflowStep(id="s", cmd="--version", args={})],
    )
    plan.steps[0].cmd = "--version"
    res = wf.run_plan_nextflow(plan, run_dir=str(tmp_path / "run"))
    assert res["returncode"] == 0


# ─── option A: per-case extract + shipped cohort plans ──────────────────────

PIPELINES = Path(__file__).resolve().parents[1] / "pipelines"


@pytest.mark.parametrize("cohort", ["acrin_heart", "lidc_idri", "lung1", "nsclc_cetuximab"])
def test_shipped_plan_yields_qr_case_cases(tmp_path, cohort):
    plan = wf.load_plan(PIPELINES / cohort / "plan.json")
    plan.vars["outdir"] = str(tmp_path / "out")
    (tmp_path / "out").mkdir()
    # case_manifest is written by the pre step `tcia manifest`; fake it.
    (tmp_path / "out" / "dicom_manifest.csv").write_text("patient_id,image_path\nP1,a\nP2,b\n")
    cases = wf.resolve_cases(plan)
    params = wf.plan_to_nextflow_params(plan, cases, str(tmp_path / "out"))
    assert len(params["cases"]) > 0 and [c["id"] for c in params["cases"]] == ["P1", "P2"]
    pre, post = "\n".join(params["pre"]), "\n".join(params["post"])
    case = "\n".join(params["cases"][0]["cmds"])
    assert "tcia download" in pre and "tcia manifest" in pre
    assert "from-manifest" in case and " extract " in case and "--case P1" in case
    assert "results merge" in post and "extract" not in post and "from-manifest" not in post


def test_extract_case_filter(tmp_path, monkeypatch):
    import importlib

    ex = importlib.import_module("qradiomics.cli.commands.extract")

    m = tmp_path / "m.csv"
    m.write_text(
        "patient_id,modality,image_path,mask_path\n" "P1,CT,i1,m1\nP2,CT,i2,m2\nP3,CT,i3,m3\n"
    )
    seen = {}

    def fake_run(engines, job_id, manifest_path, job_dir, extraction_settings, jobs):
        import pandas as pd

        rows = pd.read_csv(manifest_path)
        seen["ids"] = list(rows["patient_id"])
        rows[["patient_id"]].assign(f=1.0).to_csv(job_dir / "features.csv", index=False)
        return {"status": "ok"}

    monkeypatch.setattr(ex, "run_multi_extraction", fake_run)
    out = tmp_path / "cases" / "P2" / "features.csv"
    r = CliRunner().invoke(
        ex.extract, ["-m", str(m), "-o", str(out), "--case", "P2", "--case", "P3"]
    )
    assert r.exit_code == 0, r.output
    assert seen["ids"] == ["P2", "P3"]
    assert out.read_text().splitlines()[1:] == ["P2,1.0", "P3,1.0"]
    r = CliRunner().invoke(ex.extract, ["-m", str(m), "-o", str(out), "--case", "NOPE"])
    assert r.exit_code != 0 and "No manifest rows" in r.output


def test_merge_concatenates_per_case_glob(tmp_path):
    from qradiomics.cli.commands.results import results

    for pid in ("P1", "P2"):
        d = tmp_path / "cases" / pid
        d.mkdir(parents=True)
        (d / "features.csv").write_text(f"patient_id,f\n{pid},1.0\n")
    clin = tmp_path / "clin.csv"
    clin.write_text("patient_id,OS_days,OS_event\nP1,400,1\nP2,500,0\n")
    out = tmp_path / "ar.csv"
    r = CliRunner().invoke(
        results,
        [
            "merge",
            "-f",
            str(tmp_path / "cases" / "*" / "features.csv"),
            "-c",
            str(clin),
            "-o",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "Merged 2 patients" in r.output


def test_nextflow_runs_pre_inline_when_case_manifest_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(wf.shutil, "which", lambda n: "/usr/bin/nextflow")
    plan = wf.template_tcia_to_ml(collection="C", clinical_csv="x.csv", outdir=str(tmp_path / "o"))

    def fake_inline(self):
        dm = tmp_path / "o" / "dicom_manifest.csv"
        dm.parent.mkdir(parents=True, exist_ok=True)
        dm.write_text("patient_id\nP1\n")
        return [{"id": st.id, "returncode": 0} for st in self.plan.steps]

    monkeypatch.setattr(wf.WorkflowRunner, "_run_inline", fake_inline)

    class P:
        returncode = 0

    monkeypatch.setattr(wf.subprocess, "run", lambda argv, **kw: P())
    res = wf.run_plan_nextflow(plan, run_dir=str(tmp_path / "run"))
    params = json.loads(Path(res["params_file"]).read_text())
    assert params["pre"] == [] and [c["id"] for c in params["cases"]] == ["P1"]


# ─── hardening: pair fan-out, nested rendering, fingerprints, on_error ──────

DIGEST = "sha256:" + "b" * 64
IMG = f"reg.example/qr@{DIGEST}"


def _pair_plan(tmp_path):
    return wf.template_rt_pair_radiomics(
        pair_manifest=str(tmp_path / "pairs.parquet"),
        clinical_csv="clin.csv",
        outdir=str(tmp_path / "out"),
    )


def test_fanout_per_patient_alias():
    s = wf.WorkflowStep(id="a", cmd="x", per_patient=True)
    assert s.fanout == "case" and s.per_patient is True
    s = wf.WorkflowStep(id="a", cmd="x", fanout="case")
    assert s.per_patient is True
    s = wf.WorkflowStep(id="a", cmd="x", fanout="pair")
    assert s.per_patient is False
    assert wf.WorkflowStep(id="a", cmd="x").fanout is None
    with pytest.raises(ValueError, match="fanout"):
        wf.WorkflowStep(id="a", cmd="x", fanout="row")
    with pytest.raises(ValueError, match="conflicts"):
        wf.WorkflowStep(id="a", cmd="x", fanout="pair", per_patient=True)


def test_plan_roundtrip_keeps_fanout(tmp_path):
    plan = _pair_plan(tmp_path)
    again = wf.WorkflowPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert [s.fanout for s in again.steps] == [s.fanout for s in plan.steps]
    legacy = {"name": "n", "steps": [{"id": "a", "cmd": "x", "per_patient": True}]}
    assert wf.WorkflowPlan.from_dict(legacy).steps[0].fanout == "case"


def test_pair_template_nested_rendering(tmp_path):
    plan = _pair_plan(tmp_path)
    pairs = make_pairs(2, ("manual", "auto"))
    params = wf.plan_to_nextflow_params(
        plan, ["C1", "C2"], str(tmp_path / "out"), pairs=pairs, on_error="ignore", max_forks=3
    )
    assert params["has_pairs"] and params["on_error"] == "ignore" and params["max_forks"] == 3
    assert any("select" in c and "validate" in c for c in params["pre"])
    assert [c["id"] for c in params["cases"]] == ["C1", "C2"]
    c1 = params["cases"][0]
    assert [p["id"] for p in c1["pairs"]] == ["C1_CT_manual", "C1_CT_auto"]
    case_cmds = "\n".join(c1["cmds"])
    assert "--only-case C1" in case_cmds and "--part image" in case_cmds
    p = c1["pairs"][0]
    pair_cmds = "\n".join(p["cmds"])
    assert "--only C1_CT_manual" in pair_cmds
    assert "pairs/C1_CT_manual/features.csv" in pair_cmds and "C1_CT_auto" not in pair_cmds
    assert "workflow prov --pair-id C1_CT_manual" in p["cmds"][-1]
    assert '"$WD/C1_CT_manual.prov.json"' in p["cmds"][-1]
    assert p["outputs"] and all("C1_CT_manual" in o for o in p["outputs"])
    post = "\n".join(params["post"])
    assert "results merge" in post and "--coverage" in post and "--pairs" in post
    assert "extract" not in case_cmds
    assert not re.search(r"\{(case_id|pair_id|patient_id)\}", json.dumps(params))
    json.dumps(params)


def test_pair_plan_requires_pairs_and_order(tmp_path):
    plan = _pair_plan(tmp_path)
    with pytest.raises(ValueError, match="no pair rows"):
        wf.plan_to_nextflow_params(plan, ["C1"], str(tmp_path))
    # case step after a pair step is rejected
    plan.steps[2], plan.steps[1] = plan.steps[1], plan.steps[2]
    with pytest.raises(ValueError, match="case-level step after a pair-level"):
        wf.plan_to_nextflow_params(plan, ["C1"], str(tmp_path), pairs=make_pairs(1))


def test_pair_fanout_cannot_be_scaffolded(tmp_path):
    for name in ("shell", "nextflow", "prefect"):
        with pytest.raises(ValueError, match="fanout='pair'"):
            wf.SCAFFOLDERS[name](_pair_plan(tmp_path))


def test_fingerprint_changes():
    base = wf.input_fingerprint()
    assert base == wf.input_fingerprint()
    assert wf.input_fingerprint(container=IMG) != base
    assert wf.input_fingerprint(spec_hash="abc") != base
    assert wf.input_fingerprint(series_uid="1.2.3") != base
    assert wf.input_fingerprint(series_uid="1.2.3") != wf.input_fingerprint(series_uid="1.2.4")
    assert wf.input_fingerprint(rtstruct_sha="a" * 64) != wf.input_fingerprint(
        rtstruct_sha="b" * 64
    )


def test_fingerprint_tracks_tool_versions(monkeypatch):
    base = wf.input_fingerprint()
    v = wf.tool_versions()
    monkeypatch.setattr(wf, "tool_versions", lambda: {**v, "pyradiomics": "9.9.9"})
    assert wf.input_fingerprint() != base
    monkeypatch.setattr(wf, "tool_versions", lambda: {**v, "pysera": "9.9.9"})
    assert wf.input_fingerprint() != base
    assert {"qradiomics", "pyradiomics", "pysera"} <= set(v)


def test_spec_hash_follows_pattern_content(monkeypatch, tmp_path):
    plan = _dicom_plan(tmp_path)
    h = wf.spec_hash(plan)
    assert h == wf.spec_hash(plan)
    plan.vars["pattern"] = "lung-baseline" if "pattern" in plan.vars else None
    import qradiomics

    class T:
        def __init__(self, tag):
            self.tag = tag

        def model_dump_json(self):
            return self.tag

    class L:
        tag = "v1"

        def get_pattern(self, pid):
            return T(L.tag)

    monkeypatch.setattr(qradiomics, "PatternLoader", L)
    a = wf.spec_hash(plan)
    L.tag = "v2"
    assert wf.spec_hash(plan) != a


def test_pair_fp_isolated_to_edited_pair(tmp_path):
    plan = _pair_plan(tmp_path)
    pairs = make_pairs(2, ("manual",))
    a = wf.plan_to_nextflow_params(plan, ["C1", "C2"], str(tmp_path), pairs=pairs)
    pairs2 = pairs.copy()
    pairs2.loc[0, "rtstruct_sha"] = "c" * 64
    b = wf.plan_to_nextflow_params(plan, ["C1", "C2"], str(tmp_path), pairs=pairs2)
    fp = lambda p, i: p["cases"][i]["pairs"][0]["fp"]  # noqa: E731
    assert fp(a, 0) != fp(b, 0), "edited rtstruct_sha must invalidate that pair"
    assert fp(a, 1) == fp(b, 1), "other pair stays cached"
    assert a["cases"][0]["fp"] == b["cases"][0]["fp"], "rtstruct change does not redo image convert"
    assert a["fingerprint"] == b["fingerprint"], "pair manifest content is not in the global fp"
    pairs3 = pairs.copy()
    pairs3.loc[0, "series_uid"] = "9.9.9"
    c = wf.plan_to_nextflow_params(plan, ["C1", "C2"], str(tmp_path), pairs=pairs3)
    assert fp(a, 0) != fp(c, 0) and a["cases"][0]["fp"] != c["cases"][0]["fp"]
    assert a["cases"][1]["fp"] == c["cases"][1]["fp"]
    d = wf.plan_to_nextflow_params(plan, ["C1", "C2"], str(tmp_path), pairs=pairs, container=IMG)
    assert d["fingerprint"] != a["fingerprint"] and fp(d, 0) != fp(a, 0)
    assert d["container_digest"] == DIGEST


def test_container_must_be_digest_pinned():
    assert wf.container_digest(None) == ""
    assert wf.container_digest(IMG) == DIGEST
    for bad in ("reg.example/qr:latest", "qr", f"qr@sha256:{'z' * 64}"):
        with pytest.raises(ValueError, match="pinned by digest"):
            wf.container_digest(bad)


def test_container_flag_selects_docker_profile(tmp_path, monkeypatch):
    (tmp_path / "coh" / "P1").mkdir(parents=True)
    plan = _dicom_plan(tmp_path)
    r = wf.run_plan_nextflow(plan, run_dir=str(tmp_path / "r"), dry_run=True, container=IMG)
    assert r["cmd"][r["cmd"].index("-profile") + 1] == "local,docker"
    assert r["params"]["container"] == IMG
    # inside a container the interpreter is `python`, not the host path
    assert r["params"]["cases"][0]["cmds"][-1].startswith("python -m qradiomics.cli.main")
    r = wf.run_plan_nextflow(
        plan, run_dir=str(tmp_path / "r"), dry_run=True, container=IMG, profile="slurm,singularity"
    )
    assert r["cmd"][r["cmd"].index("-profile") + 1] == "slurm,singularity"
    with pytest.raises(ValueError, match="pinned by digest"):
        wf.run_plan_nextflow(plan, dry_run=True, container="qr:latest")


def test_pair_manifest_resolution(tmp_path):
    from qradiomics import pairs as pm

    df = make_pairs(2, ("manual", "auto"))
    df.loc[1, ["status", "reason"]] = ["failed", "no RTSTRUCT"]
    df.loc[2:3, ["status", "reason"]] = ["skipped", "4D"]
    m = tmp_path / "pairs.parquet"
    pm.write_manifest(df, m)
    assert wf.read_case_ids(str(m)) == ["C1"], "only cases owning an ok pair"
    assert wf.resolve_pairs(str(m))["pair_id"].tolist() == ["C1_CT_manual"]
    pm.write_manifest(df.assign(case_id="bad id"), m)
    with pytest.raises(Exception):
        wf.resolve_pairs(str(m))


def _fake_subprocess(monkeypatch, fail_on):
    calls = []

    class P:
        def __init__(self, rc):
            self.returncode, self.stderr = rc, "boom" if rc else ""

    def run(cmd, **kw):
        calls.append(cmd)
        return P(1 if any(f in " ".join(cmd) for f in fail_on) else 0)

    monkeypatch.setattr(wf.subprocess, "run", run)
    return calls


def _write_pairs(tmp_path):
    from qradiomics import pairs as pm

    m = tmp_path / "pairs.parquet"
    pm.write_manifest(make_pairs(2, ("manual", "auto")), m)
    return str(m)


def test_inline_pair_fanout_dry_run(tmp_path):
    m = _write_pairs(tmp_path)
    r = wf.WorkflowRunner(_pair_plan(tmp_path), dry_run=True, manifest=m).run()
    pair_ids = {x["pair_id"] for x in r if "pair_id" in x}
    assert pair_ids == {"C1_CT_manual", "C1_CT_auto", "C2_CT_manual", "C2_CT_auto"}
    ext = [x for x in r if x["id"] == "extract"]
    assert len(ext) == 4 and all("--only" in x["cmd"] for x in ext)
    with pytest.raises(ValueError, match="needs --manifest"):
        wf.WorkflowRunner(_pair_plan(tmp_path), dry_run=True).run()


def test_inline_on_error(tmp_path, monkeypatch):
    m = _write_pairs(tmp_path)
    monkeypatch.setenv("QR_INLINE_WORKERS", "1")
    calls = _fake_subprocess(monkeypatch, fail_on=["--only C1_CT_auto"])
    res = wf.WorkflowRunner(_pair_plan(tmp_path), manifest=m, on_error="ignore").run()
    assert not [
        c for c in calls if "extract" in c and "C1_CT_auto" in " ".join(c)
    ], "failed pair dropped"
    assert len([c for c in calls if "extract" in c]) == 3
    assert any(" ".join(c).find("results merge") >= 0 for c in map(list, calls)), "post still runs"
    assert [r for r in res if r.get("returncode") == 1 and r.get("pair_id") == "C1_CT_auto"]
    calls.clear()
    wf.WorkflowRunner(_pair_plan(tmp_path), manifest=m, on_error="finish").run()
    assert not any("results" in c and "merge" in c for c in calls), "finish stops the run"


def test_write_provenance(tmp_path):
    f = tmp_path / "f.csv"
    f.write_text("a,b\n1,2\n")
    rec = wf.write_provenance("P_1", [str(f)], str(tmp_path / "P_1.prov.json"), container=IMG)
    on_disk = json.loads((tmp_path / "P_1.prov.json").read_text())
    import hashlib

    assert on_disk == rec
    assert rec["outputs"][str(f)]["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    assert rec["container"] == IMG and {"qradiomics", "pyradiomics", "pysera"} <= set(
        rec["versions"]
    )
    with pytest.raises(FileNotFoundError):
        wf.write_provenance("P_1", [str(tmp_path / "nope")], str(tmp_path / "x.json"))


def test_cli_new_flags_and_prov():
    r = CliRunner().invoke(workflow, ["run", "--help"])
    for flag in ("--on-error", "--max-forks", "--container"):
        assert flag in r.output
    assert "rt_pair_radiomics" in CliRunner().invoke(workflow, ["templates"]).output


def test_cli_prov_command(tmp_path):
    f = tmp_path / "o.txt"
    f.write_text("x")
    out = tmp_path / "p.json"
    r = CliRunner().invoke(
        workflow, ["prov", "--pair-id", "P", "--file", str(f), "--output", str(out)]
    )
    assert r.exit_code == 0 and json.loads(out.read_text())["pair_id"] == "P"
    r = CliRunner().invoke(
        workflow, ["prov", "--pair-id", "P", "--file", str(tmp_path / "no"), "--output", str(out)]
    )
    assert r.exit_code != 0


def test_nextflow_sources_wire_pairs_and_policies():
    src = wf.nextflow_main_path().read_text()
    for needle in ("process QR_PAIR", "by: 0", "params.has_pairs", ".prov.json"):
        assert needle in src
    cfg = (wf.nextflow_main_path().parent / "nextflow.config").read_text()
    for needle in (
        "params.on_error",
        "params.max_forks",
        "docker",
        "singularity",
        "trace",
        "timeline",
        "dag",
        "report",
    ):
        assert needle in cfg


def test_manifest_select_filters_aliases_and_missing_files(tmp_path):
    from qradiomics.cli.commands.convert import convert

    root = tmp_path / "ds"
    (root / "img").mkdir(parents=True)
    for n in ("a_ct", "a_m", "b_ct", "b_m", "c_ct"):
        (root / "img" / f"{n}.nrrd").write_text("x")
    src = root / "nrrd.csv"
    src.write_text(
        "anon_pid,modality,image_nrrd,mask_nrrd\n"
        "A,preCT,img/a_ct.nrrd,img/a_m.nrrd\n"
        "A,postCT,img/a_ct.nrrd,img/a_m.nrrd\n"
        "B,preCT,img/b_ct.nrrd,img/b_m.nrrd\n"
        "C,preCT,img/c_ct.nrrd,img/c_m.nrrd\n"
    )
    out = tmp_path / "m.csv"
    res = CliRunner().invoke(
        convert,
        [
            "manifest-select",
            "-m",
            str(src),
            "--root",
            str(root),
            "--where",
            "modality=preCT",
            "--patients",
            "A,C",
            "--modality",
            "CT",
            "-o",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    lines = out.read_text().splitlines()
    assert lines[0] == "patient_id,modality,image_path,mask_path"
    assert lines[1] == f"A,CT,{root}/img/a_ct.nrrd,{root}/img/a_m.nrrd"
    assert len(lines) == 2  # postCT filtered, B not in --patients, C mask missing


_COHORT_PLANS = sorted(
    (Path(__file__).resolve().parents[1] / "nf-modules" / "cohorts").glob("*.json")
)


@pytest.mark.skipif(not _COHORT_PLANS, reason="nf-modules/cohorts is dev-only")
@pytest.mark.parametrize("path", _COHORT_PLANS, ids=lambda p: p.stem)
def test_cohort_plans_render_pre_and_per_case(path, tmp_path):
    plan = wf.load_plan(path)
    params = wf.plan_to_nextflow_params(plan, ["P1"], str(tmp_path))
    assert any(" convert manifest-select " in c for c in params["pre"])
    case = " ".join(params["cases"][0]["cmds"])
    assert " extract " in case and "--case P1" in case and "{" not in case
