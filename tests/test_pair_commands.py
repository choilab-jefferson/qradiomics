"""Pair-aware convert / preprocess / extract / results merge (shared manifest reader)."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from click.testing import CliRunner

from qradiomics import pairs as pm
from qradiomics.cli.commands.convert import _resolve_roi, convert
from qradiomics.cli.commands.extract import extract
from qradiomics.cli.commands.preprocess import preprocess
from qradiomics.cli.commands.results import results
from tests.test_pairs import make_pairs

# ─── ROI fallback ────────────────────────────────────────────────────────────


def test_resolve_roi_candidates_first_match_wins():
    names = ["Lung", "heart", "Heart", "Cord"]
    assert _resolve_roi(names, candidates=["Heart", "heart"]) == ("Heart", "candidate")
    assert _resolve_roi(names, candidates=["heart", "Heart"]) == ("heart", "candidate")
    assert _resolve_roi(["HEART"], candidates=["Heart"]) == ("HEART", "candidate-ci")
    assert _resolve_roi(["Cord", "Heart"], candidates=["ROI-1", "heart"]) == (
        "Heart",
        "candidate-ci",
    )


def test_resolve_roi_fails_when_no_candidate_matches():
    with pytest.raises(ValueError, match="none of the ROI candidates"):
        _resolve_roi(["Lung", "Cord", "Esophagus"], candidates=["Heart", "heart"])


def test_resolve_roi_plain_modes():
    assert _resolve_roi(["A", "B"]) == ("A", "default-first")
    assert _resolve_roi(["A", "B"], roi="B") == ("B", "exact")
    assert _resolve_roi(["Gtv"], roi="GTV") == ("Gtv", "ci")
    with pytest.raises(ValueError, match="not in structure set"):
        _resolve_roi(["A"], roi="Z")


# ─── convert from-manifest (pair mode), subprocess stubbed ───────────────────


def _stub_qr(monkeypatch, calls, fail_pairs=(), rois=("Heart",)):
    """Stand in for `python -m qradiomics.cli.main convert ...` subprocesses."""

    class R:
        returncode, stderr = 0, b""

    def run(argv, check=False, capture_output=False, **kw):
        a = list(argv)
        calls.append(a)
        verb = a[a.index("convert") + 1]
        out = Path(a[a.index("--output") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        if verb == "rtstruct":
            cands = a[a.index("--roi-candidates") + 1].split("|")
            if out.name.removesuffix("_label.nrrd") in fail_pairs or not set(cands) & set(rois):
                raise subprocess.CalledProcessError(1, a, stderr=b"ROI candidates not found")
            used = next(c for c in cands if c in rois)
            Path(a[a.index("--roi-record") + 1]).write_text(json.dumps({"roi_used": used}))
        out.write_text("nrrd")
        return R()

    monkeypatch.setattr(subprocess, "run", run)


def _pair_file(tmp_path, df=None, suffix=".parquet"):
    p = tmp_path / f"pairs{suffix}"
    pm.write_manifest(make_pairs(2, ("manual", "auto")) if df is None else df, p)
    return p


def test_from_manifest_pair_image_then_label(tmp_path, monkeypatch):
    calls = []
    _stub_qr(monkeypatch, calls)
    man, nrrd = _pair_file(tmp_path), tmp_path / "nrrd"
    r = CliRunner().invoke(
        convert,
        ["from-manifest", "-m", str(man), "-d", str(nrrd), "--only-case", "C1", "--part", "image",
         "-o", str(tmp_path / "img.parquet")],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    img = pm.read_manifest(tmp_path / "img.parquet")
    assert len(calls) == 1, "image converted once per (case, image_key), not per pair"
    assert img["modality"].tolist() == ["CT"] and img["mask_path"].tolist() == [""]
    r = CliRunner().invoke(
        convert,
        ["from-manifest", "-m", str(man), "-d", str(nrrd), "--only", "C1_CT_auto", "--part", "label",
         "-o", str(tmp_path / "lab.parquet")],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    lab = pm.read_manifest(tmp_path / "lab.parquet")
    assert lab["pair_id"].tolist() == ["C1_CT_auto"] and lab["roi_used"].tolist() == ["Heart"]
    assert lab["patient_id"].tolist() == ["C1"] and lab["resample_mm"].tolist() == ["1.0"]
    assert lab["mask_path"][0].endswith("C1/C1_CT_auto_label.nrrd")
    assert lab["image_path"][0].endswith("C1/CT.nrrd")
    rt = [c for c in calls if "rtstruct" in c][0]
    assert rt[rt.index("--roi-candidates") + 1] == "Heart|heart|HEART"


def test_from_manifest_label_needs_image_part_first(tmp_path, monkeypatch):
    _stub_qr(monkeypatch, [])
    r = CliRunner().invoke(
        convert,
        ["from-manifest", "-m", str(_pair_file(tmp_path)), "-d", str(tmp_path / "n"),
         "--only", "C1_CT_manual", "--part", "label", "-o", str(tmp_path / "o.parquet")],
    )  # fmt: skip
    assert r.exit_code == 1 and "run --part image first" in r.output


def test_from_manifest_no_roi_candidate_fails_loud(tmp_path, monkeypatch):
    _stub_qr(monkeypatch, [], rois=("Lung",))
    man, nrrd = _pair_file(tmp_path), tmp_path / "n"
    CliRunner().invoke(
        convert,
        ["from-manifest", "-m", str(man), "-d", str(nrrd), "--only-case", "C1", "--part", "image",
         "-o", str(tmp_path / "i.parquet")],
    )  # fmt: skip
    r = CliRunner().invoke(
        convert,
        ["from-manifest", "-m", str(man), "-d", str(nrrd), "--only", "C1_CT_manual",
         "--part", "label", "-o", str(tmp_path / "o.parquet")],
    )  # fmt: skip
    assert r.exit_code == 1 and "FAILED pair C1_CT_manual" in r.output
    assert len(pm.read_manifest(tmp_path / "o.parquet")) == 0


def test_from_manifest_skips_non_ok_and_rejects_unknown_only(tmp_path, monkeypatch):
    calls = []
    _stub_qr(monkeypatch, calls)
    df = make_pairs(1, ("manual", "auto"))
    df.loc[1, ["status", "reason", "rtstruct_sha"]] = ["failed", "no RTSTRUCT", ""]
    man = _pair_file(tmp_path, df)
    r = CliRunner().invoke(
        convert,
        ["from-manifest", "-m", str(man), "-d", str(tmp_path / "n"), "--part", "image",
         "-o", str(tmp_path / "o.parquet")],
    )  # fmt: skip
    assert r.exit_code == 0 and len(calls) == 1
    r = CliRunner().invoke(
        convert,
        ["from-manifest", "-m", str(man), "-d", str(tmp_path / "n"), "--only", "nope",
         "-o", str(tmp_path / "o.parquet")],
    )  # fmt: skip
    assert r.exit_code != 0 and "no such pair_id" in r.output


# ─── preprocess ──────────────────────────────────────────────────────────────


def _volume(path, spacing, shape=(24, 24, 24), box=(6, 18)):
    arr = np.random.RandomState(0).randint(0, 200, shape).astype("int16")
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(path))
    m = np.zeros(shape, dtype="uint8")
    m[box[0] : box[1], box[0] : box[1], box[0] : box[1]] = 1
    mk = sitk.GetImageFromArray(m)
    mk.SetSpacing(spacing)
    return mk


def _nrrd_manifest(tmp_path):
    rows = []
    for pid, mod, sp, rs in (("C1_CT_manual", "CT", 1.0, "2.0"), ("C1_PT_manual", "PT", 2.0, "4.0"),
                             ("C2_CT_manual", "CT", 1.0, "")):  # fmt: skip
        img, msk = tmp_path / f"{pid}.nrrd", tmp_path / f"{pid}_m.nrrd"
        sitk.WriteImage(_volume(img, (sp,) * 3), str(msk))
        rows.append({"patient_id": pid.split("_")[0], "pair_id": pid, "modality": mod,
                     "image_path": str(img), "mask_path": str(msk), "roi_used": "Heart",
                     "resample_mm": rs})  # fmt: skip
    p = tmp_path / "nm.parquet"
    pm.write_manifest(pd.DataFrame(rows), p)
    return p


def test_preprocess_per_row_resample_and_only(tmp_path):
    man = _nrrd_manifest(tmp_path)
    out = tmp_path / "pre"
    r = CliRunner().invoke(preprocess, ["-m", str(man), "-o", str(out), "--pad-mm", "2",
                                        "--out-manifest", str(tmp_path / "pre.parquet")])  # fmt: skip
    assert r.exit_code == 0, r.output
    pre = pm.read_manifest(tmp_path / "pre.parquet").set_index("pair_id")
    sp = lambda pid: sitk.ReadImage(pre.loc[pid, "image_path"]).GetSpacing()  # noqa: E731
    assert sp("C1_CT_manual") == (2.0, 2.0, 2.0)
    assert sp("C1_PT_manual") == (4.0, 4.0, 4.0)
    assert sp("C2_CT_manual") == (1.0, 1.0, 1.0), "empty resample_mm -> no resample"
    assert (
        pre.loc["C1_PT_manual", "modality"] == "PT"
        and pre.loc["C1_PT_manual", "patient_id"] == "C1"
    )
    assert pre.loc["C1_CT_manual", "roi_used"] == "Heart"
    assert Path(pre.loc["C1_PT_manual", "image_path"]).name.endswith("_PT.nrrd")
    assert len(list(out.glob("*_CT.nrrd"))) == 2, "files keyed by pair_id: one case, several pairs"

    out2 = tmp_path / "pre2"
    r = CliRunner().invoke(preprocess, ["-m", str(man), "--only", "C2_CT_manual", "-o", str(out2),
                                        "--out-manifest", str(tmp_path / "p2.parquet")])  # fmt: skip
    assert r.exit_code == 0
    assert pm.read_manifest(tmp_path / "p2.parquet")["pair_id"].tolist() == ["C2_CT_manual"]
    bad = CliRunner().invoke(preprocess, ["-m", str(man), "--only", "zzz", "-o", str(out2)])
    assert bad.exit_code != 0 and "no such pair_id" in bad.output


def test_preprocess_cli_resample_default_and_failing_only(tmp_path):
    man = _nrrd_manifest(tmp_path)
    r = CliRunner().invoke(preprocess, ["-m", str(man), "--only", "C2_CT_manual", "--resample", "3",
                                        "-o", str(tmp_path / "p"), "--out-manifest", str(tmp_path / "m.csv")])  # fmt: skip
    assert r.exit_code == 0
    row = pd.read_csv(tmp_path / "m.csv").iloc[0]
    assert sitk.ReadImage(row["image_path"]).GetSpacing() == (3.0, 3.0, 3.0)
    df = pm.read_manifest(man)
    df.loc[df["pair_id"] == "C2_CT_manual", "mask_path"] = str(tmp_path / "missing.nrrd")
    pm.write_manifest(df, man)
    r = CliRunner().invoke(
        preprocess, ["-m", str(man), "--only", "C2_CT_manual", "-o", str(tmp_path / "p")]
    )
    assert r.exit_code == 1, "a failing single-pair preprocess must exit non-zero"


def test_preprocess_legacy_csv_manifest_unchanged(tmp_path):
    img, msk = tmp_path / "i.nrrd", tmp_path / "m.nrrd"
    sitk.WriteImage(_volume(img, (1.0,) * 3), str(msk))
    man = tmp_path / "legacy.csv"
    pd.DataFrame(
        [{"patient_id": "P1", "modality": "CT", "image_path": str(img), "mask_path": str(msk)}]
    ).to_csv(man, index=False)
    r = CliRunner().invoke(preprocess, ["-m", str(man), "-o", str(tmp_path / "o"), "--roi-name", "GTV",
                                        "--out-manifest", str(tmp_path / "o.csv")])  # fmt: skip
    assert r.exit_code == 0, r.output
    out = pd.read_csv(tmp_path / "o.csv")
    assert list(out.columns) == ["patient_id", "modality", "image_path", "mask_path"]
    assert out["image_path"][0].endswith("P1_cropped_CT.nrrd")


# ─── extract --only ──────────────────────────────────────────────────────────


def test_extract_only_pair_keeps_case_and_pair_id(tmp_path, monkeypatch):
    seen = {}

    def fake(engines, job_id, manifest_path, job_dir, extraction_settings, jobs):
        df = pd.read_csv(manifest_path)
        seen["manifest"] = df
        out = pd.DataFrame({"patient_id": df["patient_id"], "f1": [1.5] * len(df)})
        out.to_csv(Path(job_dir) / "features.csv", index=False)
        return {"status": "extracted", "patients_processed": len(df), "feature_count": 1}

    import importlib

    ex = importlib.import_module("qradiomics.cli.commands.extract")

    monkeypatch.setattr(ex, "run_multi_extraction", fake)
    man = _nrrd_manifest(tmp_path)
    o = tmp_path / "f.csv"
    r = CliRunner().invoke(extract, ["-m", str(man), "--only", "C1_PT_manual", "-o", str(o)])
    assert r.exit_code == 0, r.output
    assert seen["manifest"]["patient_id"].tolist() == ["C1_PT_manual"], "engines key on pair_id"
    out = pd.read_csv(o)
    assert out["pair_id"].tolist() == ["C1_PT_manual"] and out["patient_id"].tolist() == ["C1"]
    assert not list(tmp_path.glob(".extract_*")), "temp job dir cleaned"
    # whole manifest: one row per pair, case id restored
    r = CliRunner().invoke(extract, ["-m", str(man), "-o", str(o)])
    assert r.exit_code == 0
    out = pd.read_csv(o)
    assert sorted(out["pair_id"]) == ["C1_CT_manual", "C1_PT_manual", "C2_CT_manual"]
    assert out.set_index("pair_id").loc["C1_PT_manual", "patient_id"] == "C1"


def test_extract_only_unknown_or_empty_fails(tmp_path, monkeypatch):
    import importlib

    ex = importlib.import_module("qradiomics.cli.commands.extract")

    def empty(engines, job_id, manifest_path, job_dir, extraction_settings, jobs):
        (Path(job_dir) / "features.csv").write_text("patient_id\n")
        return {"status": "error", "patients_processed": 0}

    monkeypatch.setattr(ex, "run_multi_extraction", empty)
    man = _nrrd_manifest(tmp_path)
    r = CliRunner().invoke(
        extract, ["-m", str(man), "--only", "nope", "-o", str(tmp_path / "f.csv")]
    )
    assert r.exit_code == 1 and "no such pair_id" in r.output
    r = CliRunner().invoke(
        extract, ["-m", str(man), "--only", "C1_CT_manual", "-o", str(tmp_path / "f.csv")]
    )
    assert r.exit_code == 1 and "no features were produced" in r.output
    assert not (tmp_path / "f.csv").exists()


# ─── results merge: glob + coverage ──────────────────────────────────────────


def _feature_files(tmp_path, ids):
    for pid in ids:
        d = tmp_path / "pairs" / pid
        d.mkdir(parents=True)
        pd.DataFrame({"pair_id": [pid], "patient_id": [pid.split("_")[0]], "f1": [1.0]}).to_csv(
            d / "features.csv", index=False
        )


def _clinical(tmp_path, ids=("C1", "C2")):
    p = tmp_path / "clin.csv"
    pd.DataFrame({"patient_id": list(ids), "OS_days": [100, 200], "OS_event": [1, 0]}).to_csv(
        p, index=False
    )
    return p


def test_merge_tolerates_missing_pairs_and_writes_coverage(tmp_path):
    df = make_pairs(2, ("manual", "auto"))
    df.loc[3, ["status", "reason"]] = ["skipped", "4D series"]
    man = _pair_file(tmp_path, df)
    _feature_files(
        tmp_path, ["C1_CT_manual", "C2_CT_manual"]
    )  # C1_CT_auto failed, C2_CT_auto skipped
    out, cov = tmp_path / "ready.csv", tmp_path / "cov.parquet"
    r = CliRunner().invoke(
        results,
        ["merge", "-f", str(tmp_path / "pairs/*/features.csv"), "-c", str(_clinical(tmp_path)),
         "--pairs", str(man), "--coverage", str(cov), "-o", str(out)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    merged = pd.read_csv(out)
    assert sorted(merged["pair_id"]) == ["C1_CT_manual", "C2_CT_manual"]
    c = pd.read_parquet(cov).set_index("pair_id")
    assert c.loc["C1_CT_manual", "outcome"] == "merged"
    assert c.loc["C1_CT_auto", "outcome"] == "missing_features"
    assert (
        c.loc["C2_CT_auto", "outcome"] == "skipped" and c.loc["C2_CT_auto", "reason"] == "4D series"
    )
    assert "C1_CT_auto" in r.output and "missing_features" in r.output


def test_merge_coverage_flags_no_clinical(tmp_path):
    man = _pair_file(tmp_path, make_pairs(2, ("manual",)))
    _feature_files(tmp_path, ["C1_CT_manual", "C2_CT_manual"])
    cov = tmp_path / "cov.csv"
    r = CliRunner().invoke(
        results,
        ["merge", "-f", str(tmp_path / "pairs/*/features.csv"), "-c", str(_clinical(tmp_path, ("C1", "C9"))),
         "--pairs", str(man), "--coverage", str(cov), "-o", str(tmp_path / "o.csv")],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    c = pd.read_csv(cov).set_index("pair_id")
    assert c.loc["C2_CT_manual", "outcome"] == "no_clinical"


def test_merge_glob_without_matches_and_coverage_without_pairs(tmp_path):
    r = CliRunner().invoke(results, ["merge", "-f", str(tmp_path / "nope/*.csv"), "-c", str(_clinical(tmp_path)),
                                     "-o", str(tmp_path / "o.csv")])  # fmt: skip
    assert r.exit_code != 0 and "No features CSV matches" in r.output
    f = tmp_path / "f.csv"
    pd.DataFrame({"patient_id": ["C1"], "x": [1]}).to_csv(f, index=False)
    r = CliRunner().invoke(results, ["merge", "-f", str(f), "-c", str(_clinical(tmp_path)),
                                     "--coverage", str(tmp_path / "c.csv"), "-o", str(tmp_path / "o.csv")])  # fmt: skip
    assert r.exit_code != 0 and "--coverage needs --pairs" in r.output


def test_merge_comma_list_of_files(tmp_path):
    _feature_files(tmp_path, ["C1_CT_manual", "C2_CT_manual"])
    spec = ",".join(str(p) for p in sorted((tmp_path / "pairs").glob("*/features.csv")))
    r = CliRunner().invoke(
        results,
        ["merge", "-f", spec, "-c", str(_clinical(tmp_path)), "-o", str(tmp_path / "o.csv")],
    )
    assert r.exit_code == 0, r.output
    assert len(pd.read_csv(tmp_path / "o.csv")) == 2


def test_from_manifest_pt_pair_uses_case_ct_as_rtstruct_reference(tmp_path, monkeypatch):
    calls = []
    _stub_qr(monkeypatch, calls)
    df = make_pairs(1, ("manual",))
    pt = df.iloc[[0]].copy()
    pt["pair_id"], pt["image_key"], pt["series_dir"] = "C1_PT_manual", "PT", "/data/C1/PT"
    man = _pair_file(tmp_path, pd.concat([df, pt], ignore_index=True))
    nrrd = tmp_path / "n"
    for part, only in (("image", None), ("label", "C1_PT_manual")):
        args = [
            "from-manifest",
            "-m",
            str(man),
            "-d",
            str(nrrd),
            "--part",
            part,
            "-o",
            str(tmp_path / f"{part}.parquet"),
        ]
        r = CliRunner().invoke(convert, args + (["--only", only] if only else []))
        assert r.exit_code == 0, r.output
    rt = [c for c in calls if "rtstruct" in c][0]
    assert rt[rt.index("--dicom-dir") + 1] == "/data/C1/CT", "PT label is drawn on the CT series"
    lab = pm.read_manifest(tmp_path / "label.parquet")
    assert lab["modality"].tolist() == ["PT"] and lab["image_path"][0].endswith("C1/PT.nrrd")
