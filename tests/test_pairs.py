"""Pair manifest contract: schema, validator, shared reader, `qr select validate`."""

import pandas as pd
import pytest
from click.testing import CliRunner

from qradiomics import pairs as pm
from qradiomics.cli.commands.select_cmd import select

SHA = "a" * 64


def make_pairs(n_cases=2, labels=("manual", "auto")):
    rows = []
    for c in range(n_cases):
        for lab in labels:
            cid = f"C{c + 1}"
            rows.append(
                {
                    "cohort": "synth",
                    "case_id": cid,
                    "pair_id": f"{cid}_CT_{lab}",
                    "image_key": "CT",
                    "label_key": lab,
                    "series_uid": f"1.2.{c}",
                    "series_dir": f"/data/{cid}/CT",
                    "rtstruct_path": f"/data/{cid}/RS.{lab}.dcm",
                    "rtstruct_sha": SHA,
                    "roi_candidates": "Heart|heart|HEART",
                    "resample_mm": "1.0",
                    "status": "ok",
                    "reason": "",
                }
            )
    return pd.DataFrame(rows, columns=list(pm.PAIR_COLUMNS))


def test_valid_manifest():
    assert pm.validate_pairs(make_pairs()) == []


def test_skipped_row_needs_only_reason():
    df = make_pairs()
    df.loc[0, ["status", "reason", "series_dir", "rtstruct_sha", "roi_candidates"]] = [
        "skipped",
        "4D series",
        "",
        "",
        "",
    ]
    assert pm.validate_pairs(df) == []
    df.loc[0, "reason"] = ""
    assert any("requires a reason" in e for e in pm.validate_pairs(df))


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda d: d.drop(columns=["rtstruct_sha"]), "missing column"),
        (lambda d: d.assign(pair_id="dup"), "duplicate pair_id"),
        (lambda d: d.assign(case_id="bad id/x"), "must match"),
        (lambda d: d.assign(status="maybe"), "not in"),
        (lambda d: d.assign(rtstruct_sha="xyz"), "64 hex"),
        (lambda d: d.assign(resample_mm="-1"), "positive number"),
        (lambda d: d.assign(resample_mm="abc"), "positive number"),
        (lambda d: d.assign(roi_candidates=""), "requires roi_candidates"),
        (lambda d: d.assign(series_uid=""), "requires series_uid"),
        (lambda d: d.iloc[0:0], "no rows"),
    ],
)
def test_invalid_manifests(mutate, needle):
    errs = pm.validate_pairs(mutate(make_pairs()))
    assert errs and any(needle in e for e in errs), errs


def test_empty_resample_is_valid():
    assert pm.validate_pairs(make_pairs().assign(resample_mm="")) == []


def test_check_paths(tmp_path):
    assert any("does not exist" in e for e in pm.validate_pairs(make_pairs(), check_paths=True))
    d = tmp_path / "ser"
    d.mkdir()
    f = tmp_path / "rs.dcm"
    f.write_bytes(b"x")
    df = make_pairs(1, ("manual",)).assign(series_dir=str(d), rtstruct_path=str(f))
    assert pm.validate_pairs(df, check_paths=True) == []


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_roundtrip_and_select_rows(tmp_path, suffix):
    p = tmp_path / f"pairs{suffix}"
    pm.write_manifest(make_pairs(), p)
    df = pm.load_pairs(p)
    assert len(df) == 4 and df["resample_mm"].tolist() == ["1.0"] * 4
    assert pm.select_rows(df, only="C1_CT_auto")["pair_id"].tolist() == ["C1_CT_auto"]
    assert pm.select_rows(df, only_case="C2")["pair_id"].tolist() == ["C2_CT_manual", "C2_CT_auto"]
    with pytest.raises(ValueError, match="no such pair_id"):
        pm.select_rows(df, only="nope")
    with pytest.raises(ValueError, match="No manifest rows for --case"):
        pm.select_rows(df, only_case="nope")


def test_select_rows_legacy_patient_manifest():
    df = pd.DataFrame({"patient_id": ["P1", "P2"], "image_path": ["a", "b"]})
    assert pm.select_rows(df, only="P2")["image_path"].tolist() == ["b"]
    assert pm.select_rows(df, only_case="P1")["image_path"].tolist() == ["a"]


def test_load_pairs_raises(tmp_path):
    p = tmp_path / "bad.csv"
    make_pairs().assign(status="maybe").to_csv(p, index=False)
    with pytest.raises(pm.PairManifestError):
        pm.load_pairs(p)


def test_split_roi_candidates():
    assert pm.split_roi_candidates(" Heart | heart||HEART ") == ["Heart", "heart", "HEART"]


def test_cli_validate(tmp_path):
    good, bad = tmp_path / "g.csv", tmp_path / "b.csv"
    make_pairs().to_csv(good, index=False)
    make_pairs().assign(pair_id="x").to_csv(bad, index=False)
    r = CliRunner()
    ok = r.invoke(select, ["validate", str(good)])
    assert ok.exit_code == 0 and "OK: 4 pair(s), 2 case(s)" in ok.output
    ko = r.invoke(select, ["validate", str(bad)])
    assert ko.exit_code == 1 and "duplicate pair_id" in ko.output


def test_parquet_without_engine_raises_actionable_error(tmp_path, monkeypatch):
    """A missing pyarrow must surface an install hint, not pandas' raw ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("No module named 'pyarrow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = pd.DataFrame({"a": ["1"]})
    for call in (
        lambda: pm.write_manifest(df, tmp_path / "m.parquet"),
        lambda: pm.read_manifest(tmp_path / "m.parquet"),
    ):
        with pytest.raises(ImportError, match=r"pip install .*pyarrow|pyarrow>=14"):
            call()
