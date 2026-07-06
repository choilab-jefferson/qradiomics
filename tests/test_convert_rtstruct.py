"""Tests for `qr convert rtstruct` error handling.

Reproduces the real TCIA LUNG1-035 case, where the reference CT series is
not GDCM-readable: rt-utils raises a bare Exception deep inside
`RTStructBuilder.create_from`. The command must turn that into a clean,
non-zero exit rather than dumping a traceback in the middle of a
cohort-conversion loop.
"""

import pytest
from click.testing import CliRunner

pytest.importorskip("rt_utils")

from qradiomics.cli.commands.convert import convert


def test_rtstruct_unreadable_ct_fails_cleanly(tmp_path, monkeypatch):
    # Empty CT directory (no DICOM) + a stub RTSTRUCT file.
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    rt_file = tmp_path / "rs.dcm"
    rt_file.write_text("not-a-real-dicom")

    # Skip the real preamble fixer — the RTSTRUCT here is a stub.
    monkeypatch.setattr("qradiomics.io.dicom.fix_dicom_preamble", lambda *a, **k: True)

    result = CliRunner().invoke(
        convert,
        [
            "rtstruct",
            "-d", str(ct_dir),
            "-r", str(rt_file),
            "--roi", "GTV-1",
            "-o", str(tmp_path / "out.nrrd"),
        ],
    )

    assert result.exit_code == 1
    assert "Could not build RTSTRUCT" in result.output
    # A clean SystemExit, not an unhandled traceback.
    assert result.exception is None or isinstance(result.exception, SystemExit)
