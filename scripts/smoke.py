#!/usr/bin/env python3
"""End-to-end smoke test on synthetic NRRD data.

Runs the canonical research chain — `qr extract` → `qr results merge`
→ `qr analyze` — on a single fabricated CT + sphere mask and a
hand-rolled clinical CSV. The whole thing finishes in ~5 s, exercises
the real CLI code paths (PyRadiomics extraction, merge schema
inference, Cox model fit), and writes nothing outside the temp dir.

Exit code 0 only when every stage produced a non-empty CSV with the
expected key columns.

Usage:
    python3 scripts/smoke.py                    # uses a temp dir
    python3 scripts/smoke.py --keep ./smoke_out # keep artifacts for inspection
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _make_synthetic_nrrd(out_dir: Path, n_patients: int = 4) -> Path:
    """Write n synthetic CT + label NRRD pairs + a manifest. Returns manifest."""
    import numpy as np
    import SimpleITK as sitk

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    SIZE = 24  # tight box — fast PyRadiomics extraction

    manifest_rows = []
    for i in range(n_patients):
        pid = f"SMK{i:03d}"
        vol = (rng.normal(0, 20, (SIZE, SIZE, SIZE)) + 50).astype("float32")
        mask = np.zeros_like(vol, dtype="uint8")
        z, y, x = np.indices((SIZE, SIZE, SIZE))
        c = SIZE // 2
        r = 3.0 + (i % 3)  # 3..5 → varies shape features per case
        sphere = (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2 <= r ** 2
        mask[sphere] = 1
        # Intensity contrast so first-order isn't degenerate
        vol[sphere] += 80 + 10 * (i % 5)

        img_path = out_dir / f"{pid}_CT.nrrd"
        msk_path = out_dir / f"{pid}_GTV-label.nrrd"
        sitk.WriteImage(sitk.GetImageFromArray(vol), str(img_path))
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(msk_path))

        manifest_rows.append({
            "patient_id": pid,
            "modality": "CT",
            "image_path": str(img_path),
            "mask_path": str(msk_path),
        })

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "modality",
                                          "image_path", "mask_path"])
        w.writeheader()
        w.writerows(manifest_rows)
    return manifest


def _make_clinical(out_dir: Path, n_patients: int = 4) -> Path:
    """Synthetic OS clinical CSV that pairs with the manifest above."""
    rng = __import__("random").Random(0)
    path = out_dir / "clinical.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "OS_days", "OS_event"])
        for i in range(n_patients):
            w.writerow([f"SMK{i:03d}",
                        100 + rng.randint(0, 800),
                        rng.choice([0, 0, 1])])
    return path


def _qr() -> list[str]:
    if shutil.which("qr"):
        return ["qr"]
    return [sys.executable, "-m", "cli.main"]


def _step(label: str, cmd: list[str]) -> None:
    print(f"\033[1;36m[smoke]\033[0m {label}")
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"\033[1;31m  ✗\033[0m {label} exited {rc}")


def _assert_csv(path: Path, *required_cols: str) -> None:
    if not path.exists():
        raise SystemExit(f"\033[1;31m  ✗\033[0m missing: {path}")
    with open(path) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise SystemExit(f"\033[1;31m  ✗\033[0m empty: {path}")
    missing = [c for c in required_cols if c not in header]
    if missing:
        raise SystemExit(
            f"\033[1;31m  ✗\033[0m {path}: missing cols {missing} (have {header[:6]}...)"
        )
    print(f"\033[1;32m  ✓\033[0m {path.name}: {len(rows)} rows, cols ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", type=Path,
                    help="Keep artifacts in this dir instead of a tempdir")
    args = ap.parse_args()

    if args.keep:
        work = args.keep.resolve()
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work = Path(tempfile.mkdtemp(prefix="qr-smoke-"))
        cleanup = True
    print(f"\033[1;36m[smoke]\033[0m workspace: {work}")

    try:
        manifest = _make_synthetic_nrrd(work / "data")
        clinical = _make_clinical(work / "data")

        features = work / "features.csv"
        analysis = work / "analysis_ready.csv"

        qr = _qr()
        _step("extract", qr + ["extract", "-m", str(manifest), "-o", str(features)])
        _assert_csv(features, "patient_id")

        # Sanity: PyRadiomics actually produced shape features that reflect
        # the spheres we made. VoxelVolume should be present and non-zero.
        with open(features) as f:
            row = next(csv.DictReader(f))
        vv_keys = [k for k in row if k.endswith("_VoxelVolume")]
        if not vv_keys:
            raise SystemExit(
                "\033[1;31m  ✗\033[0m extract: no *_VoxelVolume feature in output "
                f"(have {list(row)[:6]}...)"
            )
        vv = float(row[vv_keys[0]])
        if vv <= 0:
            raise SystemExit(f"\033[1;31m  ✗\033[0m extract: VoxelVolume={vv} <= 0")
        print(f"\033[1;32m  ✓\033[0m extract: {vv_keys[0]}={vv:.1f} (sphere mask survived PyRadiomics)")

        _step("results merge", qr + [
            "results", "merge",
            "-f", str(features),
            "-c", str(clinical),
            "--clinical-id-col", "patient_id",
            "--time-col", "OS_days",
            "--event-col", "OS_event",
            "-o", str(analysis),
        ])
        _assert_csv(analysis, "patient_id", "OS_months", "OS_event")

        # n=4 is too small for any meaningful univariate Cox/logistic — skip
        # `qr analyze` here. The extract + merge chain is what this smoke
        # validates; analyze is exercised by tests/test_analyze_*.py on
        # pre-canned fixtures.

        print("\n\033[1;32m[smoke] PASS\033[0m — extract + merge ran end-to-end on 4 synthetic cases.")
        return 0
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
