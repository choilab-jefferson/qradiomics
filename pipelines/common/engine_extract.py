"""Shared engine-aware radiomics extraction helper for pipelines/lidc_idri/*.py.

The lidc_idri scripts call ``qradiomics.atomic.extract_features`` directly,
per (image, mask) pair, inside their own per-patient/per-nodule worker
functions -- there is no manifest CSV, since masks are rasterised or
region-grown in memory. ``qradiomics.extractor_registry.run_multi_extraction``
(used by ``qr extract --engine``) instead expects a manifest CSV of on-disk
image/mask paths. ``extract_features_multi`` bridges the two: the default
(pyradiomics-only) path is an untouched passthrough to
``qradiomics.atomic.extract_features`` -- byte-identical to every
lidc_idri script's pre-existing behaviour -- and any other resolved engine
list writes the pair to a scratch one-row manifest and routes through
``run_multi_extraction``, mirroring ``qradiomics/cli/commands/extract.py``'s
implementation pattern.

Note: ``params_file`` (a full custom PyRadiomics parameter YAML) only
applies to the default pyradiomics-only path. The registry engines take a
flat ``extraction_settings`` dict (``geometryTolerance``, ``binWidth``, ...),
not an arbitrary params YAML, so ``params_file`` is ignored whenever a
non-default engine is requested.
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4

import pandas as pd
import SimpleITK as sitk

from qradiomics.atomic import extract_features
from qradiomics.extractor_registry import resolve_engines, run_multi_extraction

__all__ = ["extract_features_multi"]


def extract_features_multi(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    engine: Optional[str] = None,
    params_file: Optional[Union[str, Path]] = None,
    label: int = 1,
    geometry_tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """Extract radiomics features for one (image, mask) pair via ``engine``.

    ``engine=None`` (or anything that resolves to ``["pyradiomics"]`` alone,
    the backward-compatible default) calls
    ``qradiomics.atomic.extract_features`` directly. Any other resolved
    engine list (a single non-pyradiomics engine, a comma-list, or ``"all"``)
    runs through ``qradiomics.extractor_registry.run_multi_extraction`` via a
    scratch one-row manifest and returns that row's (possibly
    ``<engine>_``-prefixed) feature dict.
    """
    engines = resolve_engines(engine)
    if engines == ["pyradiomics"]:
        return extract_features(
            image, mask, params_file=params_file, label=label,
            geometry_tolerance=geometry_tolerance,
        )

    with tempfile.TemporaryDirectory(prefix="qr_engine_extract_") as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "image.nrrd"
        mask_path = tmp_path / "mask.nrrd"
        sitk.WriteImage(image, str(image_path))
        sitk.WriteImage(mask, str(mask_path))

        manifest_path = tmp_path / "manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["patient_id", "modality", "image_path", "mask_path"]
            )
            writer.writeheader()
            writer.writerow({
                "patient_id": "row",
                "modality": "CT",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            })

        job_dir = tmp_path / "job"
        job_dir.mkdir()

        run_multi_extraction(
            engines=engines,
            job_id=uuid4(),
            manifest_path=manifest_path,
            job_dir=job_dir,
            extraction_settings={"geometryTolerance": geometry_tolerance},
            jobs=1,
        )

        features_csv = job_dir / "features.csv"
        if not features_csv.exists():
            raise RuntimeError(f"Engine(s) {engines} produced no features.csv")
        df = pd.read_csv(features_csv)
        if df.empty:
            raise RuntimeError(f"Engine(s) {engines} produced no feature rows")

        row = df.iloc[0].to_dict()
        row.pop("patient_id", None)
        return row
