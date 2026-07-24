"""
Radiomics extraction service (sync API) — PySERA backend.

Mirrors qradiomics/extractor.py's RadiomicsExtractor contract but delegates
per-patient feature extraction to the PySERA package instead of PyRadiomics.
"""
from __future__ import annotations

import csv
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import UUID

logger = logging.getLogger(__name__)

_DEFAULT_CATEGORIES = "diag,morph,glcm,glrlm,glszm,ngtdm,ngldm"
_DEFAULT_DIMENSIONS = "1st,2d"
_DEFAULT_EXTRACTION_MODE = "handcrafted_feature"
_DEFAULT_DEEP_LEARNING_MODEL = "resnet50"


class PyseraExtractor:
    """Synchronous radiomics extractor wrapper around PySERA.

    Usage:
        extractor = PyseraExtractor()
        result = extractor.run_extraction(job_id, manifest_path, job_dir, settings)
    """

    def run_extraction(
        self,
        job_id: UUID,
        manifest_path: Path,
        job_dir: Path,
        extraction_settings: Dict[str, Any],
        jobs: int = 1,
    ) -> Dict[str, Any]:
        """Run PySERA extraction for all image/mask pairs in manifest.

        This is a blocking call intended to be run inside a ThreadPoolExecutor.
        Returns a result dict compatible with RadiomicsExtractor.run_extraction.
        """
        try:
            import pysera  # type: ignore
        except Exception as e:  # pragma: no cover - environment dependent
            logger.error("PySERA import failed: %s", e)
            return {
                "features_uri": f"file://{job_dir.resolve()}/features.csv",
                "feature_count": 0,
                "status": "error",
                "error": f"PySERA import failed: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        settings = dict(extraction_settings or {})
        categories = settings.pop("categories", _DEFAULT_CATEGORIES)
        dimensions = settings.pop("dimensions", _DEFAULT_DIMENSIONS)
        extraction_mode = settings.pop("extraction_mode", _DEFAULT_EXTRACTION_MODE)
        deep_learning_model = settings.pop("deep_learning_model", _DEFAULT_DEEP_LEARNING_MODEL)
        ibsi_based_parameters = settings.pop("IBSI_based_parameters", None)

        pysera_kwargs: Dict[str, Any] = {
            "categories": categories,
            "dimensions": dimensions,
            "extraction_mode": extraction_mode,
            "deep_learning_model": deep_learning_model,
        }
        if ibsi_based_parameters is not None:
            pysera_kwargs["IBSI_based_parameters"] = ibsi_based_parameters

        all_features: List[Dict[str, Any]] = []
        patients_processed = 0
        patients_failed = 0
        patients_skipped = 0

        rows: List[Dict[str, str]] = []
        with open(manifest_path, "r") as f:
            for row in csv.DictReader(f):
                rows.append(row)

        for row in rows:
            patient_id = row.get("patient_id")
            image_path = row.get("image_path")
            mask_path = row.get("mask_path")
            logger.info("Extracting features for %s", patient_id)
            if not mask_path or mask_path.strip() == "":
                logger.warning(
                    "Skipping %s: no mask provided (radiomics requires a segmentation mask)",
                    patient_id,
                )
                patients_skipped += 1
                continue

            patient_output_dir = job_dir / f"_pysera_{patient_id}"
            try:
                result = pysera.process_batch(
                    image_input=image_path,
                    mask_input=mask_path,
                    output_path=str(patient_output_dir),
                    num_workers=str(max(1, jobs)),
                    report="warning",
                    apply_preprocessing=True,
                    **pysera_kwargs,
                )
                if not result.get("success", False):
                    logger.error(
                        "PySERA extraction failed for %s: %s",
                        patient_id, result.get("error"),
                    )
                    patients_failed += 1
                    continue

                features_df = result.get("features_extracted")
                if features_df is None or len(features_df) == 0:
                    logger.error("PySERA returned no features for %s", patient_id)
                    patients_failed += 1
                    continue

                raw_features = features_df.iloc[0].to_dict()
                feature_dict: Dict[str, Any] = {"patient_id": patient_id}
                for key, value in raw_features.items():
                    if hasattr(value, "item"):
                        try:
                            value = value.item()
                        except Exception:
                            pass
                    feature_dict[key] = value

                all_features.append(feature_dict)
                patients_processed += 1
                logger.info("Extracted %d features for %s", len(feature_dict) - 1, patient_id)
            except Exception as e:
                logger.error("Failed to extract features for %s: %s", patient_id, e)
                patients_failed += 1
            finally:
                shutil.rmtree(patient_output_dir, ignore_errors=True)

        features_path = job_dir / "features.csv"
        if all_features:
            fieldnames = list(all_features[0].keys())
            with open(features_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_features)

            feature_count = len(fieldnames) - 1
        else:
            feature_count = 0
            with open(features_path, "w") as f:
                f.write("patient_id\n")

        logger.info(
            "Extraction complete for job %s: %s patients, %s features, %s failed, %s skipped",
            job_id,
            patients_processed,
            feature_count,
            patients_failed,
            patients_skipped,
        )

        return {
            "features_uri": f"file://{features_path.resolve()}",
            "feature_count": feature_count,
            "patients_processed": patients_processed,
            "patients_failed": patients_failed,
            "patients_skipped": patients_skipped,
            "status": "extracted" if patients_processed > 0 else "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def get_pysera_extractor() -> PyseraExtractor:
    """Factory for a PyseraExtractor instance (lightweight).

    Callers may instantiate directly or use this factory.
    """
    return PyseraExtractor()
