"""
Radiomics extraction service (sync API).

This module encapsulates PyRadiomics extraction logic so the JobWorker
can delegate extraction without holding PyRadiomics-specific code inline.
The API is synchronous so it can be executed inside a threadpool by callers.
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import UUID

logger = logging.getLogger(__name__)


class RadiomicsExtractor:
    """Synchronous radiomics extractor wrapper around PyRadiomics.

    Usage:
        extractor = RadiomicsExtractor()
        result = extractor.run_extraction(job_id, manifest_path, job_dir, settings)
    """

    def run_extraction(
        self,
        job_id: UUID,
        manifest_path: Path,
        job_dir: Path,
        extraction_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run PyRadiomics extraction for all image/mask pairs in manifest.

        This is a blocking call intended to be run inside a ThreadPoolExecutor.
        Returns a result dict compatible with the previous worker implementation.
        """
        try:
            from radiomics import featureextractor  # type: ignore
        except Exception as e:  # pragma: no cover - environment dependent
            logger.error("PyRadiomics import failed: %s", e)
            return {
                "features_uri": f"file://{job_dir.resolve()}/features.csv",
                "feature_count": 0,
                "status": "error",
                "error": f"PyRadiomics import failed: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Default extractor params (kept consistent with previous implementation)
        extractor_params: Dict[str, Any] = {
            "binWidth": 25,
            "resampledPixelSpacing": None,
            "interpolator": "sitkBSpline",
            "verbose": False,
            "geometryTolerance": 1e-3,
            "correctMask": True,
        }

        # Ensure geometryTolerance numeric
        if "geometryTolerance" in extraction_settings:
            try:
                extraction_settings["geometryTolerance"] = float(
                    extraction_settings["geometryTolerance"]
                )
            except (ValueError, TypeError):
                pass

        extractor_params.update(extraction_settings or {})

        image_types = extractor_params.pop("image_types", ["Original"])

        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(**extractor_params)
            extractor.disableAllImageTypes()
            for img_type in image_types:
                extractor.enableImageTypeByName(img_type)
        except Exception as e:
            logger.error("Failed to initialize PyRadiomics extractor: %s", e)
            return {
                "features_uri": f"file://{job_dir.resolve()}/features.csv",
                "feature_count": 0,
                "status": "error",
                "error": f"Extractor init failed: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        all_features: List[Dict[str, Any]] = []
        patients_processed = 0
        patients_failed = 0
        patients_skipped = 0

        # Read manifest and execute extraction per patient
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row.get("patient_id")
                image_path = row.get("image_path")
                mask_path = row.get("mask_path")

                logger.info("Extracting features for %s", patient_id)
                try:
                    if not mask_path or mask_path.strip() == "":
                        logger.warning(
                            "Skipping %s: no mask provided (radiomics requires a segmentation mask)",
                            patient_id,
                        )
                        patients_skipped += 1
                        continue

                    features = extractor.execute(image_path, mask_path)

                    feature_dict: Dict[str, Any] = {"patient_id": patient_id}
                    for key, value in features.items():
                        if key.startswith("diagnostics_"):
                            continue
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


def get_radiomics_extractor() -> RadiomicsExtractor:
    """Factory for a RadiomicsExtractor instance (lightweight).

    Callers may instantiate directly or use this factory.
    """
    return RadiomicsExtractor()
