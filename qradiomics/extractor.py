"""
Radiomics extraction service (sync API).

This module encapsulates PyRadiomics extraction logic so the JobWorker
can delegate extraction without holding PyRadiomics-specific code inline.
The API is synchronous so it can be executed inside a threadpool by callers.
"""
from __future__ import annotations

import csv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
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
        jobs: int = 1,
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

        # Read manifest rows up-front so we can decide between the sequential
        # in-process path (jobs=1, original semantics) and the parallel
        # process-pool path (jobs>1, ~jobs× wall-clock speedup on multi-core
        # boxes since PyRadiomics' .execute() is CPU-bound).
        rows: List[Dict[str, str]] = []
        with open(manifest_path, "r") as f:
            for row in csv.DictReader(f):
                rows.append(row)

        if jobs <= 1:
            # Sequential path — preserved verbatim from the original
            # implementation so single-job behaviour is unchanged.
            for row in rows:
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
                    feature_dict = _extract_one(extractor, patient_id, image_path, mask_path)
                    all_features.append(feature_dict)
                    patients_processed += 1
                    logger.info("Extracted %d features for %s", len(feature_dict) - 1, patient_id)
                except Exception as e:
                    logger.error("Failed to extract features for %s: %s", patient_id, e)
                    patients_failed += 1
        else:
            # Parallel path. The extractor itself is pickleable (PyRadiomics
            # stores its config as plain attributes), and ProcessPoolExecutor
            # workers each instantiate their own through _worker_init so the
            # extractor doesn't get shared across processes (avoids GIL +
            # PyRadiomics' internal caches racing).
            init_args = (extractor_params, image_types)
            with ProcessPoolExecutor(
                max_workers=jobs, initializer=_worker_init, initargs=init_args
            ) as ex:
                future_to_pid: Dict[Any, str] = {}
                for row in rows:
                    patient_id = row.get("patient_id") or ""
                    image_path = row.get("image_path") or ""
                    mask_path = row.get("mask_path") or ""
                    if not mask_path or mask_path.strip() == "":
                        logger.warning(
                            "Skipping %s: no mask provided", patient_id,
                        )
                        patients_skipped += 1
                        continue
                    fut = ex.submit(_extract_one_worker, patient_id, image_path, mask_path)
                    future_to_pid[fut] = patient_id

                for fut in as_completed(future_to_pid):
                    patient_id = future_to_pid[fut]
                    try:
                        feature_dict = fut.result()
                        all_features.append(feature_dict)
                        patients_processed += 1
                        logger.info(
                            "Extracted %d features for %s",
                            len(feature_dict) - 1, patient_id,
                        )
                    except BrokenProcessPool:
                        # A worker killed mid-extraction (commonly the OS OOM
                        # killer on a large volume) poisons the whole pool:
                        # this future and every still-pending one raise
                        # BrokenProcessPool. Report per-patient and keep going
                        # instead of letting the pool exception propagate.
                        logger.error(
                            "Worker process died extracting %s (likely out of "
                            "memory) — re-run with fewer jobs (e.g. jobs=1)",
                            patient_id,
                        )
                        patients_failed += 1
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


# ─── Module-level helpers (must be picklable for ProcessPoolExecutor) ───────
# These live at module scope so ProcessPoolExecutor can pickle them. Each
# worker process instantiates its own PyRadiomics extractor in _worker_init
# and reuses it across calls.

_WORKER_EXTRACTOR: Optional[Any] = None  # one per process


def _worker_init(extractor_params: Dict[str, Any], image_types: List[str]) -> None:
    """ProcessPoolExecutor initializer — build a per-process extractor once."""
    global _WORKER_EXTRACTOR
    from radiomics import featureextractor  # type: ignore

    fx = featureextractor.RadiomicsFeatureExtractor(**extractor_params)
    fx.disableAllImageTypes()
    for img_type in image_types:
        fx.enableImageTypeByName(img_type)
    _WORKER_EXTRACTOR = fx


def _extract_one_worker(patient_id: str, image_path: str, mask_path: str) -> Dict[str, Any]:
    """Single-patient extraction body executed inside a pool worker."""
    if _WORKER_EXTRACTOR is None:  # pragma: no cover
        raise RuntimeError("Worker extractor not initialised")
    return _extract_one(_WORKER_EXTRACTOR, patient_id, image_path, mask_path)


def _extract_one(extractor: Any, patient_id: Optional[str], image_path: Optional[str],
                 mask_path: Optional[str]) -> Dict[str, Any]:
    """Run PyRadiomics on one (image, mask) pair and shape the row."""
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
    return feature_dict
