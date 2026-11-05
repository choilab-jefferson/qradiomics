"""
Radiomics extraction service (sync API) — radiomics-tools engine.

Mirrors qradiomics/extractor.py's RadiomicsExtractor contract.  By default the
features come from the native Python port in :mod:`qradiomics.feature.rtools`,
which reproduces the legacy C++ ``FeatureExtraction`` binary to its printed
precision.  Settings:

- ``rtools_backend``: ``"python"`` (default) or ``"cxx"`` (shell out to the
  legacy binary; parity checks only).
- ``rtools_ibsi``: ``True`` selects the opt-in IBSI-1 profile (python only;
  ``rtools_ibsi_bin_width`` / ``rtools_ibsi_bin_number`` choose the
  discretisation).
"""
from __future__ import annotations

import csv
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


def parse_feature_txt(text: str) -> Dict[str, float]:
    """Parse FeatureExtraction's ``key=value`` text output into a dict.

    Ported logic-for-logic from ``pipelines/common/featparse.py`` (see that
    module for the canonical copy used by the standalone
    ``pipelines/heart_local/`` scripts). Kept as a separate copy here since
    ``qradiomics/`` is the installed package and must not import across the
    ``pipelines/`` boundary.

    - Lines without ``=`` are skipped (blank lines, headers, comments).
    - Lines mentioning ``PrincipalAxes`` or ``Eigenvectors`` are skipped
      (matrix/vector components the callers don't consume).
    - Key is everything before the first ``=``, with any trailing ``=``
      and surrounding whitespace stripped.
    - Value is everything after the first ``=``, truncated at the first
      ``[`` (bracketed array values keep only their scalar prefix, e.g. an
      empty/absent one truncates to ``""`` and is dropped below) and
      whitespace-stripped.
    - Values that don't parse as ``float`` are silently dropped.
    """
    feats: Dict[str, float] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        if "PrincipalAxes" in line or "Eigenvectors" in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip().rstrip("=").strip()
        v = v.strip().split("[")[0].strip()
        try:
            feats[k] = float(v)
        except ValueError:
            pass
    return feats


def _default_binary_path() -> Path:
    override = os.environ.get("QRADIOMICS_RTOOLS_BIN")
    if override:
        return Path(override)
    return Path.home() / "gitRepos/radiomics-tools/bin/FeatureExtraction"


class RtoolsExtractor:
    """Synchronous radiomics extractor wrapper around the legacy
    radiomics-tools ``FeatureExtraction`` C++ binary.

    Usage:
        extractor = RtoolsExtractor()
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
        """Run legacy radiomics-tools extraction for all image/mask pairs in
        manifest.

        This is a blocking call intended to be run inside a ThreadPoolExecutor.
        Returns a result dict compatible with RadiomicsExtractor.run_extraction.
        """
        settings = dict(extraction_settings or {})
        backend = str(settings.get("rtools_backend", "python"))
        if settings.get("rtools_ibsi"):
            backend = "ibsi"
        binary_path = _default_binary_path()
        if backend == "cxx" and not binary_path.is_file():
            logger.error("radiomics-tools binary not found at %s", binary_path)
            return {
                "features_uri": f"file://{job_dir.resolve()}/features.csv",
                "feature_count": 0,
                "status": "error",
                "error": (
                    f"radiomics-tools binary not found at {binary_path} "
                    "(set QRADIOMICS_RTOOLS_BIN to override)"
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        label = str(settings.get("rtools_label", 1))
        feature_select = str(settings.get("rtools_feature_select", "2isgcr"))
        bins = str(settings.get("rtools_bins", 64))
        timeout = settings.get("rtools_timeout", 120)
        if backend == "ibsi":
            feature_select = {
                k: settings[k]
                for k in ("rtools_ibsi_bin_width", "rtools_ibsi_bin_number")
                if settings.get(k) is not None
            }

        all_features: List[Dict[str, Any]] = []
        patients_processed = 0
        patients_failed = 0
        patients_skipped = 0

        rows: List[Dict[str, str]] = []
        with open(manifest_path, "r") as f:
            for row in csv.DictReader(f):
                rows.append(row)

        to_run: List[Dict[str, str]] = []
        for row in rows:
            patient_id = row.get("patient_id")
            mask_path = row.get("mask_path")
            if not mask_path or mask_path.strip() == "":
                logger.warning(
                    "Skipping %s: no mask provided (radiomics requires a segmentation mask)",
                    patient_id,
                )
                patients_skipped += 1
                continue
            to_run.append(row)

        # Subprocess calls release the GIL for most of their wall-clock time,
        # so a thread pool (not a process pool) is enough to parallelize
        # jobs>1 here — each worker just waits on its own `FeatureExtraction`
        # child process.
        if jobs <= 1:
            for row in to_run:
                result = self._extract_one(
                    backend,
                    binary_path,
                    label,
                    feature_select,
                    bins,
                    timeout,
                    job_dir,
                    row,
                )
                if result is not None:
                    all_features.append(result)
                    patients_processed += 1
                else:
                    patients_failed += 1
        else:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                future_to_pid = {
                    ex.submit(
                        self._extract_one,
                        backend,
                        binary_path,
                        label,
                        feature_select,
                        bins,
                        timeout,
                        job_dir,
                        row,
                    ): row.get("patient_id")
                    for row in to_run
                }
                for fut in as_completed(future_to_pid):
                    result = fut.result()
                    if result is not None:
                        all_features.append(result)
                        patients_processed += 1
                    else:
                        patients_failed += 1

        features_path = job_dir / "features.csv"
        if all_features:
            from qradiomics.extractor import union_fieldnames

            fieldnames = union_fieldnames(all_features)
            with open(features_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
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

    @staticmethod
    def _extract_one(
        backend: str,
        binary_path: Path,
        label: str,
        feature_select: str,
        bins: str,
        timeout: float,
        job_dir: Path,
        row: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Extract one manifest row (native port, or the legacy binary when
        ``backend == "cxx"``); return the feature dict, or
        None on failure (caller counts the None as patients_failed)."""
        patient_id = row.get("patient_id")
        image_path = row.get("image_path")
        mask_path = row.get("mask_path")
        logger.info("Extracting features for %s", patient_id)

        out_txt = job_dir / f"_rtools_{patient_id}.txt"
        try:
            if backend in ("python", "ibsi"):
                feats = _extract_native(backend, image_path, mask_path, label, feature_select, bins)
                feature_dict = {"patient_id": patient_id, **feats}
                logger.info("Extracted %d features for %s", len(feature_dict) - 1, patient_id)
                return feature_dict
            subprocess.run(
                [
                    str(binary_path),
                    str(image_path),
                    str(mask_path),
                    str(out_txt),
                    label,
                    feature_select,
                    bins,
                ],
                capture_output=True,
                timeout=timeout,
                check=True,
            )
            text = out_txt.read_text()
            feats = parse_feature_txt(text)
            feature_dict: Dict[str, Any] = {"patient_id": patient_id, **feats}
            logger.info("Extracted %d features for %s", len(feature_dict) - 1, patient_id)
            return feature_dict
        except subprocess.CalledProcessError as e:
            logger.error(
                "radiomics-tools binary failed for %s: exit %s: %s",
                patient_id,
                e.returncode,
                e.stderr,
            )
            return None
        except subprocess.TimeoutExpired:
            logger.error("radiomics-tools binary timed out for %s", patient_id)
            return None
        except Exception as e:
            logger.error("Failed to extract features for %s: %s", patient_id, e)
            return None
        finally:
            try:
                out_txt.unlink(missing_ok=True)
            except Exception:
                pass


def _extract_native(backend, image_path, mask_path, label, feature_select, bins) -> Dict[str, float]:
    """Features from the native port (``backend`` = ``python`` | ``ibsi``)."""
    from qradiomics.feature.rtools import extract_features

    if backend == "ibsi":
        kw = feature_select if isinstance(feature_select, dict) else {}
        ibsi_kw = {}
        if "rtools_ibsi_bin_width" in kw:
            ibsi_kw["bin_width"] = float(kw["rtools_ibsi_bin_width"])
        if "rtools_ibsi_bin_number" in kw:
            ibsi_kw["bin_number"] = int(kw["rtools_ibsi_bin_number"])
        return extract_features(image_path, mask_path, label=int(label), ibsi=True, **ibsi_kw)
    return extract_features(
        image_path, mask_path, label=int(label), feature_select=feature_select, number_of_bins=int(bins)
    )


def get_rtools_extractor() -> RtoolsExtractor:
    """Factory for a RtoolsExtractor instance (lightweight).

    Callers may instantiate directly or use this factory.
    """
    return RtoolsExtractor()
