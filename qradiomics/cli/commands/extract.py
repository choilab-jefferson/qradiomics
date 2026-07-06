"""qr extract — read a manifest CSV, run PyRadiomics, write features.csv.

Supports multi-process extraction via `--jobs N`. Each worker processes one
patient at a time; results are streamed to `features.csv` as they complete,
so the output file is observable in real time (monitor.py, tail).
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

import click

from qradiomics import PatternLoader


def _process_one(args):
    """Worker entry: extract features for a single patient.

    Runs in a separate worker process. Loads (image, mask) NRRDs and
    delegates to qradiomics.atomic.extract_features, which builds the
    PyRadiomics extractor + strips diagnostics_* per call. Pattern YAML
    can be passed via extraction_settings["params_file"].
    """
    patient_id, image_path, mask_path, extraction_settings = args
    try:
        from qradiomics.atomic import extract_features, load_image_and_mask

        settings = dict(extraction_settings or {})
        params_file = settings.get("params_file")
        label = int(settings.get("label", 1))
        geom_tol = float(settings.get("geometryTolerance", 1e-3))

        image, mask = load_image_and_mask(image_path, mask_path,
                                          require_compatible_geometry=False)
        features = extract_features(
            image, mask,
            params_file=params_file,
            label=label,
            geometry_tolerance=geom_tol,
        )
        feature_dict = {"patient_id": patient_id}
        feature_dict.update(features)
        return (patient_id, feature_dict, None)
    except Exception as e:
        return (patient_id, None, str(e))


@click.command()
@click.option(
    "--manifest",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Manifest CSV with columns: patient_id, modality, image_path, mask_path",
)
@click.option(
    "--pattern",
    "-p",
    default=None,
    help="Pattern ID (qr pattern list). Omit to enable ALL image types and "
         "feature classes (~1400 features, incl. wavelet/LoG/square/... — "
         "NOT prefixed 'original_'). Pass a pattern for a curated subset.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path for features CSV",
)
@click.option(
    "--bin-width",
    default=None,
    type=float,
    help="Override PyRadiomics binWidth (default: pattern setting or 25)",
)
@click.option(
    "--jobs",
    "-j",
    default=1,
    type=int,
    help="Parallel worker processes (default 1 = sequential). Set to nproc "
         "for full multi-process extraction; each worker takes one patient.",
)
def extract(manifest, pattern, output, bin_width, jobs):
    """Extract radiomics features from a manifest.

    \b
    Manifest CSV columns (lowercase, required):
        patient_id, modality, image_path, mask_path

    \b
    Examples:
        qr extract -m manifest.csv -p nsclc-survival -o features.csv
        qr extract -m manifest.csv -p ct-default -o features.csv -j 16
    """
    extraction_settings: dict = {
        "binWidth": 25,
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
        "verbose": False,
        "geometryTolerance": 1e-3,
        "correctMask": True,
    }
    if pattern:
        loader = PatternLoader()
        tmpl = loader.get_pattern(pattern)
        if tmpl is None:
            click.echo(f"Pattern '{pattern}' not found. Run 'qr pattern list'.", err=True)
            raise SystemExit(1)
        fx = tmpl.feature_extraction
        if fx.settings:
            extraction_settings.update(fx.settings)
            try:
                extraction_settings["geometryTolerance"] = float(
                    extraction_settings.get("geometryTolerance", 1e-3)
                )
            except (TypeError, ValueError):
                extraction_settings["geometryTolerance"] = 1e-3
        extraction_settings["image_types"] = list(fx.image_types or ["Original"])
        click.echo(
            f"Using pattern '{pattern}': image_types={extraction_settings['image_types']}, "
            f"feature_classes={list(fx.feature_classes)}"
        )

    if bin_width is not None:
        extraction_settings["binWidth"] = bin_width

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read manifest into a list so we know the total up front.
    with open(manifest, "r") as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    if total == 0:
        click.echo("Manifest is empty.", err=True)
        raise SystemExit(1)

    click.echo(
        f"Reading manifest: {manifest}  ({total} patients, {jobs} worker"
        + ("s" if jobs != 1 else "") + ")"
    )

    work = [
        (row["patient_id"], row["image_path"], row.get("mask_path", ""), extraction_settings)
        for row in rows
    ]

    # Stream features.csv: write header on first success, append per result.
    n_ok = 0
    n_failed = 0
    n_skipped = 0
    field_count = 0
    with open(output_path, "w", newline="") as fout:
        writer: csv.DictWriter | None = None

        def _emit(patient_id: str, feature_dict: dict | None, err: str | None):
            nonlocal writer, n_ok, n_failed, n_skipped, field_count
            if feature_dict is None:
                if err and "no mask" in err.lower():
                    n_skipped += 1
                else:
                    n_failed += 1
                return
            if writer is None:
                writer = csv.DictWriter(fout, fieldnames=list(feature_dict.keys()))
                writer.writeheader()
                field_count = len(feature_dict) - 1
            writer.writerow(feature_dict)
            fout.flush()
            n_ok += 1

        if jobs <= 1:
            # Sequential — preserves manifest order
            for i, w in enumerate(work, 1):
                patient_id = w[0]
                print(f"  extract [{i}/{total}] {patient_id} ...", flush=True)
                pid, feats, err = _process_one(w)
                _emit(pid, feats, err)
                if feats:
                    print(f"  extract [{i}/{total}] {pid} ok ({len(feats)-1} features)",
                          flush=True)
                else:
                    print(f"  extract [{i}/{total}] {pid} FAILED: {err}", flush=True)
        else:
            # Multi-process — order is non-deterministic but the output CSV
            # contains one row per patient, identified by patient_id.
            done = 0
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(_process_one, w): w[0] for w in work}
                for fut in as_completed(futs):
                    done += 1
                    pid, feats, err = fut.result()
                    _emit(pid, feats, err)
                    if feats:
                        print(
                            f"  extract [{done}/{total}] {pid} ok ({len(feats)-1} features)",
                            flush=True,
                        )
                    else:
                        print(f"  extract [{done}/{total}] {pid} FAILED: {err}", flush=True)
                    sys.stdout.flush()

    status = "extracted" if n_ok > 0 else "error"
    click.echo(
        f"\nExtraction {status}: {n_ok} processed, {n_failed} failed, {n_skipped} skipped"
    )
    click.echo(f"Features ({field_count} per patient) -> {output_path}")
