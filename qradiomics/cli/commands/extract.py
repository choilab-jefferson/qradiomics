"""qr extract — read a manifest CSV, run one or more extraction engines
(pyradiomics/pysera/rtools) in-process, write features.csv."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import click

from qradiomics import PatternLoader
from qradiomics.extractor_registry import resolve_engines, run_multi_extraction

# PySERA category names for each PyRadiomics feature_classes entry (pattern
# feature_extraction.feature_classes -> pysera categories string).
_FEATURE_CLASS_TO_PYSERA_CATEGORY = {
    "firstorder": "stat",
    "shape": "morph",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "ngtdm": "ngtdm",
    "ngldm": "ngldm",
    "gldzm": "gldzm",
}


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
    help="Pattern ID (qr pattern list). Omit to use PyRadiomics defaults.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path for features CSV",
)
@click.option(
    "--jobs",
    "-j",
    type=int,
    default=1,
    show_default=True,
    help="Parallel extraction workers (ProcessPoolExecutor). Default 1 keeps "
         "the existing sequential behaviour. Use 4-8 for typical workstations "
         "or nproc for batch cohorts — PyRadiomics is CPU-bound so wall-clock "
         "scales close to linearly with worker count.",
)
@click.option(
    "--bin-width",
    default=None,
    type=float,
    help="Override PyRadiomics binWidth (default: pattern setting or 25)",
)
@click.option(
    "--engine",
    type=str,
    default=None,
    help="Comma-separated extraction engines (pyradiomics,pysera,rtools) or "
         "'all'. Defaults to the pattern's configured extractor, or "
         "'pyradiomics' if none.",
)
def extract(manifest, pattern, output, jobs, bin_width, engine):
    """Extract radiomics features from a manifest.

    \b
    Manifest CSV columns (lowercase, required):
        patient_id, modality, image_path, mask_path

    \b
    Example:
        qr extract -m manifest.csv -p nsclc-survival -o features.csv
        qr extract -m manifest.csv -o features.csv --engine pysera
        qr extract -m manifest.csv -o features.csv --engine pyradiomics,pysera
        qr extract -m manifest.csv -o features.csv --engine all
    """
    extraction_settings: dict = {}
    tmpl = None
    if pattern:
        loader = PatternLoader()
        tmpl = loader.get_pattern(pattern)
        if tmpl is None:
            click.echo(f"Pattern '{pattern}' not found. Run 'qr pattern list'.", err=True)
            raise SystemExit(1)
        fx = tmpl.feature_extraction
        extraction_settings = dict(fx.settings or {})
        extraction_settings["image_types"] = list(fx.image_types or ["Original"])
        click.echo(
            f"Using pattern '{pattern}': image_types={extraction_settings['image_types']}, "
            f"feature_classes={list(fx.feature_classes)}"
        )

    try:
        engines = resolve_engines(
            engine, tmpl.feature_extraction.extractor if tmpl else None
        )
    except KeyError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1)

    if tmpl is not None and "pysera" in engines:
        categories = []
        for feature_class in tmpl.feature_extraction.feature_classes:
            categories.append(_FEATURE_CLASS_TO_PYSERA_CATEGORY.get(feature_class, feature_class))
        # dedupe while preserving order
        seen = set()
        deduped_categories = []
        for category in categories:
            if category not in seen:
                seen.add(category)
                deduped_categories.append(category)
        extraction_settings["categories"] = ",".join(deduped_categories)

    if bin_width is not None:
        extraction_settings["binWidth"] = bin_width

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    job_dir = output_path.parent / f".extract_{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Reading manifest: {manifest}")
    try:
        result = run_multi_extraction(
            engines=engines,
            job_id=uuid4(),
            manifest_path=Path(manifest),
            job_dir=job_dir,
            extraction_settings=extraction_settings,
            jobs=jobs,
        )
    except KeyError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1)

    produced = job_dir / "features.csv"
    if not produced.exists():
        click.echo("Extraction produced no features.csv", err=True)
        raise SystemExit(1)

    os.replace(produced, output_path)
    try:
        job_dir.rmdir()
    except OSError:
        pass

    click.echo(
        f"\nExtraction {result.get('status', 'unknown')}: "
        f"{result.get('patients_processed', 0)} processed, "
        f"{result.get('patients_failed', 0)} failed, "
        f"{result.get('patients_skipped', 0)} skipped"
    )
    click.echo(f"Features ({result.get('feature_count', 0)} per patient) -> {output_path}")
