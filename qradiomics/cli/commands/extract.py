"""qr extract — read a manifest CSV, run one or more extraction engines
(rtools [default] / pysera / pyradiomics) in-process, write features.csv."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import click

from qradiomics import PatternLoader
from qradiomics.extractor import PYRADIOMICS_INSTALL_HINT, pyradiomics_available
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


def _warn_rtools_ignored(tmpl, bin_width) -> None:
    """Tell the user which pattern/CLI settings the rtools engine does not apply.

    rtools computes Original-image 2D/3D intensity, shape and GLCM/GLRLM
    features only. Silent differences from a pyradiomics-oriented pattern would
    be easy to miss, so say so on stderr (stdout stays clean for scripts).
    """
    ignored = []
    if tmpl is not None:
        fx = tmpl.feature_extraction
        filtered = [t for t in (fx.image_types or []) if t != "Original"]
        if filtered:
            ignored.append(f"image_types {filtered}")
        if fx.settings:
            ignored.append("pattern settings (binWidth, resampling, ...)")
    if bin_width is not None:
        ignored.append("--bin-width")
    if ignored:
        click.echo(
            "Note: engine rtools computes Original-image features only and does not "
            f"apply: {'; '.join(ignored)}. Use --engine pyradiomics for the full "
            "PyRadiomics feature set.",
            err=True,
        )


@click.command()
@click.option(
    "--manifest",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Manifest (.csv/.parquet) with columns: patient_id, modality, image_path,"
    " mask_path; pair manifests also carry pair_id (one features row per pair).",
)
@click.option(
    "--only",
    default=None,
    help="Extract only this pair_id (else patient_id). Fails if it matches nothing or"
    " produces no features, so a per-pair task fails loudly.",
)
@click.option(
    "--pattern",
    "-p",
    default=None,
    help="Pattern ID (qr pattern list). Omit to use the default engine's own "
    "settings. Shipped patterns use rtools, which ignores their pyradiomics-only "
    "settings (filtered image types, binWidth); add --engine pyradiomics to apply them.",
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
    "or nproc for batch cohorts — extraction is CPU-bound so wall-clock "
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
    help="Comma-separated extraction engines (rtools,pysera,pyradiomics) or "
    "'all'. Defaults to the pattern's configured extractor (rtools for every "
    "shipped pattern), or 'rtools' (native, no optional dependency) if none. "
    "pyradiomics is optional: install it from git and request it with "
    "--engine pyradiomics.",
)
@click.option(
    "--case",
    "cases",
    multiple=True,
    help="Only extract rows whose case_id / patient_id equals this id (repeatable). "
    "Used for per-case fan-out (qr workflow run --executor nextflow); "
    "--output then holds just those cases.",
)
def extract(manifest, only, pattern, output, jobs, bin_width, engine, cases):
    """Extract radiomics features from a manifest.

    \b
    Manifest CSV columns (lowercase, required):
        patient_id, modality, image_path, mask_path

    \b
    Example:
        qr extract -m manifest.csv -p nsclc-survival -o features.csv
        qr extract -m manifest.csv -o features.csv --engine pyradiomics
        qr extract -m manifest.csv -o features.csv --engine pysera
        qr extract -m manifest.csv -o features.csv --engine pyradiomics,pysera
        qr extract -m manifest.csv -o features.csv --engine all
        qr extract -m manifest.csv -o cases/P001/features.csv --case P001
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
        # Pass the pattern's declared feature classes through to the pyradiomics
        # engine so a restrictive pattern actually limits extraction (previously
        # printed but ignored — every class was always emitted). Only for
        # pyradiomics; pysera uses `categories` below.
        if fx.feature_classes:
            extraction_settings["feature_classes"] = list(fx.feature_classes)
        click.echo(
            f"Using pattern '{pattern}': image_types={extraction_settings['image_types']}, "
            f"feature_classes={list(fx.feature_classes)}"
        )

    try:
        engines = resolve_engines(engine, tmpl.feature_extraction.engine_spec if tmpl else None)
    except KeyError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1)

    if "pyradiomics" in engines and not pyradiomics_available():
        click.echo(f"Engine 'pyradiomics' requested but unavailable. {PYRADIOMICS_INSTALL_HINT}", err=True)
        raise SystemExit(1)

    if "rtools" in engines:
        _warn_rtools_ignored(tmpl, bin_width)

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
    from qradiomics import pairs as pm

    try:
        mdf = pm.read_manifest(manifest)
        mdf = pm.select_rows(mdf, only=only, cases=cases) if (only or cases) else mdf
    except ValueError as e:
        click.echo(str(e), err=True)
        raise SystemExit(1)
    pair_mode = pm.is_pair_manifest(mdf)
    manifest_path = Path(manifest)
    case_of: dict = {}
    if only or cases or pair_mode or manifest_path.suffix == ".parquet":
        if pair_mode:
            # Engines key result rows on patient_id, which repeats across the pairs
            # of one case: key on pair_id for the run, map back to the case after.
            case_of = dict(zip(mdf["pair_id"], mdf["patient_id"]))
            mdf = mdf.assign(patient_id=mdf["pair_id"])
        manifest_path = job_dir / "_manifest.csv"
        mdf.to_csv(manifest_path, index=False)
    try:
        result = run_multi_extraction(
            engines=engines,
            job_id=uuid4(),
            manifest_path=manifest_path,
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

    if pair_mode:
        import pandas as pd

        fdf = pd.read_csv(produced, dtype={"patient_id": str})
        fdf.insert(0, "pair_id", fdf["patient_id"])
        fdf["patient_id"] = fdf["pair_id"].map(case_of)
        fdf.to_csv(produced, index=False)
    if only and result.get("patients_processed", 0) == 0:
        click.echo(f"--only {only}: no features were produced", err=True)
        raise SystemExit(1)
    os.replace(produced, output_path)
    for leftover in (job_dir / "_manifest.csv",):
        leftover.unlink(missing_ok=True)
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
