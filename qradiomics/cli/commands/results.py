"""Results commands — join features with clinical outcomes."""

from pathlib import Path

import click
import pandas as pd


@click.group()
def results():
    """Merge features with clinical CSVs to produce analysis_ready.csv."""


@results.command()
@click.option(
    "--features",
    "-f",
    required=True,
    type=click.Path(exists=True),
    help="Path to features.csv (from qr extract)",
)
@click.option(
    "--clinical",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to clinical.csv",
)
@click.option(
    "--clinical-id-col",
    default="patient_id",
    help="Patient ID column name in clinical CSV (default: patient_id)",
)
@click.option(
    "--time-col",
    default="OS_days",
    help="Survival time column — auto-detected as days (>100 median) or months",
)
@click.option(
    "--event-col",
    default="OS_event",
    help="Event column (1=event, 0=censored). Only processed when --time-col "
         "is also present in the clinical CSV — see --outcome-col for "
         "classify-task cohorts that have neither.",
)
@click.option(
    "--outcome-col",
    default=None,
    help="Extra clinical column to pass through into the merged output "
         "verbatim (e.g. a classify-task outcome like fdg_uptake_pattern). "
         "Independent of --time-col/--event-col: a cohort can have "
         "survival columns, an outcome-col, or both. At least one of "
         "(time-col + event-col) or --outcome-col must be present in the "
         "clinical CSV.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path for analysis_ready.csv",
)
def merge(features, clinical, clinical_id_col, time_col, event_col, outcome_col, output):
    """Merge radiomics features with clinical outcome data on patient_id.

    \b
    Survival: auto-detects days vs months (if median time > 100, treats as
    days and divides by 30.44 to convert to OS_months). Only kept in the
    output when both --time-col and --event-col are present in the
    clinical CSV.
    Classify (or anything else): pass --outcome-col to carry a clinical
    column through unchanged instead of (or alongside) the survival pair.
    Only the columns explicitly requested (OS_months/OS_event and/or
    --outcome-col) are kept — every other clinical column is dropped, so
    an unrelated column never silently leaks into training as a feature.
    """
    feat_df = pd.read_csv(features)
    clin_df = pd.read_csv(clinical)

    if "patient_id" not in feat_df.columns:
        click.echo("features CSV must have a 'patient_id' column", err=True)
        raise SystemExit(1)

    if clinical_id_col not in clin_df.columns:
        click.echo(f"Column '{clinical_id_col}' not found in clinical CSV", err=True)
        raise SystemExit(1)

    if clinical_id_col != "patient_id":
        clin_df = clin_df.rename(columns={clinical_id_col: "patient_id"})

    time_present = time_col in clin_df.columns
    event_present = event_col in clin_df.columns
    has_survival_cols = time_present and event_present
    if not has_survival_cols and not outcome_col:
        # No --outcome-col fallback given — same errors as the original,
        # survival-only version of this command (exact message preserved).
        if not time_present:
            click.echo(f"Time column '{time_col}' not found in clinical CSV", err=True)
        else:
            click.echo(f"Event column '{event_col}' not found in clinical CSV", err=True)
        raise SystemExit(1)

    keep = ["patient_id"]
    value_cols: list[str] = []

    if has_survival_cols:
        time_values = pd.to_numeric(clin_df[time_col], errors="coerce")
        if float(time_values.median()) > 100:
            clin_df["OS_months"] = time_values / 30.44
        else:
            clin_df["OS_months"] = time_values
        clin_df["OS_event"] = pd.to_numeric(clin_df[event_col], errors="coerce")
        keep += ["OS_months", "OS_event"]
        value_cols += ["OS_months", "OS_event"]

    if outcome_col:
        if outcome_col not in clin_df.columns:
            click.echo(f"--outcome-col '{outcome_col}' not found in clinical CSV", err=True)
            raise SystemExit(1)
        if outcome_col not in keep:
            keep.append(outcome_col)
            value_cols.append(outcome_col)

    clin_slim = clin_df[keep].dropna(subset=value_cols)

    merged = feat_df.merge(clin_slim, on="patient_id", how="inner")

    unmatched = len(feat_df) - len(merged)
    if unmatched > 0:
        missing_ids = set(feat_df["patient_id"]) - set(clin_slim["patient_id"])
        sample = ", ".join(sorted(str(x) for x in missing_ids)[:5])
        click.echo(
            f"{unmatched} unmatched patients (features not in clinical): {sample}"
            + ("..." if unmatched > 5 else "")
        )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    click.echo(f"Merged {len(merged)} patients, {len(merged.columns)} columns -> {output}")


@results.command()
@click.option("--input-csv", "-i", required=True,
              type=click.Path(exists=True),
              help="Feature CSV (from qr extract / qr results merge)")
@click.option("--output-csv", "-o", required=True, type=click.Path(),
              help="Output path for the sanitized feature CSV")
@click.option("--site-col", "-s", default=None,
              help="Site / scanner column for ComBat batch correction. "
                   "Omit to skip ComBat (single-site cohort).")
@click.option("--label-col", "-l", default=None,
              help="Categorical outcome column to PRESERVE through ComBat "
                   "(so harmonization does not erase outcome-correlated variance).")
@click.option("--confounders", "-c", default="",
              help="Comma-separated continuous confounders to preserve in "
                   "ComBat AND residualize out afterwards "
                   "(e.g. 'original_shape_MeshVolume,blood_glucose').")
@click.option("--icc-threshold", default=None, type=float,
              help="If set, drop features whose pre/post-ComBat ICC(3,1) is "
                   "below this (e.g. 0.75). Requires --site-col.")
@click.option("--no-residualize", is_flag=True, default=False,
              help="Skip the linear residualization step (ComBat / ICC only).")
@click.option("--no-preserve-scale", is_flag=True, default=False,
              help="Centre residuals at zero instead of re-adding the "
                   "feature's global mean (default preserves scale).")
@click.option("--feature-prefix", default="original_",
              help="Treat columns starting with this prefix as features "
                   "(default 'original_'; everything else is metadata).")
def sanitize(input_csv, output_csv, site_col, label_col, confounders,
             icc_threshold, no_residualize, no_preserve_scale, feature_prefix):
    """Harmonize + residualize + ICC-filter an extracted feature table.

    \b
    Pipeline (each stage optional):
      1. ComBat        remove site/scanner batch effect (--site-col),
                       preserving --label-col and --confounders
      2. ICC filter    drop features with pre/post ICC < --icc-threshold
      3. residualize   regress out --confounders (scale-preserving)

    \b
    Example (multi-site cardiac, volume + glucose confounders):
      qr results sanitize -i raw.csv -o clean.csv \\
        -s Center -l Cardiotoxicity \\
        -c original_shape_MeshVolume,blood_glucose --icc-threshold 0.75
    """
    import numpy as np
    from qradiomics.analytics import (
        combat_harmonize, residualize_linear, icc_filter,
    )

    df = pd.read_csv(input_csv)
    conf = [c.strip() for c in confounders.split(",") if c.strip()]
    meta = set(filter(None, [site_col, label_col, "patient_id", "PID",
                             "anon_pid", "modality", "image_path",
                             "mask_path", "cohort", "contour", "source_root"]))
    meta |= set(conf)
    feature_cols = [c for c in df.columns
                    if c.startswith(feature_prefix) and c not in meta]
    if not feature_cols:
        click.echo(f"No feature columns matched prefix '{feature_prefix}'", err=True)
        raise SystemExit(1)

    # drop NaN/zero-variance features up front (ComBat is sensitive to both)
    bad = [c for c in feature_cols
           if df[c].isna().any() or float(pd.to_numeric(df[c], errors="coerce").var()) == 0]
    if bad:
        click.echo(f"Dropping {len(bad)} NaN / zero-variance features")
        feature_cols = [c for c in feature_cols if c not in bad]

    click.echo(f"sanitize: {len(df)} samples, {len(feature_cols)} features")

    df_raw = df.copy()
    # 1. ComBat
    if site_col:
        if site_col not in df.columns:
            click.echo(f"--site-col '{site_col}' not in CSV", err=True)
            raise SystemExit(1)
        n_sites = df[site_col].nunique()
        if n_sites < 2:
            click.echo(f"  [combat] only {n_sites} site — skipping ComBat")
        else:
            click.echo(f"  [combat] harmonizing across {n_sites} sites "
                       f"(preserve label={label_col}, confounders={conf})")
            df = combat_harmonize(
                df, feature_cols, site_col,
                categorical_covariates=[label_col] if label_col else (),
                continuous_covariates=conf,
            )

    # 2. ICC filter (pre vs post ComBat)
    if icc_threshold is not None and site_col and df[site_col].nunique() >= 2:
        pid = "patient_id" if "patient_id" in df.columns else (
            "PID" if "PID" in df.columns else None)
        if pid is None:
            click.echo("  [icc] no patient_id/PID column — skipping ICC", err=True)
        else:
            kept, table = icc_filter(df_raw, df, feature_cols, pid,
                                     threshold=icc_threshold)
            click.echo(f"  [icc] {len(kept)}/{len(feature_cols)} features "
                       f"survive ICC>= {icc_threshold} "
                       f"(mean ICC {table.icc.mean():.3f})")
            feature_cols = kept

    # 3. residualize
    if conf and not no_residualize:
        click.echo(f"  [residualize] regressing out {conf} "
                   f"(preserve_scale={not no_preserve_scale})")
        df = residualize_linear(df, feature_cols, conf,
                                preserve_scale=not no_preserve_scale)

    # assemble output: metadata + surviving features
    meta_present = [c for c in df.columns
                    if c in meta or not c.startswith(feature_prefix)]
    out_cols = list(dict.fromkeys(meta_present + feature_cols))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(output_csv, index=False)
    click.echo(f"sanitized {len(df)} samples, {len(feature_cols)} features "
               f"-> {output_csv}")
