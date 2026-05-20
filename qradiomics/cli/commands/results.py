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
    default=None,
    help="Survival time column — auto-detected as days (>100 median) or months. "
         "Omit for classification tasks.",
)
@click.option(
    "--event-col",
    default=None,
    help="Event column (1=event, 0=censored). Omit for classification tasks.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path for analysis_ready.csv",
)
def merge(features, clinical, clinical_id_col, time_col, event_col, output):
    """Merge radiomics features with clinical data on patient_id.

    \b
    Survival mode (both --time-col and --event-col given):
        auto-converts days→months (median>100 ⇒ days), keeps only
        OS_months + OS_event columns.
    Classify mode (no time/event cols):
        passes through all clinical columns unchanged — caller picks the
        outcome column at `qr ml train` time.
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

    survival_mode = time_col is not None and event_col is not None
    if survival_mode:
        if time_col not in clin_df.columns:
            click.echo(f"Time column '{time_col}' not found in clinical CSV", err=True)
            raise SystemExit(1)
        if event_col not in clin_df.columns:
            click.echo(f"Event column '{event_col}' not found in clinical CSV", err=True)
            raise SystemExit(1)
        time_values = pd.to_numeric(clin_df[time_col], errors="coerce")
        if float(time_values.median()) > 100:
            clin_df["OS_months"] = time_values / 30.44
        else:
            clin_df["OS_months"] = time_values
        clin_df["OS_event"] = pd.to_numeric(clin_df[event_col], errors="coerce")
        keep = ["patient_id", "OS_months", "OS_event"]
        clin_slim = clin_df[keep].dropna(subset=keep[1:])
    else:
        clin_slim = clin_df.dropna(subset=["patient_id"])

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
