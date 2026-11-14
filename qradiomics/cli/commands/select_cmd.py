"""qr select — pair-manifest contract (see qradiomics.pairs)."""

from __future__ import annotations

from pathlib import Path

import click

from qradiomics import pairs as pairs_mod


@click.group("select")
def select():
    """Pair-manifest tools (the selection -> pipeline contract)."""


@select.command("validate")
@click.argument(
    "manifest_arg", metavar="[MANIFEST]", required=False, type=click.Path(dir_okay=False)
)
@click.option(
    "--manifest",
    "-m",
    "manifest_opt",
    type=click.Path(dir_okay=False),
    help="Same as the MANIFEST argument.",
)
@click.option(
    "--check-paths/--no-check-paths",
    default=False,
    show_default=True,
    help="Also require series_dir / rtstruct_path to exist on disk.",
)
def validate(manifest_arg, manifest_opt, check_paths):
    """Validate a pair manifest (.csv / .parquet); exit 1 and list every problem if invalid.

    \b
    Columns: cohort, case_id, pair_id, image_key, label_key, series_uid,
    series_dir, rtstruct_path, rtstruct_sha, roi_candidates, resample_mm,
    status (ok|skipped|failed), reason.
    """
    manifest = manifest_opt or manifest_arg
    if not manifest or not Path(manifest).is_file():
        raise click.UsageError("give an existing MANIFEST (argument or --manifest)")
    df = pairs_mod.read_manifest(manifest)
    errs = pairs_mod.validate_pairs(df, check_paths=check_paths)
    if errs:
        for e in errs:
            click.echo(f"INVALID: {e}", err=True)
        raise SystemExit(1)
    counts = df["status"].value_counts().to_dict()
    click.echo(
        f"OK: {len(df)} pair(s), {df['case_id'].nunique()} case(s), "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
