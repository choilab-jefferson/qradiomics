"""qr delta — compute cross-timepoint delta + trend features.

Reads a features.csv that contains `patient_id` + `timepoint` (and
optionally `relative_day` for trend) columns, applies one or more
DeltaPair definitions, and writes a wide CSV with one row per patient.

DeltaPair definitions can be passed inline (`--pair name=A-B`) or as a
JSON/YAML file (`--pairs-file …`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import click

from qradiomics.delta import DeltaPair, compute_delta, compute_trend


def _parse_inline_pair(spec: str) -> DeltaPair:
    """Parse 'name=minuend-subtrahend' → DeltaPair."""
    if "=" not in spec or "-" not in spec.split("=", 1)[1]:
        raise click.BadParameter(
            f"invalid pair spec {spec!r}; expected name=minuend-subtrahend"
        )
    name, body = spec.split("=", 1)
    minuend, subtrahend = body.split("-", 1)
    return DeltaPair(name=name.strip(),
                     minuend=minuend.strip(),
                     subtrahend=subtrahend.strip())


def _load_pairs_file(path: Path) -> List[DeltaPair]:
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, list):
        raise click.BadParameter(f"{path} must be a list of {{name,minuend,subtrahend}} dicts")
    return [DeltaPair(name=d["name"], minuend=d["minuend"],
                      subtrahend=d["subtrahend"]) for d in data]


@click.command()
@click.option("--features", "-f", "features_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="features.csv with patient_id + timepoint columns")
@click.option("--pair", "pair_specs", multiple=True,
              help="DeltaPair definition 'name=minuend-subtrahend'. Repeatable.")
@click.option("--pairs-file", type=click.Path(exists=True, dir_okay=False),
              default=None, help="JSON or YAML file with a list of "
                                 "{name, minuend, subtrahend} dicts.")
@click.option("--with-trend/--no-trend", default=False,
              help="Also emit per-feature linear slope vs relative_day.")
@click.option("--time-col", default="relative_day",
              help="Numeric time column for trend computation (default: relative_day).")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output CSV (one row per patient_id).")
def delta(features_path, pair_specs, pairs_file, with_trend, time_col, output):
    """Compute delta and (optionally) trend features per patient.

    \b
    Example:
        qr delta -f features.csv \\
            --pair w4_minus_baseline=week4-baseline \\
            --pair w8_minus_baseline=week8-baseline \\
            --with-trend -o delta_features.csv
    """
    import pandas as pd

    pairs: List[DeltaPair] = []
    if pairs_file:
        pairs.extend(_load_pairs_file(Path(pairs_file)))
    for spec in pair_specs:
        pairs.append(_parse_inline_pair(spec))
    if not pairs and not with_trend:
        raise click.UsageError("Provide at least one --pair or --pairs-file, "
                               "or pass --with-trend.")

    df = pd.read_csv(features_path)
    if "patient_id" not in df.columns or "timepoint" not in df.columns:
        raise click.UsageError("features CSV needs both 'patient_id' and 'timepoint' columns")

    parts = []
    if pairs:
        click.echo(f"[delta] computing {len(pairs)} pair(s) on {len(df)} rows")
        parts.append(compute_delta(df, pairs))
    if with_trend:
        if time_col not in df.columns:
            raise click.UsageError(f"--with-trend requested but column {time_col!r} missing")
        click.echo(f"[delta] computing trend slopes vs {time_col}")
        parts.append(compute_trend(df, time_col=time_col))

    out = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output)
    click.echo(f"[delta] wrote {out.shape[0]} rows × {out.shape[1]} cols → {output}")
