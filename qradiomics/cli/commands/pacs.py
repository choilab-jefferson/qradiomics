"""``qr pacs`` — generic PACS access (Orthanc REST / DICOMweb / DIMSE).

The active backend is selected by a YAML profile (see
:mod:`qradiomics.io.pacs.config`). All subcommands accept ``--profile`` and
``--config`` for explicit overrides.

Subcommands::

    qr pacs profiles                 List configured profiles.
    qr pacs ping                     C-ECHO / /system / QIDO probe.
    qr pacs query studies            QIDO / C-FIND at STUDY level.
    qr pacs query series             ... at SERIES level.
    qr pacs query instances          ... at IMAGE level.
    qr pacs fetch                    Download study / series / instance.
    qr pacs send                     STOW / C-STORE one file or a directory.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import click

from qradiomics.io.pacs import PACSError, get_backend, load_profiles


# ---------------------------------------------------------------------------- helpers


def _kv_to_dict(pairs: Iterable[str]) -> dict:
    out: dict = {}
    for entry in pairs:
        if "=" not in entry:
            raise click.BadParameter(f"--filter expects KEY=VALUE (got {entry!r})")
        key, value = entry.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _emit(records: list[dict], fmt: str, columns: Optional[list[str]] = None) -> None:
    if not records:
        click.echo("(no results)", err=True)
        return
    if fmt == "json":
        click.echo(json.dumps(records, indent=2, default=str))
        return
    if columns is None:
        seen: list[str] = []
        for row in records:
            for key in row:
                if key.startswith("_"):
                    continue
                if key not in seen:
                    seen.append(key)
        columns = seen
    if fmt == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
        return
    widths = {
        col: max(len(col), *(len(str(row.get(col, ""))) for row in records))
        for col in columns
    }
    click.echo("  ".join(col.ljust(widths[col]) for col in columns))
    click.echo("  ".join("-" * widths[col] for col in columns))
    for row in records:
        click.echo(
            "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
        )


_PROFILE_OPT = click.option(
    "--profile",
    "-P",
    default=None,
    help="PACS profile name (default: 'default_profile' from YAML).",
)
_CONFIG_OPT = click.option(
    "--config",
    "config_path",
    default=None,
    help="PACS config YAML path (overrides $QR_PACS_CONFIG and defaults).",
)
_FORMAT_OPT = click.option(
    "--format",
    "-f",
    "fmt",
    default="table",
    type=click.Choice(["table", "json", "csv"]),
    help="Output format.",
)


# ---------------------------------------------------------------------------- group


@click.group()
def pacs() -> None:
    """Generic PACS access (Orthanc REST / DICOMweb / DIMSE)."""


# ---------------------------------------------------------------------------- profiles


@pacs.command("profiles")
@_CONFIG_OPT
def profiles_cmd(config_path: Optional[str]) -> None:
    """List configured PACS profiles."""
    profiles = load_profiles(config_path)
    if not profiles:
        click.echo(
            "No profiles found. Create ./qradiomics-pacs.yaml or "
            "~/.config/qradiomics/pacs.yaml, or set $QR_PACS_CONFIG."
        )
        return
    for name, prof in profiles.items():
        cfg: Any = prof.settings
        if prof.backend == "dimse":
            target = f"{cfg.get('host')}:{cfg.get('port')} {cfg.get('aet')}"
        else:
            target = cfg.get("base_url", "?")
        click.echo(f"  {name:<24s} backend={prof.backend:<10s} {target}")


# ---------------------------------------------------------------------------- ping


@pacs.command("ping")
@_PROFILE_OPT
@_CONFIG_OPT
def ping_cmd(profile: Optional[str], config_path: Optional[str]) -> None:
    """Verify connectivity (HTTP /system, QIDO probe, or C-ECHO)."""
    try:
        with get_backend(profile, config_path) as backend:
            ok = backend.ping()
    except PACSError as exc:
        click.echo(f"FAIL: {exc}", err=True)
        sys.exit(2)
    click.echo("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------- query


@pacs.group("query")
def query_group() -> None:
    """Query metadata via QIDO-RS / C-FIND."""


@query_group.command("studies")
@_PROFILE_OPT
@_CONFIG_OPT
@_FORMAT_OPT
@click.option("--patient-id", help="Filter by PatientID.")
@click.option("--modality", help="Filter by ModalitiesInStudy.")
@click.option(
    "--study-date",
    help="StudyDate (YYYYMMDD or YYYYMMDD-YYYYMMDD).",
)
@click.option(
    "--filter",
    "extra",
    multiple=True,
    help="Extra KEY=VALUE filter (repeatable).",
)
def query_studies_cmd(
    profile: Optional[str],
    config_path: Optional[str],
    fmt: str,
    patient_id: Optional[str],
    modality: Optional[str],
    study_date: Optional[str],
    extra: tuple,
) -> None:
    """Query studies."""
    filters = _kv_to_dict(extra)
    if patient_id:
        filters["PatientID"] = patient_id
    if modality:
        filters["ModalitiesInStudy"] = modality
    if study_date:
        filters["StudyDate"] = study_date
    with get_backend(profile, config_path) as backend:
        rows = backend.query_studies(**filters)
    _emit(
        rows,
        fmt,
        columns=[
            "PatientID",
            "StudyInstanceUID",
            "StudyDate",
            "StudyDescription",
            "ModalitiesInStudy",
        ],
    )


@query_group.command("series")
@_PROFILE_OPT
@_CONFIG_OPT
@_FORMAT_OPT
@click.option("--study-uid", help="StudyInstanceUID to scope the search.")
@click.option("--patient-id", help="Filter by PatientID.")
@click.option("--modality", help="Filter by Modality.")
@click.option("--filter", "extra", multiple=True)
def query_series_cmd(
    profile: Optional[str],
    config_path: Optional[str],
    fmt: str,
    study_uid: Optional[str],
    patient_id: Optional[str],
    modality: Optional[str],
    extra: tuple,
) -> None:
    """Query series."""
    filters = _kv_to_dict(extra)
    if patient_id:
        filters["PatientID"] = patient_id
    if modality:
        filters["Modality"] = modality
    with get_backend(profile, config_path) as backend:
        rows = backend.query_series(study_uid=study_uid, **filters)
    _emit(
        rows,
        fmt,
        columns=[
            "PatientID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SeriesNumber",
            "Modality",
            "SeriesDescription",
            "NumberOfSeriesRelatedInstances",
        ],
    )


@query_group.command("instances")
@_PROFILE_OPT
@_CONFIG_OPT
@_FORMAT_OPT
@click.option("--study-uid", required=True)
@click.option("--series-uid", required=True)
@click.option("--filter", "extra", multiple=True)
def query_instances_cmd(
    profile: Optional[str],
    config_path: Optional[str],
    fmt: str,
    study_uid: str,
    series_uid: str,
    extra: tuple,
) -> None:
    """Query instances."""
    filters = _kv_to_dict(extra)
    with get_backend(profile, config_path) as backend:
        rows = backend.query_instances(study_uid, series_uid, **filters)
    _emit(rows, fmt, columns=["SOPInstanceUID", "InstanceNumber", "SOPClassUID"])


# ---------------------------------------------------------------------------- fetch


@pacs.command("fetch")
@_PROFILE_OPT
@_CONFIG_OPT
@click.option("--study-uid", help="StudyInstanceUID (required).")
@click.option("--series-uid", help="SeriesInstanceUID to limit the fetch.")
@click.option(
    "--instance-uid",
    help="SOPInstanceUID — fetch one instance only.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory (or file when --instance-uid is given).",
)
def fetch_cmd(
    profile: Optional[str],
    config_path: Optional[str],
    study_uid: Optional[str],
    series_uid: Optional[str],
    instance_uid: Optional[str],
    output: str,
) -> None:
    """Download DICOM data from the PACS."""
    if instance_uid and not (study_uid and series_uid):
        raise click.UsageError("--instance-uid requires --study-uid and --series-uid")
    if series_uid and not study_uid:
        raise click.UsageError("--series-uid requires --study-uid")
    if not study_uid:
        raise click.UsageError("provide at least --study-uid")
    out_path = Path(output)
    with get_backend(profile, config_path) as backend:
        if instance_uid:
            written = backend.fetch_instance(
                study_uid, series_uid or "", instance_uid, out_path
            )
            click.echo(str(written))
        elif series_uid:
            paths = backend.fetch_series(study_uid, series_uid, out_path)
            for path in paths:
                click.echo(str(path))
            click.echo(f"[{len(paths)} instances] {out_path}", err=True)
        else:
            tree = backend.fetch_study(study_uid, out_path)
            total = sum(len(v) for v in tree.values())
            click.echo(
                f"[{len(tree)} series, {total} instances] {out_path}",
                err=True,
            )


# ---------------------------------------------------------------------------- send


@pacs.command("send")
@_PROFILE_OPT
@_CONFIG_OPT
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="DICOM file or directory.",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Recurse when --input is a directory.",
)
def send_cmd(
    profile: Optional[str],
    config_path: Optional[str],
    input_path: str,
    recursive: bool,
) -> None:
    """Upload DICOM file(s) (STOW-RS / Orthanc /instances / C-STORE)."""
    path = Path(input_path)
    with get_backend(profile, config_path) as backend:
        if path.is_file():
            result = backend.store_dicom(path)
            click.echo(json.dumps(result, indent=2, default=str))
        else:
            results = backend.store_directory(path, recursive=recursive)
            ok = sum(1 for r in results if r.get("success", True))
            click.echo(f"Uploaded {ok}/{len(results)} files", err=True)
