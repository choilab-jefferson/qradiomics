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
    qr pacs modalities               List remote modalities (Orthanc).
    qr pacs retrieve                 C-MOVE from a remote modality via Orthanc.
    qr pacs watch                    Stream Orthanc /changes for push-driven flows.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import click

from qradiomics.io.pacs import PACSError, get_backend, load_profiles
from qradiomics.io.pacs.orthanc import OrthancBackend


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


# ---------------------------------------------------------------------------- Orthanc-only helpers


def _require_orthanc(profile: Optional[str], config_path: Optional[str]) -> OrthancBackend:
    backend = get_backend(profile, config_path)
    if not isinstance(backend, OrthancBackend):
        backend.close()
        raise click.UsageError(
            f"This subcommand needs an Orthanc REST profile; '{profile}' "
            f"resolves to backend={backend.name!r}."
        )
    return backend


@pacs.command("modalities")
@_PROFILE_OPT
@_CONFIG_OPT
@_FORMAT_OPT
def modalities_cmd(
    profile: Optional[str], config_path: Optional[str], fmt: str
) -> None:
    """List remote modalities registered on the Orthanc profile."""
    with _require_orthanc(profile, config_path) as backend:
        rows = []
        for name in backend.list_modalities():
            try:
                cfg = backend.get_modality_config(name)
            except PACSError as exc:
                cfg = {"_error": str(exc)}
            rows.append(
                {
                    "Name": name,
                    "AET": cfg.get("AET"),
                    "Host": cfg.get("Host"),
                    "Port": cfg.get("Port"),
                    "Manufacturer": cfg.get("Manufacturer"),
                }
            )
    _emit(rows, fmt, columns=["Name", "AET", "Host", "Port", "Manufacturer"])


@pacs.command("retrieve")
@_PROFILE_OPT
@_CONFIG_OPT
@click.option(
    "--remote",
    "-R",
    required=True,
    help="Remote modality name as registered on Orthanc (e.g. MIM).",
)
@click.option(
    "--level",
    type=click.Choice(["Patient", "Study", "Series", "Instance"]),
    default="Study",
    help="C-FIND/C-MOVE level (default: Study).",
)
@click.option("--patient-id", help="Filter by PatientID.")
@click.option("--study-uid", help="Filter by StudyInstanceUID.")
@click.option("--series-uid", help="Filter by SeriesInstanceUID (requires --level Series or Instance).")
@click.option("--modality", help="Filter by Modality (Series-level).")
@click.option(
    "--filter",
    "extra",
    multiple=True,
    help="Extra KEY=VALUE filter (repeatable).",
)
@click.option(
    "--target",
    "target_aet",
    default=None,
    help="C-MOVE destination AET (default: this Orthanc's own DicomAet).",
)
@click.option(
    "--async",
    "asynchronous",
    is_flag=True,
    help="Return immediately with the Orthanc job handle.",
)
def retrieve_cmd(
    profile: Optional[str],
    config_path: Optional[str],
    remote: str,
    level: str,
    patient_id: Optional[str],
    study_uid: Optional[str],
    series_uid: Optional[str],
    modality: Optional[str],
    extra: tuple,
    target_aet: Optional[str],
    asynchronous: bool,
) -> None:
    """C-MOVE from a remote modality via Orthanc.

    Orthanc issues C-FIND against ``--remote`` with the given filters, then
    C-MOVE every answer to ``--target`` (defaults to Orthanc's own AET, so
    the data arrives in *this* Orthanc — fetch it locally afterwards with
    ``qr pacs fetch -P <orthanc-profile> --study-uid ...``).
    """
    filters: Dict[str, Any] = _kv_to_dict(extra)
    if patient_id:
        filters["PatientID"] = patient_id
    if study_uid:
        filters["StudyInstanceUID"] = study_uid
    if series_uid:
        filters["SeriesInstanceUID"] = series_uid
    if modality:
        filters["Modality"] = modality
    if not filters:
        raise click.UsageError(
            "At least one filter required (--patient-id, --study-uid, etc.)"
        )
    with _require_orthanc(profile, config_path) as backend:
        result = backend.retrieve_from(
            remote,
            level=level,
            target_aet=target_aet,
            synchronous=not asynchronous,
            **filters,
        )
    click.echo(json.dumps(result, indent=2, default=str))
    answer_count = len(result.get("answers", []))
    click.echo(
        f"[{answer_count} answers, C-MOVE {'queued' if asynchronous else 'completed'}]",
        err=True,
    )


@pacs.command("watch")
@_PROFILE_OPT
@_CONFIG_OPT
@click.option(
    "--since",
    type=int,
    default=0,
    help="Resume from this Orthanc change sequence id (default: 0).",
)
@click.option(
    "--type",
    "change_types",
    multiple=True,
    help="Filter to one of: NewInstance, NewSeries, NewStudy, StableStudy, "
    "StablePatient, etc. (repeatable). Default: all.",
)
@click.option(
    "--match",
    "tag_matches",
    multiple=True,
    help="Tag-match filter (rule-style): KEY=VALUE, repeatable. The resource "
    "is fetched and its simplified DICOM tags must match every entry. "
    "Glob wildcards (*, ?) allowed. Costs one extra HTTP call per event.",
)
@click.option(
    "--interval",
    type=float,
    default=5.0,
    help="Polling interval in seconds (default 5).",
)
@click.option(
    "--once",
    is_flag=True,
    help="Drain pending changes and exit instead of looping forever.",
)
@click.option(
    "--state-file",
    type=click.Path(),
    default=None,
    help="Persist the last-seen sequence id here so re-runs resume.",
)
def watch_cmd(
    profile: Optional[str],
    config_path: Optional[str],
    since: int,
    change_types: tuple,
    tag_matches: tuple,
    interval: float,
    once: bool,
    state_file: Optional[str],
) -> None:
    """Stream Orthanc /changes — one NDJSON line per matched event.

    Pairs naturally with MIM-Assistant-style push-to-start workflows:
    external SCUs C-STORE into this Orthanc, ``qr pacs watch`` emits the
    resource IDs, downstream commands (``qr pacs fetch``, ``qr extract``)
    consume them line by line. ``--match`` adds rule-style tag filtering
    (multi-patient gate) — workflow-style single-patient handling is left
    to the consumer of the emitted events.
    """
    import fnmatch

    type_filter = set(change_types) if change_types else None
    tag_filter = _kv_to_dict(tag_matches) if tag_matches else None
    state_path = Path(state_file).expanduser() if state_file else None
    if state_path and state_path.exists():
        since = int(state_path.read_text().strip() or since)

    def fetch_tags(backend: OrthancBackend, event: dict) -> dict:
        kind = event.get("ResourceType")
        rid = event.get("ID")
        if not (kind and rid):
            return {}
        path_map = {
            "Study": f"/studies/{rid}/shared-tags?simplify",
            "Series": f"/series/{rid}/shared-tags?simplify",
            "Patient": f"/patients/{rid}/shared-tags?simplify",
            "Instance": f"/instances/{rid}/simplified-tags",
        }
        path = path_map.get(kind)
        if not path:
            return {}
        try:
            return backend._get(path).json()
        except PACSError:
            return {}

    def tags_match(tags: dict) -> bool:
        if not tag_filter:
            return True
        for key, pattern in tag_filter.items():
            value = str(tags.get(key, ""))
            if not fnmatch.fnmatch(value, pattern):
                return False
        return True

    with _require_orthanc(profile, config_path) as backend:
        while True:
            page = backend.get_changes(since=since, limit=500)
            for event in page.get("Changes", []):
                if type_filter and event.get("ChangeType") not in type_filter:
                    continue
                if tag_filter:
                    tags = fetch_tags(backend, event)
                    if not tags_match(tags):
                        continue
                    event = {**event, "Tags": tags}
                click.echo(json.dumps(event, default=str), nl=True)
            last = int(page.get("Last", since))
            if last > since:
                since = last
                if state_path:
                    state_path.write_text(str(since))
            sys.stdout.flush()
            if once and page.get("Done", True):
                return
            time.sleep(max(0.1, interval))
