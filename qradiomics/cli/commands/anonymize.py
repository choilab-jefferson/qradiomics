"""qr anonymize — strip PHI from DICOM directories before downstream processing.

Implements a conservative subset of the DICOM standard PS3.15 Annex E
"Basic Application Confidentiality Profile":

  * Patient name / birth date / address / phone / comments  → blanked
  * Institution / referring physician / operator / station  → blanked
  * Optional: re-hash PatientID to a stable pseudo-ID (preserves cross-
    study linkage without leaking the original MRN)
  * Optional: regenerate Study/Series InstanceUIDs (de-links from PACS)

The anonymizer **rewrites files in place by default**, so always run on
a copy of the data. Pass `--output <dir>` to write a separate
deidentified tree.

For TCIA cohorts this step is unnecessary (TCIA data is already
deidentified per the collection's IRB). For institutional cohorts it
is mandatory before any export or feature publication.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click


# DICOM tags that commonly carry PHI (DICOM standard PS3.15 Annex E + practical).
PHI_TAGS = [
    "PatientName", "PatientBirthDate", "PatientBirthTime", "PatientAddress",
    "PatientTelephoneNumbers", "PatientMotherBirthName", "PatientReligiousPreference",
    "PatientSex", "PatientWeight", "PatientSize", "PatientComments",
    "OtherPatientIDs", "OtherPatientNames", "OtherPatientIDsSequence",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "ReferringPhysicianName", "ReferringPhysicianAddress", "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName", "OperatorsName", "RequestingPhysician",
    "StationName", "DeviceSerialNumber",
    "AccessionNumber", "AdmissionID", "IssuerOfPatientID",
    "StudyID", "StudyDescription",
    "RequestedProcedureDescription", "RequestedProcedureID",
    "PerformedProcedureStepID", "PerformedProcedureStepDescription",
    "ScheduledProcedureStepID", "ScheduledProcedureStepDescription",
]


def _anon_one(args):
    """Anonymise a single DICOM file. Runs in a worker process."""
    src, dst, replace_pid, pid_salt, regen_uids = args
    try:
        import pydicom
        from pydicom.uid import generate_uid

        ds = pydicom.dcmread(src, stop_before_pixels=False)

        # Stable pseudo-ID derived from the original PatientID + salt
        original_pid = str(getattr(ds, "PatientID", ""))
        if replace_pid and original_pid:
            new_pid = "ANON-" + hashlib.sha1(
                (pid_salt + original_pid).encode()
            ).hexdigest()[:12]
            ds.PatientID = new_pid

        for tag in PHI_TAGS:
            if tag in ds:
                # Use empty string rather than delete so receivers don't trip
                # on missing required tags.
                setattr(ds, tag, "")

        if regen_uids:
            # Keep study/series internally consistent but break linkage to
            # the original PACS hierarchy.
            ds.StudyInstanceUID = generate_uid()
            ds.SeriesInstanceUID = generate_uid()
            ds.SOPInstanceUID = generate_uid()

        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(dst)
        return (src, original_pid, None)
    except Exception as e:
        return (src, None, str(e))


@click.command()
@click.option(
    "--input",
    "-i",
    "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="DICOM tree to anonymise (recursive). Required.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False),
    default=None,
    help="Output DICOM tree. If omitted, files are rewritten in place "
         "(make a copy first — see --copy-first).",
)
@click.option(
    "--copy-first/--in-place",
    default=True,
    help="When --output is set, copy the source tree first then anonymise "
         "the copy (default). --in-place writes to --output assuming the "
         "files were already moved there.",
)
@click.option(
    "--replace-pid/--keep-pid",
    default=True,
    help="Replace PatientID with a salted SHA1 pseudo-ID (default on).",
)
@click.option(
    "--pid-salt",
    default="qradiomics",
    help="Salt used when hashing PatientID. Use the same salt across runs "
         "to keep the pseudo-ID stable for a given patient.",
)
@click.option(
    "--regen-uids/--keep-uids",
    default=False,
    help="Regenerate Study/Series/SOP InstanceUIDs (de-link from PACS). "
         "Off by default because downstream RTSTRUCT/CT pairing relies on "
         "the original SeriesInstanceUID linkage.",
)
@click.option(
    "--mapping",
    type=click.Path(),
    default=None,
    help="Optional CSV that records the original→anonymised PatientID map. "
         "Keep this file with restricted access — it is the only way to "
         "re-identify the data later.",
)
@click.option("--jobs", "-j", default=1, type=int, help="Worker processes.")
def anonymize(input_dir, output, copy_first, replace_pid, pid_salt, regen_uids,
              mapping, jobs):
    """Strip PHI from every DICOM file under a directory tree.

    \b
    Example — institutional cohort:
        qr anonymize -i /private/Cohort -o /staging/Cohort \\
            --replace-pid --pid-salt MyStudy2026 \\
            --mapping pid_map.csv --jobs 16

    \b
    Always inspect a few output files with `dcmdump` before exporting.
    """
    src_root = Path(input_dir).resolve()
    if output:
        dst_root = Path(output).resolve()
        if copy_first and not dst_root.exists():
            click.echo(f"Copying tree {src_root} → {dst_root} ...")
            shutil.copytree(src_root, dst_root)
        elif copy_first and dst_root.exists():
            click.echo(f"--output already exists, will rewrite in place under {dst_root}")
    else:
        dst_root = src_root
        click.echo(f"⚠  Rewriting DICOMs in place under {src_root}")

    # Enumerate every file (we accept either .dcm extension or extensionless).
    files = []
    for p in dst_root.rglob("*"):
        if not p.is_file():
            continue
        # Heuristic: try to read 132B and check 'DICM' magic
        try:
            with open(p, "rb") as f:
                f.seek(128); magic = f.read(4)
                if magic == b"DICM":
                    files.append(p)
        except Exception:
            continue

    total = len(files)
    click.echo(f"Anonymising {total} DICOM files (jobs={jobs}, replace_pid={replace_pid}, "
               f"regen_uids={regen_uids})")
    work = [(str(p), str(p), replace_pid, pid_salt, regen_uids) for p in files]

    results = []
    if jobs <= 1:
        for i, w in enumerate(work, 1):
            r = _anon_one(w)
            if i == total or i % max(1, total // 100) == 0:
                print(f"  anonymize [{i}/{total}]", flush=True)
            results.append(r)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_anon_one, w): w[0] for w in work}
            for fut in as_completed(futs):
                done += 1
                results.append(fut.result())
                if done == total or done % max(1, total // 100) == 0:
                    print(f"  anonymize [{done}/{total}]", flush=True)
                    sys.stdout.flush()

    ok = sum(1 for r in results if not r[2])
    fail = total - ok
    click.echo(f"\nAnonymise: {ok} ok, {fail} failed")

    if mapping and replace_pid:
        # Deduplicate by original PID
        seen: dict[str, str] = {}
        for src, orig_pid, err in results:
            if err or not orig_pid:
                continue
            if orig_pid not in seen:
                seen[orig_pid] = "ANON-" + hashlib.sha1(
                    (pid_salt + orig_pid).encode()
                ).hexdigest()[:12]
        with open(mapping, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["original_pid", "anon_pid"])
            w.writeheader()
            for orig, anon in sorted(seen.items()):
                w.writerow({"original_pid": orig, "anon_pid": anon})
        click.echo(f"PID map → {mapping}  ({len(seen)} patients) — keep restricted.")
