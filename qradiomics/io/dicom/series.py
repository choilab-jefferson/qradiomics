"""DICOM image-series loading via SimpleITK / GDCM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import SimpleITK as sitk

__all__ = ["SeriesInfo", "list_series_in_directory", "load_dicom_series"]


@dataclass(frozen=True)
class SeriesInfo:
    """Lightweight summary of one DICOM series in a directory."""

    series_uid: str
    n_slices: int
    file_paths: List[str]


def list_series_in_directory(directory: Union[str, Path]) -> List[SeriesInfo]:
    """Enumerate every DICOM ``SeriesInstanceUID`` under ``directory``.

    Recurses into subdirectories. Each returned :class:`SeriesInfo`
    pairs a unique series UID with the file paths SimpleITK would use
    when reading it. The list is stable-ordered by ``series_uid``.
    """
    directory = str(directory)
    reader = sitk.ImageSeriesReader()
    series_uids = reader.GetGDCMSeriesIDs(directory)
    out: List[SeriesInfo] = []
    for uid in sorted(series_uids):
        files = list(reader.GetGDCMSeriesFileNames(directory, uid))
        out.append(SeriesInfo(series_uid=uid, n_slices=len(files), file_paths=files))
    return out


def load_dicom_series(
    directory: Union[str, Path],
    *,
    series_uid: Optional[str] = None,
) -> sitk.Image:
    """Read a DICOM series from disk and return a single SimpleITK image.

    When the directory contains multiple series (e.g. 4D-CT phases),
    the ``series_uid`` argument is required — picking the first one is
    almost always wrong and can silently mix phases.

    Args:
        directory: Folder containing the DICOM files. Searched
            recursively by GDCM.
        series_uid: Optional ``SeriesInstanceUID`` to restrict the
            read. If omitted **and** the directory has exactly one
            series, that series is loaded. If omitted with multiple
            series present, a ``ValueError`` is raised.
    """
    directory = str(directory)
    reader = sitk.ImageSeriesReader()
    if series_uid is None:
        uids = reader.GetGDCMSeriesIDs(directory)
        if not uids:
            raise ValueError(f"No DICOM series found under {directory!r}.")
        if len(uids) > 1:
            raise ValueError(
                f"Directory {directory!r} contains {len(uids)} series — "
                f"pass series_uid explicitly. UIDs: {sorted(uids)}"
            )
        series_uid = sorted(uids)[0]
    files = reader.GetGDCMSeriesFileNames(directory, series_uid)
    if not files:
        raise ValueError(
            f"SeriesInstanceUID {series_uid!r} produced 0 files under {directory!r}."
        )
    reader.SetFileNames(files)
    return reader.Execute()
