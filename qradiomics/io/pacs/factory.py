"""Resolve a profile name to a live :class:`PACSBackend` instance."""

from __future__ import annotations

from typing import Any, Optional, Union
from pathlib import Path

from .base import PACSBackend, PACSUnsupportedError
from .config import PACSProfile, load_profile


_BACKEND_ALIASES = {
    "orthanc": "orthanc",
    "orthanc-rest": "orthanc",
    "dicomweb": "dicomweb",
    "wado": "dicomweb",
    "qido": "dicomweb",
    "dimse": "dimse",
    "pynetdicom": "dimse",
    "dicom": "dimse",
}


def make_backend(profile: PACSProfile) -> PACSBackend:
    """Instantiate the backend class implied by ``profile.backend``."""
    kind = _BACKEND_ALIASES.get(profile.backend)
    if kind == "orthanc":
        from .orthanc import OrthancBackend

        return OrthancBackend(name=profile.name, **profile.settings)
    if kind == "dicomweb":
        from .dicomweb import DICOMwebBackend

        return DICOMwebBackend(name=profile.name, **profile.settings)
    if kind == "dimse":
        from .dimse import DIMSEBackend

        return DIMSEBackend(name=profile.name, **profile.settings)
    raise PACSUnsupportedError(
        f"Unknown PACS backend '{profile.backend}' "
        f"(known: {sorted(set(_BACKEND_ALIASES.values()))})"
    )


def get_backend(
    profile_name: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    **overrides: Any,
) -> PACSBackend:
    """Look up a profile by name and instantiate its backend.

    ``overrides`` shallow-merge into ``profile.settings`` before
    instantiation — handy when a CLI flag should win over the YAML value.
    """
    profile = load_profile(profile_name, config_path)
    if overrides:
        profile.settings = {**profile.settings, **overrides}
    return make_backend(profile)
