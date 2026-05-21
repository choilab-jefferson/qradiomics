"""qradiomics.io.pacs — generic PACS access.

Three swappable backends share one ABC (:class:`PACSBackend`):

* :class:`~qradiomics.io.pacs.orthanc.OrthancBackend`   - Orthanc REST API
* :class:`~qradiomics.io.pacs.dicomweb.DICOMwebBackend` - DICOMweb (QIDO/WADO/STOW)
* :class:`~qradiomics.io.pacs.dimse.DIMSEBackend`       - DIMSE (C-FIND/C-GET/C-STORE)

Profiles live in a YAML file (``qradiomics-pacs.yaml`` in CWD,
``~/.config/qradiomics/pacs.yaml``, or ``$QR_PACS_CONFIG``); pick one with
:func:`get_backend`.
"""

from .base import (
    PACSBackend,
    PACSConnectionError,
    PACSError,
    PACSNotFoundError,
    PACSUnsupportedError,
)
from .config import PACSProfile, find_config_file, load_profile, load_profiles
from .factory import get_backend, make_backend

__all__ = [
    "PACSBackend",
    "PACSError",
    "PACSConnectionError",
    "PACSNotFoundError",
    "PACSUnsupportedError",
    "PACSProfile",
    "find_config_file",
    "load_profile",
    "load_profiles",
    "get_backend",
    "make_backend",
]
