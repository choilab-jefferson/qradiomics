"""PACS connection profiles — YAML config with ``${VAR[:-default]}`` expansion.

Search order for the config file:

1. Path passed to :func:`load_profile` / :func:`load_profiles`.
2. ``$QR_PACS_CONFIG`` environment variable.
3. ``./qradiomics-pacs.yaml`` in the current working directory.
4. ``./.qradiomics/pacs.yaml``.
5. ``~/.config/qradiomics/pacs.yaml``.

Example file::

    default_profile: local_orthanc
    profiles:
      local_orthanc:
        backend: orthanc
        base_url: http://localhost:8042
        username: orthanc
        password: ${ORTHANC_PASSWORD:-}
      hospital_dimse:
        backend: dimse
        host: pacs.hospital.org
        port: 4242
        aet: HOSPITAL_PACS
        aec: QRADIOMICS
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""

    def repl(match: re.Match) -> str:
        var, default = match.group(1), match.group(2)
        return os.environ.get(var, "" if default is None else default)

    if isinstance(value, str):
        return _ENV_RE.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


@dataclass
class PACSProfile:
    """One backend instance described in YAML."""

    name: str
    backend: str
    settings: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, raw: Dict[str, Any]) -> "PACSProfile":
        raw = dict(raw)
        backend = raw.pop("backend", None)
        if not backend:
            raise ValueError(f"profile '{name}' missing 'backend' field")
        return cls(name=name, backend=str(backend).lower(), settings=raw)


def _default_search_paths() -> tuple[Path, ...]:
    """Search paths re-evaluated per call so tests can chdir into tmp dirs."""
    return (
        Path.cwd() / "qradiomics-pacs.yaml",
        Path.cwd() / ".qradiomics" / "pacs.yaml",
        Path.home() / ".config" / "qradiomics" / "pacs.yaml",
    )


def find_config_file(
    explicit: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Return the active config path, or ``None`` if none exists."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    env = os.environ.get("QR_PACS_CONFIG")
    if env:
        path = Path(env).expanduser()
        if path.exists():
            return path
    for cand in _default_search_paths():
        if cand.exists():
            return cand
    return None


def _read_raw(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def load_profiles(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, PACSProfile]:
    """Return every profile defined in the active config file."""
    path = find_config_file(config_path)
    if path is None:
        return {}
    raw = _read_raw(path)
    profiles_raw = raw.get("profiles") or {}
    return {
        name: PACSProfile.from_dict(name, _expand_env(cfg))
        for name, cfg in profiles_raw.items()
    }


def load_profile(
    name: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> PACSProfile:
    """Look up a named profile (or the file's ``default_profile``)."""
    profiles = load_profiles(config_path)
    if not profiles:
        raise FileNotFoundError(
            "No PACS profiles found. Create qradiomics-pacs.yaml in the "
            "working directory or set $QR_PACS_CONFIG."
        )
    if name is None:
        path = find_config_file(config_path)
        if path is not None:
            raw = _read_raw(path)
            name = raw.get("default_profile") or next(iter(profiles))
        else:
            name = next(iter(profiles))
    if name not in profiles:
        raise KeyError(
            f"Unknown PACS profile '{name}'. Available: {sorted(profiles)}"
        )
    return profiles[name]
