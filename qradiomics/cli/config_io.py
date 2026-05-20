"""CLI common utilities."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CLIConfig:
    """CLI configuration."""

    verbose: bool = False
    log_level: str = "INFO"
    project_root: str = field(default_factory=lambda: os.environ.get("PROJECT_ROOT", os.getcwd()))


def load_cli_config(config_path: Optional[str] = None) -> CLIConfig:
    """Load CLI configuration from environment, then optional YAML file."""
    config = CLIConfig()
    config.verbose = os.getenv("QRADIOMICS_VERBOSE", "false").lower() == "true"
    config.log_level = os.getenv("QRADIOMICS_LOG_LEVEL", config.log_level)

    if config_path and os.path.exists(config_path):
        try:
            import yaml

            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            for key, value in file_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except Exception:
            pass

    return config


def save_cli_config(config: CLIConfig, config_path: str) -> None:
    """Persist CLI configuration to a YAML file."""
    import yaml

    payload = {
        "verbose": config.verbose,
        "log_level": config.log_level,
    }
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(payload, f)
