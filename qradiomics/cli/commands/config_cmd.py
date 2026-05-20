"""User-config management for the qr CLI.

Stores user preferences (verbose, log_level) in ~/.qradiomics/config.yaml.
"""
from __future__ import annotations

from pathlib import Path

import click

from qradiomics.cli.config_io import load_cli_config, save_cli_config

DEFAULT_CONFIG_PATH = Path.home() / ".qradiomics" / "config.yaml"


@click.group()
def config():
    """Manage user CLI preferences (~/.qradiomics/config.yaml)."""


@config.command()
@click.option(
    "--path",
    type=click.Path(),
    default=None,
    help="Optional explicit config path (defaults to ~/.qradiomics/config.yaml)",
)
def show(path):
    """Show current CLI configuration."""
    cfg = load_cli_config(path or (str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None))
    click.echo("QRadiomics CLI configuration:")
    click.echo(f"  verbose:      {cfg.verbose}")
    click.echo(f"  log_level:    {cfg.log_level}")
    click.echo(f"  project_root: {cfg.project_root}")
    if DEFAULT_CONFIG_PATH.exists():
        click.echo(f"\nLoaded from: {DEFAULT_CONFIG_PATH}")
    else:
        click.echo(f"\nNo config file at {DEFAULT_CONFIG_PATH} (using env defaults)")


@config.command()
@click.argument("key")
@click.argument("value")
def set(key, value):
    """Set a configuration key (verbose | log_level)."""
    if key not in ("verbose", "log_level"):
        click.echo(f"Unknown key '{key}'. Allowed: verbose, log_level", err=True)
        raise SystemExit(1)

    cfg = load_cli_config(str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None)
    if key == "verbose":
        setattr(cfg, key, value.lower() in ("true", "1", "yes", "on"))
    else:
        setattr(cfg, key, value)
    save_cli_config(cfg, str(DEFAULT_CONFIG_PATH))
    click.echo(f"Saved {key}={getattr(cfg, key)} to {DEFAULT_CONFIG_PATH}")


@config.command()
@click.argument("key")
def get(key):
    """Print a single configuration value."""
    cfg = load_cli_config(str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None)
    if not hasattr(cfg, key):
        click.echo(f"Unknown key '{key}'", err=True)
        raise SystemExit(1)
    click.echo(getattr(cfg, key))


@config.command()
def path():
    """Print the default config file path."""
    click.echo(str(DEFAULT_CONFIG_PATH))
