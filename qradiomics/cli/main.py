"""qradiomics CLI entrypoint — `qr` / `qradiomics` / `qrdx`."""

import os

import click

from qradiomics import __version__
from qradiomics.cli.commands import (
    analyze,
    anonymize,
    config,
    convert,
    delta,
    extract,
    hu_correct,
    lidc,
    ml,
    pacs,
    preprocess,
    register,
    results,
    shape,
    tcia,
    workflow,
)
from qradiomics.cli.commands.bench import bench
from qradiomics.cli.commands.models_cmd import models
from qradiomics.cli.config_io import load_cli_config
from qradiomics.cli.pattern import list_patterns, search_patterns

os.environ.setdefault("OTEL_SDK_DISABLED", "true")


@click.group()
@click.version_option(version=__version__, prog_name="qradiomics")
@click.option("--config", "-c", "config_path", help="User config file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, config_path, verbose):
    """Radiomics research toolkit.

    \b
    Canonical research flow:
      qr extract   -m manifest.csv -p <pattern> -o features.csv
      qr results merge  -f features.csv -c clinical.csv -o analysis_ready.csv
      qr analyze {survival,classify,importance} -i analysis_ready.csv ...

    Aliases: qradiomics, qr, qrdx
    """
    ctx.ensure_object(dict)
    cli_config = load_cli_config(config_path)
    if verbose:
        cli_config.verbose = True
    ctx.obj["config"] = cli_config


cli.add_command(tcia)
cli.add_command(anonymize)
cli.add_command(convert)
cli.add_command(preprocess)
cli.add_command(register)
cli.add_command(hu_correct)
cli.add_command(extract)
cli.add_command(shape)
cli.add_command(delta)
cli.add_command(results)
cli.add_command(analyze)
cli.add_command(lidc)
cli.add_command(ml)
cli.add_command(pacs)
cli.add_command(workflow)
cli.add_command(config)
cli.add_command(models)
cli.add_command(bench)

# `qr ml models` — alias of the top-level `qr models` stub under the `ml`
# group, since model management is a modeling-stage concern. Same Group
# object registered under both parents (Click has no single-parent
# restriction on commands), so `qr models` keeps working unchanged.
ml.add_command(models, name="models")


@cli.group()
def pattern():
    """Browse bundled pattern templates."""


pattern.add_command(list_patterns, name="list")
pattern.add_command(search_patterns, name="search")


# ---------------------------------------------------------------------------
# Private overlay (best-effort) — mirrors the guarded-import pattern used by
# the legacy `cli/main.py` entrypoint. When the optional `qradiomics_private`
# distribution is installed, it attaches `qr private ...` subcommands here.
# Public users (overlay absent) see exactly the commands defined above, with
# no error, no traceback, no missing-import crash.
# ---------------------------------------------------------------------------

try:
    from qradiomics_private.cli import register_private_commands  # type: ignore[import-not-found]

    register_private_commands(cli)
except ImportError:
    pass


@cli.command()
@click.pass_context
def info(ctx):
    """Show CLI version."""
    click.echo(f"qradiomics v{__version__}")
    click.echo("Aliases: qradiomics, qr, qrdx")
    click.echo(f"Verbose: {ctx.obj['config'].verbose}")


if __name__ == "__main__":
    cli()
