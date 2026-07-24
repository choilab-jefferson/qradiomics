"""`qr models` — manage external pretrained models.

Recovers the command group lost during the public-prep PHI+PACS backport
(commit 7dd09ce); main.py still imports `models_cmd` and `cli.add_command(models)`
so the import itself blocks the CLI from starting. This minimal stub keeps
the command tree intact and surfaces a clear "not implemented" message for
each subcommand the previous implementation exposed (segmentation /
classification / cardiac toxicity model registries). Concrete handlers will
be restored when the model-registry feature lands again.
"""
from __future__ import annotations

import click


@click.group()
def models() -> None:
    """Manage external pretrained models (e.g. segmentation, classification).

    This command group is a stub — the underlying registry was lost during a
    backport and is being restored separately. Run `qr models <subcommand>`
    to see the placeholder message for the previously-shipped subcommands.
    """


@models.command("list")
def _list() -> None:
    """List registered external models."""
    ctx = click.get_current_context()
    click.echo(
        f"{ctx.command_path}: registry not yet restored.\n"
        "Tracking: pending re-introduction of models_cmd handlers."
    )


@models.command("pull")
@click.argument("name")
def _pull(name: str) -> None:
    """Fetch a registered model by name."""
    ctx = click.get_current_context()
    click.echo(
        f"{ctx.command_path} '{name}': registry not yet restored.\n"
        "Tracking: pending re-introduction of models_cmd handlers."
    )


@models.command("info")
@click.argument("name")
def _info(name: str) -> None:
    """Show metadata for a registered model."""
    ctx = click.get_current_context()
    click.echo(
        f"{ctx.command_path} '{name}': registry not yet restored.\n"
        "Tracking: pending re-introduction of models_cmd handlers."
    )
