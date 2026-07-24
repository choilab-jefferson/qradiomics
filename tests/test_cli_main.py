"""CLI-level smoke tests for the `qr` entrypoint (qradiomics.cli.main).

Covers Phase 1 of the CLI reorg (qradiomics#9):
  * `qr ml models` — new alias of the top-level `qr models` stub under the
    `ml` group.
  * `qr private` — the previously-dead `register_private_commands()` wiring
    (only reachable when the optional `qradiomics_private` overlay is
    installed; degrades to "command not found" otherwise).

These are `--help`-only smoke tests: they assert the command tree is wired
correctly and importable, not that the underlying business logic behaves
(that is covered by module-specific tests).

Note: the `qr phi` group (and its tests) is a dev-only feature backed by the
`qradiomics.phi` package, which this public package does not ship -- see
PUBLIC_MANIFEST.txt.
"""
from __future__ import annotations

import importlib.util

from click.testing import CliRunner

from qradiomics.cli.main import cli

_PRIVATE_OVERLAY_INSTALLED = importlib.util.find_spec("qradiomics_private") is not None


def _invoke(*args):
    runner = CliRunner()
    return runner.invoke(cli, list(args))


# ---------------------------------------------------------------------------
# New alias paths (additive — Phase 1 of qradiomics#9)
# ---------------------------------------------------------------------------

def test_ml_models_help():
    result = _invoke("ml", "models", "--help")
    assert result.exit_code == 0, result.output
    assert "Manage external pretrained models" in result.output


def test_ml_models_pull_shows_alias_command_path():
    """`qr ml models pull` and `qr models pull` share the same underlying
    Click command object (registered twice under different groups); the
    stub message must reflect whichever path actually invoked it, not a
    hardcoded 'qr models pull'."""
    result = _invoke("ml", "models", "pull", "foo")
    assert result.exit_code == 0, result.output
    assert "ml models pull 'foo'" in result.output
    assert "cli models pull 'foo'" not in result.output


def test_top_level_models_pull_shows_top_level_command_path():
    result = _invoke("models", "pull", "foo")
    assert result.exit_code == 0, result.output
    assert "cli models pull 'foo'" in result.output
    assert "ml models pull 'foo'" not in result.output


def test_ml_models_info_shows_alias_command_path():
    result = _invoke("ml", "models", "info", "foo")
    assert result.exit_code == 0, result.output
    assert "ml models info 'foo'" in result.output


def test_ml_models_list_shows_alias_command_path():
    result = _invoke("ml", "models", "list")
    assert result.exit_code == 0, result.output
    assert "ml models" in result.output
    assert result.output.strip().startswith(("cli ml models", "qr ml models"))


def test_private_help():
    """`qr private --help` is reachable iff qradiomics_private is installed.

    register_private_commands() previously wasn't called by
    qradiomics/cli/main.py, so `qr private` was unreachable regardless of
    whether the overlay was installed. This confirms the wiring now works,
    and degrades cleanly (command not found, not a crash) when the overlay
    is absent.
    """
    result = _invoke("private", "--help")
    if _PRIVATE_OVERLAY_INSTALLED:
        assert result.exit_code == 0, result.output
        assert "Private overlay subcommands" in result.output
    else:
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_private_subcommands_help():
    """Whichever private subcommands are actually registered stay reachable."""
    if not _PRIVATE_OVERLAY_INSTALLED:
        return
    import click

    private_group = cli.commands["private"]
    assert isinstance(private_group, click.Group)
    subcommand_names = private_group.list_commands(click.Context(private_group))
    assert subcommand_names, "qr private has no registered subcommands"
    for name in subcommand_names:
        sub = _invoke("private", name, "--help")
        assert sub.exit_code == 0, f"qr private {name} --help failed:\n{sub.output}"


# ---------------------------------------------------------------------------
# Backward compatibility — old top-level commands must keep working unchanged
# ---------------------------------------------------------------------------

def test_top_level_anonymize_still_works():
    result = _invoke("anonymize", "--help")
    assert result.exit_code == 0, result.output
    assert "Strip PHI from every DICOM file" in result.output


def test_top_level_models_still_works():
    result = _invoke("models", "--help")
    assert result.exit_code == 0, result.output
    assert "Manage external pretrained models" in result.output


def test_top_level_help_lists_all_groups():
    result = _invoke("--help")
    assert result.exit_code == 0, result.output
    for name in ("anonymize", "models", "ml", "private"):
        assert name in result.output, f"'{name}' missing from `qr --help`"
