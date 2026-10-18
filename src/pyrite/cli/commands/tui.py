"""Lazy ``pyrite tui`` entry point for the optional terminal detector editor."""

import sys

import click

from pyrite.cli import _catalog_io
from pyrite.cli._extras import require_cli_extra
from pyrite.console.output import CLIError


@click.command("tui")
def command():
    """Open the terminal detector editor (prototype; needs the tui extra).

    Browse scalar and planar detectors with their source layer, create one,
    clone a bundled entry into the writable layer, or edit a writable entry.
    Fields are the `pyrite detector create` options with the same units and
    validation. Typing, cancel and invalid input write nothing; Save writes
    only after you review the TOML diff. Edits go where `pyrite detector`
    writes: the user layer for the bundled catalog, else the selected catalog.

    Needs an interactive terminal. Scripts use `pyrite detector`; plots stay
    in `pyrite app`. Install with: pip install "pyrite-mc[tui]".

    \b
    Examples:
      pyrite tui
      pyrite --catalog ./lab-catalog tui
    """
    require_cli_extra("tui", "textual", feature="pyrite tui")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise click.UsageError(
            "pyrite tui needs an interactive terminal; use pyrite detector for scripts"
        )
    from pyrite.tui.detector_app import run

    try:
        return run(_catalog_io.active_catalog_path())
    except (OSError, ValueError) as exc:  # unreadable or invalid selected catalog
        raise CLIError(str(exc)) from None
