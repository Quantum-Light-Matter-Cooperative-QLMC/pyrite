"""``cxr`` Click command-line entry point."""

from __future__ import annotations

from collections.abc import Sequence

import click

from . import __version__
from ._cli_core import LazyGroup, run

_COMMANDS = {
    "scan": "cxr_mc.scan.command",
    "blaze": "cxr_mc.blaze.command",
    "export": "cxr_mc.export.command",
    "analyze": "cxr_mc.analyze.command",
    "slim": "cxr_mc.slim.command",
    "rebrem": "cxr_mc.rebrem.command",
    "reline": "cxr_mc.reline.command",
    "archive": "cxr_mc.archive.archive_command",
    "restore": "cxr_mc.archive.restore_command",
    "archives": "cxr_mc.archive.archives_command",
    "union": "cxr_mc.archive.union_command",
    "remote": "cxr_mc.remote.command",
    "line-grid": "cxr_mc.line_grid.command",
    "check": "cxr_mc.check.command",
    "check-config": "cxr_mc.check_config.command",
}

_COMMAND_HELP = {
    "scan": "Run one material's MC sweep and write a checkpoint.",
    "blaze": "Run a grooved-crystal sweep and write a checkpoint.",
    "export": "Export the analysis app as static HTML.",
    "analyze": "Launch the analysis app.",
    "slim": "Shrink a checkpoint for transfer.",
    "rebrem": "Recompute bremsstrahlung arrays in local checkpoints.",
    "reline": "Recompute line spectra in local checkpoints.",
    "archive": "Copy an active checkpoint to the archive shelf.",
    "restore": "Restore a shelved checkpoint to an active slot.",
    "archives": "List checkpoint archives.",
    "union": "Merge a shelved checkpoint into an active slot.",
    "remote": "Run and manage MC sweeps on a remote GPU host.",
    "line-grid": "Derive, submit, inspect, and apply line-energy grids.",
    "check": "Launch validation or export cached validation figures.",
    "check-config": "Validate a material catalog without starting simulation.",
}


@click.command(
    cls=LazyGroup,
    lazy_commands=_COMMANDS,
    lazy_help=_COMMAND_HELP,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="cxr-mc", message="cxr-mc %(version)s")
def command() -> None:
    """Coherent X-ray radiation (PXR + coherent bremsstrahlung) toolkit.

    Run ``cxr COMMAND --help`` for command options, units, defaults, and side
    effects.

    \b
    Examples:
      cxr scan mose2 --quick
      cxr analyze mose2
      cxr remote start mose2 --dry-run
    """


def main(argv: Sequence[str] | None = None):
    """Run ``cxr`` while preserving project exit-code and stream contracts."""
    result = run(command, argv, prog_name="cxr")
    if isinstance(result, int):
        raise SystemExit(result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
