"""Canonical grouped CLI for local checkpoint operations."""

import click

from ...console.output import run
from .._groups import LazyGroup

_COMMANDS = {
    "slim": "pyrite.cli.commands.slim.command",
    "recompute": "pyrite.cli.commands.checkpoint.recompute_command",
    "archive": "pyrite.cli.commands.archive.archive_group",
    "list": "pyrite.cli.commands.archive.list_command",
    "gc": "pyrite.cli.commands.cleanup.gc_command",
    "rm": "pyrite.cli.commands.cleanup.rm_command",
}

_COMMAND_HELP = {
    "slim": "Shrink one checkpoint for transfer.",
    "recompute": "Recompute selected checkpoint datasets.",
    "archive": "Manage the long-term checkpoint shelf.",
    "list": "List active local checkpoints.",
    "gc": "Reclaim records obsolete under current scan profiles.",
    "rm": "Delete local datasets and newly unreachable shared cases.",
}

_RECOMPUTE_COMMANDS = {
    "brem": "pyrite.cli.commands.recompute.brem_command",
    "cxr": "pyrite.cli.commands.recompute.cxr_command",
}

_RECOMPUTE_HELP = {
    "brem": "Recompute bremsstrahlung backgrounds.",
    "cxr": "Recompute PXR/CBS spectra.",
}


@click.group(
    "recompute",
    cls=LazyGroup,
    lazy_commands=_RECOMPUTE_COMMANDS,
    lazy_help=_RECOMPUTE_HELP,
    no_args_is_help=True,
)
def recompute_command() -> None:
    """Recompute one checkpoint dataset without changing the other."""


@click.group(
    "checkpoint",
    cls=LazyGroup,
    lazy_commands=_COMMANDS,
    lazy_help=_COMMAND_HELP,
    no_args_is_help=True,
)
def command() -> None:
    """Inspect, transform, recompute, archive, and reclaim local checkpoints.

    \b
    Examples:
      pyrite checkpoint list
      pyrite checkpoint archive store hopg keeper
      pyrite checkpoint recompute cxr hopg
      pyrite checkpoint gc --profile standard
      pyrite checkpoint rm --profile standard
    """


def main(argv=None):
    return run(command, argv, prog_name="pyrite-checkpoint")


if __name__ == "__main__":
    raise SystemExit(main())
