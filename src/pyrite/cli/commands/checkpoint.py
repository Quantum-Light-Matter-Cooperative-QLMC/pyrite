"""Canonical grouped CLI for local checkpoint operations."""

import click

from ...console.output import run
from .._groups import LazyGroup

_COMMANDS = {
    "slim": "pyrite.cli.commands.slim.command",
    "recompute": "pyrite.cli.commands.checkpoint.recompute_command",
    "archive": "pyrite.cli.commands.archive.archive_command",
    "restore": "pyrite.cli.commands.archive.restore_command",
    "list": "pyrite.cli.commands.archive.archives_command",
    "merge": "pyrite.cli.commands.archive.union_command",
    "gc": "pyrite.cli.commands.cleanup.gc_command",
    "rm": "pyrite.cli.commands.cleanup.rm_command",
}

_COMMAND_HELP = {
    "slim": "Shrink one checkpoint for transfer.",
    "recompute": "Recompute selected checkpoint datasets.",
    "archive": "Copy an active checkpoint to long-term shelf.",
    "restore": "Copy a shelved checkpoint back to active slot.",
    "list": "List long-term checkpoint shelf.",
    "merge": "Merge a shelved checkpoint into active slot.",
    "gc": "Reclaim records obsolete under current scan profiles.",
    "rm": "Delete local datasets and newly unreachable shared cases.",
}

_RECOMPUTE_COMMANDS = {
    "brem": "pyrite.cli.commands.recompute.brem_command",
    "line": "pyrite.cli.commands.recompute.line_command",
}

_RECOMPUTE_HELP = {
    "brem": "Recompute bremsstrahlung backgrounds.",
    "line": "Recompute line spectra.",
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
      pyrite checkpoint archive hopg keeper
      pyrite checkpoint recompute line hopg
      pyrite checkpoint gc --profile standard
      pyrite checkpoint rm --profile standard
    """


def main(argv=None):
    return run(command, argv, prog_name="pyrite-checkpoint")


if __name__ == "__main__":
    raise SystemExit(main())
