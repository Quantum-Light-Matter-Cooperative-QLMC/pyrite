"""Canonical grouped CLI for local checkpoint operations."""

from __future__ import annotations

import click

from .._core import LazyGroup, run

_COMMANDS = {
    "slim": "cxr_mc.cli.commands.slim.command",
    "recompute": "cxr_mc.cli.commands.checkpoint.recompute_command",
    "archive": "cxr_mc.checkpoints.archive.archive_command",
    "restore": "cxr_mc.checkpoints.archive.restore_command",
    "list": "cxr_mc.checkpoints.archive.archives_command",
    "merge": "cxr_mc.checkpoints.archive.union_command",
    "gc": "cxr_mc.cli.commands.cleanup.gc_command",
    "rm": "cxr_mc.cli.commands.cleanup.rm_command",
    "prune": "cxr_mc.cli.commands.cleanup.gc_command",
    "clear": "cxr_mc.cli.commands.cleanup.rm_command",
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
    "prune": "Retired spelling of `gc`.",
    "clear": "Retired spelling of `rm`.",
}

_RECOMPUTE_COMMANDS = {
    "brem": "cxr_mc.cli.commands.recompute.brem_command",
    "line": "cxr_mc.cli.commands.recompute.line_command",
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
    lazy_hidden={"prune", "clear"},
    no_args_is_help=True,
)
def command() -> None:
    """Inspect, transform, recompute, archive, and reclaim local checkpoints.

    Existing top-level paths such as ``pyrite slim`` and ``pyrite archive`` remain
    compatibility aliases.

    \b
    Examples:
      pyrite checkpoint list
      pyrite checkpoint archive hopg keeper
      pyrite checkpoint recompute line hopg
      pyrite checkpoint gc --profile standard
      pyrite checkpoint rm --profile standard
    """


def main(argv=None):
    return run(command, argv, prog_name="cxr-checkpoint")


if __name__ == "__main__":
    raise SystemExit(main())
