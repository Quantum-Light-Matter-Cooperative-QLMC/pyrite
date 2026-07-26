"""Canonical grouped CLI for local checkpoint operations."""

from __future__ import annotations

import click

from ._core import LazyGroup, run

_COMMANDS = {
    "slim": "cxr_mc.slim.command",
    "recompute": "cxr_mc.cli.checkpoint.recompute_command",
    "archive": "cxr_mc.archive.archive_command",
    "restore": "cxr_mc.archive.restore_command",
    "list": "cxr_mc.archive.archives_command",
    "merge": "cxr_mc.archive.union_command",
}

_COMMAND_HELP = {
    "slim": "Shrink one checkpoint for transfer.",
    "recompute": "Recompute selected checkpoint datasets.",
    "archive": "Copy an active checkpoint to long-term shelf.",
    "restore": "Copy a shelved checkpoint back to active slot.",
    "list": "List long-term checkpoint shelf.",
    "merge": "Merge a shelved checkpoint into active slot.",
}

_RECOMPUTE_COMMANDS = {
    "brem": "cxr_mc.rebrem.command",
    "line": "cxr_mc.reline.command",
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
    """Inspect, transform, recompute, and archive local checkpoints.

    Existing top-level paths such as ``cxr slim`` and ``cxr archive`` remain
    compatibility aliases.

    \b
    Examples:
      cxr checkpoint list
      cxr checkpoint archive hopg keeper
      cxr checkpoint recompute line hopg
    """


def main(argv=None):
    return run(command, argv, prog_name="cxr-checkpoint")


if __name__ == "__main__":
    raise SystemExit(main())
