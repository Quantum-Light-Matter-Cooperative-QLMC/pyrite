"""PyRITE Click command-line entry point."""

from __future__ import annotations

from collections.abc import Sequence

import click

from .. import __version__
from ._core import LazyGroup, color_option, run

_COMMANDS = {
    "run": "pyrite.cli.commands.scan.command",
    "app": "pyrite.cli.commands.app.command",
    "checkpoint": "pyrite.cli.commands.checkpoint.command",
    "config": "pyrite.cli.commands.config.command",
    "remote": "pyrite.cli.commands.remote.command",
    "job": "pyrite.cli.commands.job.command",
    "profile": "pyrite.cli.commands.profile.command",
    "material": "pyrite.cli.commands.material.command",
    "beam": "pyrite.cli.commands.beam.command",
    "detector": "pyrite.cli.commands.detector.command",
}

_COMMAND_HELP = {
    "run": "Run a profile's MC sweeps and write checkpoints.",
    "app": "Launch or export interactive analysis notebooks.",
    "checkpoint": "Inspect, transform, recompute, archive, and reclaim checkpoints.",
    "config": "Set and inspect current profile and remote-target defaults.",
    "remote": "Run and manage MC sweeps on a remote GPU host.",
    "job": "List, inspect, follow, or stop asynchronous remote jobs.",
    "profile": "Manage named catalog campaigns and material membership.",
    "material": "Inspect, validate, edit, and blaze individual materials.",
    "beam": "Manage named beams, attachable to profiles by name.",
    "detector": "Manage named detector geometries.",
}


@click.command(
    cls=LazyGroup,
    lazy_commands=_COMMANDS,
    lazy_help=_COMMAND_HELP,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="PyRITE", message="PyRITE %(version)s")
@color_option
def command() -> None:
    """Manage coherent X-ray radiation simulation campaigns.

    Start with a named profile: it defines campaign ranges, workloads, and
    material membership. Then run campaigns, inspect checkpoints, or open analysis
    notebooks.

    Run ``pyrite COMMAND --help`` for command options, units, defaults, and side
    effects.

    \b
    Examples:
      pyrite config setup
      pyrite profile list
      pyrite profile show sub_100keV
      pyrite run sub_100keV -m hopg
      pyrite run sub_100keV --remote --dry-run
      pyrite app analysis launch
    """


def main(argv: Sequence[str] | None = None):
    """Run canonical ``pyrite`` preserving exit-code and stream contracts."""
    result = run(command, argv, prog_name="pyrite")
    if isinstance(result, int):
        raise SystemExit(result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
