"""PyRITE Click command-line entry point."""

from __future__ import annotations

from collections.abc import Sequence

import click

from .. import __version__
from .._compat import warn_legacy_command
from ._core import LazyGroup, color_option, run

_COMMANDS = {
    "run": "pyrite.cli.commands.scan.command",
    "setup": "pyrite.cli.commands.backend_setup.command",
    "app": "pyrite.cli.commands.app.command",
    "checkpoint": "pyrite.cli.commands.checkpoint.command",
    "completion": "pyrite.cli.commands.completion.command",
    "config": "pyrite.cli.commands.config.command",
    "performance": "pyrite.cli.commands.performance.command",
    "slim": "pyrite.cli.commands.slim.command",
    "rebrem": "pyrite.cli.commands.recompute.brem_command",
    "reline": "pyrite.cli.commands.recompute.line_command",
    "archive": "pyrite.checkpoints.archive.archive_command",
    "restore": "pyrite.checkpoints.archive.restore_command",
    "archives": "pyrite.checkpoints.archive.archives_command",
    "union": "pyrite.checkpoints.archive.union_command",
    "remote": "pyrite.remote.command",
    "job": "pyrite.cli.commands.job.command",
    "energy-grid": "pyrite.cli.commands.energy_grid.command",
    "sweep": "pyrite.cli.commands.sweep.command",
    "profile": "pyrite.cli.commands.profile.command",
    "material": "pyrite.cli.commands.material.command",
    "beam": "pyrite.cli.commands.beam.command",
    "prune": "pyrite.cli.commands.cleanup.gc_command",
    "check": "pyrite.check.command",
    "check-config": "pyrite.cli.commands.check_config.command",
}

_COMMAND_HELP = {
    "run": "Run a profile's MC sweeps and write checkpoints.",
    "setup": "Detect GPU hardware and write PYRITE_MC_BACKEND to .env (first run).",
    "app": "Launch or export interactive analysis notebooks.",
    "checkpoint": "Inspect, transform, recompute, archive, and reclaim checkpoints.",
    "completion": "Manage PyRITE shell tab-completion.",
    "config": "Set and inspect current profile and remote-target defaults.",
    "performance": "List, analyze, or delete compute-performance artifacts.",
    "slim": "Shrink a checkpoint for transfer.",
    "rebrem": "Recompute bremsstrahlung arrays in local checkpoints.",
    "reline": "Recompute line spectra in local checkpoints.",
    "archive": "Copy an active checkpoint to the archive shelf.",
    "restore": "Restore a shelved checkpoint to an active slot.",
    "archives": "List checkpoint archives.",
    "union": "Merge a shelved checkpoint into an active slot.",
    "remote": "Run and manage MC sweeps on a remote GPU host.",
    "job": "List, inspect, follow, or stop asynchronous remote jobs.",
    "energy-grid": "Derive, inspect, and manage immutable photon-energy-grid artifacts.",
    "sweep": "Compatibility aliases for retired scan-range commands.",
    "profile": "Manage named catalog campaigns and material membership.",
    "material": "Inspect, validate, edit, and blaze individual materials.",
    "beam": "Manage named beams, attachable to profiles by name.",
    "prune": "Retired spelling of `pyrite checkpoint gc`.",
    "check": "Launch validation or export cached validation figures.",
    "check-config": "Validate a material catalog without starting simulation.",
}


@click.command(
    cls=LazyGroup,
    lazy_commands=_COMMANDS,
    lazy_help=_COMMAND_HELP,
    lazy_hidden={
        "setup",
        "completion",
        "performance",
        "energy-grid",
        "slim",
        "rebrem",
        "reline",
        "archive",
        "restore",
        "archives",
        "union",
        "prune",
        "sweep",
        "check",
        "check-config",
    },
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


def legacy_main(argv: Sequence[str] | None = None):
    """Run the retained ``cxr`` compatibility executable."""
    warn_legacy_command("cxr", "pyrite")
    result = run(command, argv, prog_name="cxr")
    if isinstance(result, int):
        raise SystemExit(result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
