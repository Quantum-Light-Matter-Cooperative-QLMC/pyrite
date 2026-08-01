"""``cxr`` Click command-line entry point."""

from __future__ import annotations

from collections.abc import Sequence

import click

from .. import __version__
from ._core import LazyGroup, color_option, run

_COMMANDS = {
    "run": "cxr_mc.scan.command",
    "setup": "cxr_mc.cli.backend_setup.command",
    "app": "cxr_mc.cli.app.command",
    "checkpoint": "cxr_mc.cli.checkpoint.command",
    "completion": "cxr_mc.cli.completion.command",
    "performance": "cxr_mc.cli.performance.command",
    "slim": "cxr_mc.slim.command",
    "rebrem": "cxr_mc.rebrem.command",
    "reline": "cxr_mc.reline.command",
    "archive": "cxr_mc.archive.archive_command",
    "restore": "cxr_mc.archive.restore_command",
    "archives": "cxr_mc.archive.archives_command",
    "union": "cxr_mc.archive.union_command",
    "remote": "cxr_mc.remote.command",
    "energy-grid": "cxr_mc.cli.energy_grid.command",
    "sweep": "cxr_mc.cli.sweep.command",
    "profile": "cxr_mc.cli.profile.command",
    "material": "cxr_mc.cli.material.command",
    "prune": "cxr_mc.prune.command",
    "check": "cxr_mc.check.command",
    "check-config": "cxr_mc.check_config.command",
}

_COMMAND_HELP = {
    "run": "Run a profile's MC sweeps and write checkpoints.",
    "setup": "Detect GPU hardware and write CXR_MC_BACKEND to .env (first run).",
    "app": "Launch or export interactive analysis notebooks.",
    "checkpoint": "Inspect, transform, recompute, archive, and prune checkpoints.",
    "completion": "Manage cxr shell tab-completion.",
    "performance": "List, analyze, or prune compute-performance artifacts.",
    "slim": "Shrink a checkpoint for transfer.",
    "rebrem": "Recompute bremsstrahlung arrays in local checkpoints.",
    "reline": "Recompute line spectra in local checkpoints.",
    "archive": "Copy an active checkpoint to the archive shelf.",
    "restore": "Restore a shelved checkpoint to an active slot.",
    "archives": "List checkpoint archives.",
    "union": "Merge a shelved checkpoint into an active slot.",
    "remote": "Run and manage MC sweeps on a remote GPU host.",
    "energy-grid": "Derive, submit, inspect, and apply photon-energy grids.",
    "sweep": "Compatibility aliases for retired scan-range commands.",
    "profile": "Manage named catalog campaigns and material membership.",
    "material": "Inspect, validate, edit, and blaze individual materials.",
    "prune": "Drop checkpoint records obsolete under current scan profiles.",
    "check": "Launch validation or export cached validation figures.",
    "check-config": "Validate a material catalog without starting simulation.",
}

_DEPRECATED_COMMANDS = {
    "slim": "cxr checkpoint slim",
    "rebrem": "cxr checkpoint recompute brem",
    "reline": "cxr checkpoint recompute line",
    "archive": "cxr checkpoint archive",
    "restore": "cxr checkpoint restore",
    "archives": "cxr checkpoint list",
    "union": "cxr checkpoint merge",
    "prune": "cxr checkpoint prune",
}


@click.command(
    cls=LazyGroup,
    lazy_commands=_COMMANDS,
    lazy_help=_COMMAND_HELP,
    lazy_hidden={
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
    lazy_deprecated=_DEPRECATED_COMMANDS,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="cxr-mc", message="cxr-mc %(version)s")
@color_option
def command() -> None:
    """Manage coherent X-ray radiation simulation campaigns.

    Start with a named profile: it defines campaign ranges, workloads, and
    material membership. Then run campaigns, inspect checkpoints, or open analysis
    notebooks.

    Run ``cxr COMMAND --help`` for command options, units, defaults, and side
    effects.

    \b
    Examples:
      cxr setup
      cxr profile list
      cxr profile show sub_100keV
      cxr run sub_100keV -m hopg
      cxr remote run sub_100keV --dry-run
      cxr app analysis
    """


def main(argv: Sequence[str] | None = None):
    """Run ``cxr`` while preserving project exit-code and stream contracts."""
    result = run(command, argv, prog_name="cxr")
    if isinstance(result, int):
        raise SystemExit(result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
