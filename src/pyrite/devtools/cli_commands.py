"""Click command trees relocated from the user CLI to ``pyrite-dev``."""

from __future__ import annotations

import click

from pyrite.cli._core import LazyGroup


@click.group(
    "performance",
    cls=LazyGroup,
    deprecation_prefix="pyrite-dev performance",
    lazy_commands={
        "analyze": "pyrite.cli.commands.performance.analyze_command",
        "list": "pyrite.cli.commands.performance.list_command",
        "rm": "pyrite.cli.commands.performance.rm_command",
    },
    lazy_help={
        "analyze": "Summarize one local performance profile.",
        "list": "List local performance profiles and artifact counts.",
        "rm": "Preview or delete selected local performance artifacts.",
    },
    no_args_is_help=True,
)
def performance_command() -> None:
    """List, analyze, or delete local compute-performance artifacts."""


@click.group(
    "energy-grid",
    cls=LazyGroup,
    deprecation_prefix="pyrite-dev energy-grid",
    lazy_commands={
        "add": "pyrite.cli.commands.energy_grid.add_command",
        "line": "pyrite.devtools.cli_commands.energy_grid_line_command",
        "brem": "pyrite.devtools.cli_commands.energy_grid_brem_command",
        "rm": "pyrite.cli.commands.energy_grid.rm_command",
        "verify": "pyrite.cli.commands.energy_grid.verify_command",
        "gc": "pyrite.cli.commands.energy_grid.gc_command",
    },
    lazy_help={
        "add": "Add immutable derived-grid artifacts and repoint one profile.",
        "line": "Manually maintain coherent line-energy grids.",
        "brem": "Manually maintain bremsstrahlung energy grids.",
        "rm": "Remove line rows by repointing a profile.",
        "verify": "Verify stored and referenced immutable artifacts.",
        "gc": "Reclaim unreachable immutable artifacts after the grace window.",
    },
    no_args_is_help=True,
)
def energy_grid_command() -> None:
    """Maintain derived detector energy-grid artifacts."""


@click.group(
    "line",
    cls=LazyGroup,
    deprecation_prefix="pyrite-dev energy-grid line",
    lazy_commands={"set": "pyrite.cli.commands.energy_grid.set_command"},
    lazy_help={"set": "Set one line-grid row by repointing an immutable artifact."},
    no_args_is_help=True,
)
def energy_grid_line_command() -> None:
    """Manually maintain coherent line-energy grids."""


@click.group(
    "brem",
    cls=LazyGroup,
    deprecation_prefix="pyrite-dev energy-grid brem",
    lazy_commands={"set": "pyrite.cli.commands.energy_grid.set_brem_command"},
    lazy_help={"set": "Set a bremsstrahlung grid by repointing an immutable artifact."},
    no_args_is_help=True,
)
def energy_grid_brem_command() -> None:
    """Manually maintain bremsstrahlung energy grids."""
