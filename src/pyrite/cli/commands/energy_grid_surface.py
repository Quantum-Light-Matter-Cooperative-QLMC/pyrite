"""User-facing energy-grid views below physical CLI nouns."""

from __future__ import annotations

import click

from .._core import LazyGroup


@click.group(
    "energy-grid",
    cls=LazyGroup,
    lazy_commands={
        "derive": "pyrite.cli.commands.energy_grid.derive_command",
        "show": "pyrite.cli.commands.energy_grid.show_command",
        "line": "pyrite.cli.commands.energy_grid_surface.line_command",
        "brem": "pyrite.cli.commands.energy_grid_surface.brem_command",
    },
    lazy_help={
        "derive": "Derive line and bremsstrahlung energy-grid bounds.",
        "show": "Show line and bremsstrahlung grids together.",
        "line": "Inspect coherent line-energy grids.",
        "brem": "Inspect bremsstrahlung energy grids.",
    },
    no_args_is_help=True,
)
def material_command() -> None:
    """Derive and inspect detector energy-grid inputs by material.

    \b
    Examples:
      pyrite material energy-grid derive --material mose2 --energy 30,60
      pyrite material energy-grid show mose2
    """


@click.group(
    "line",
    cls=LazyGroup,
    lazy_commands={"show": "pyrite.cli.commands.energy_grid.line_show_command"},
    lazy_help={"show": "Show coherent line-energy grids."},
    no_args_is_help=True,
)
def line_command() -> None:
    """Inspect coherent line-energy grids."""


@click.group(
    "brem",
    cls=LazyGroup,
    lazy_commands={"show": "pyrite.cli.commands.energy_grid.brem_show_command"},
    lazy_help={"show": "Show bremsstrahlung energy grids."},
    no_args_is_help=True,
)
def brem_command() -> None:
    """Inspect bremsstrahlung energy grids."""


@click.group(
    "energy-grid",
    cls=LazyGroup,
    lazy_commands={"defaults": "pyrite.cli.commands.energy_grid.defaults_command"},
    lazy_help={"defaults": "Show, update, or clear persistent derivation inputs."},
    no_args_is_help=True,
)
def profile_command() -> None:
    """Manage profile-scoped energy-grid derivation inputs."""
