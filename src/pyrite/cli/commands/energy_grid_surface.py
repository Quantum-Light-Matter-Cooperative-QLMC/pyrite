"""User-facing energy-grid views below physical CLI nouns."""

import click

from .._groups import LazyGroup


@click.group(
    "energy-grid",
    cls=LazyGroup,
    lazy_commands={
        "derive": "pyrite.cli.commands.energy_grid.derive_command",
        "show": "pyrite.cli.commands.energy_grid.show_command",
        "cxr": "pyrite.cli.commands.energy_grid_surface.cxr_command",
        "brem": "pyrite.cli.commands.energy_grid_surface.brem_command",
    },
    lazy_help={
        "derive": "Derive CXR and bremsstrahlung energy-grid bounds.",
        "show": "Show CXR and bremsstrahlung grids together.",
        "cxr": "Inspect PXR/CBS energy grids.",
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
    "cxr",
    cls=LazyGroup,
    lazy_commands={"show": "pyrite.cli.commands.energy_grid.cxr_show_command"},
    lazy_help={"show": "Show CXR photon-energy grids."},
    no_args_is_help=True,
)
def cxr_command() -> None:
    """Inspect CXR photon-energy grids."""


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
