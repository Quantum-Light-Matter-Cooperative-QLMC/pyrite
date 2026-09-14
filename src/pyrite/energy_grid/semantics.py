"""Compatibility facade for the package-root energy-grid semantics leaf."""

from .._grid_semantics import (
    NonuniformEnergyGridError,
    grid_identity,
    is_uniform_grid,
    require_uniform_grid,
    spacing_spread,
)

__all__ = [
    "NonuniformEnergyGridError",
    "grid_identity",
    "is_uniform_grid",
    "require_uniform_grid",
    "spacing_spread",
]
