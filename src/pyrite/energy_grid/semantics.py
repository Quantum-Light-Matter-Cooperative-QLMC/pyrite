"""Compatibility facade for the package-root energy-grid semantics leaf."""

from .._grid_semantics import (
    NonuniformEnergyGridError,
    grid_identity,
    is_uniform_grid,
    node_bin_edges_and_widths,
    rebin_piecewise_constant_density,
    require_uniform_grid,
    resolution_num,
    spacing_spread,
    validate_backend_spacing,
    zero_based_detector_edges,
)

__all__ = [
    "NonuniformEnergyGridError",
    "grid_identity",
    "is_uniform_grid",
    "node_bin_edges_and_widths",
    "rebin_piecewise_constant_density",
    "require_uniform_grid",
    "resolution_num",
    "spacing_spread",
    "validate_backend_spacing",
    "zero_based_detector_edges",
]
