"""Compatibility facade for the package-root energy-grid encoding leaf."""

from .._energy_grid_encoding import EnergyGridEncoding, decode_energy_grid, encode_energy_grid

__all__ = ["EnergyGridEncoding", "decode_energy_grid", "encode_energy_grid"]
