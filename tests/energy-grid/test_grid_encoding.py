"""Case encoding of photon-energy grids (#278)."""

import resource

import numpy as np

from pyrite._energy_grid_encoding import decode_energy_grid, encode_energy_grid


def test_uniform_grid_keeps_the_legacy_triple():
    grid = np.arange(50.0, 1000.0, 25.0)
    assert encode_energy_grid(grid) == (50.0, 1000.0, 25.0)
    np.testing.assert_array_equal(decode_energy_grid(encode_energy_grid(grid)), grid)


def test_wide_geometric_grid_stays_exact_without_building_a_uniform_one():
    """The first spacing of a geometric grid is tiny next to its span (100 MeV / 0.08 eV
    would be a ~1e9-point arange), so the encoder must reject it by length alone."""
    grid = 30.0 * np.exp(0.0025 * np.arange(7500))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, hard))
    try:
        encoded = encode_energy_grid(grid)
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    assert isinstance(encoded, np.ndarray)
    np.testing.assert_array_equal(encoded, grid)
