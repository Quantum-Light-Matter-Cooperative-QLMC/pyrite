"""Analytic, no-remote tests for the dielectric omission bound (#439)."""

import numpy as np
import pytest

from pyrite.materials.attenuation import plasma_energy_eV
from pyrite.montecarlo.spectrum.dielectric_suppression import (
    dielectric_omission_bound,
    ter_mikaelian_factor,
)


def test_factor_matches_formula_and_vacuum_limit():
    T = np.array([0.0, 30e6, 100e6])
    k = np.array([1e3, 1e3, 2e4])
    ep = 31.0498
    gamma = 1 + T / 510998.95
    np.testing.assert_allclose(
        ter_mikaelian_factor(T, k, ep), k**2 / (k**2 + (gamma * ep) ** 2), rtol=1e-15
    )
    np.testing.assert_array_equal(ter_mikaelian_factor(T, k, 0), np.ones(3))


def test_low_high_energy_limits_and_broadcast():
    value = ter_mikaelian_factor(30e6, np.array([1e-3, 1e3, 1e9]), 31.0498)
    assert value[0] < 1e-11
    assert value[0] < value[1] < value[2]
    assert value[2] == pytest.approx(1.0, abs=1e-10)
    grid = ter_mikaelian_factor(np.array([30e6, 100e6])[:, None], [1e3, 1e4], 31.0498)
    assert grid.shape == (2, 2)
    assert np.all(grid[1] <= grid[0])
    assert np.all(grid[:, 1] > grid[:, 0])
    assert np.all((0 <= grid) & (grid <= 1))


@pytest.mark.parametrize(
    ("kinetic", "photon", "plasma"),
    [
        (-1.0, 1.0, 1.0),
        (np.nan, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, np.inf, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, np.nan),
    ],
)
def test_factor_refuses_invalid_inputs(kinetic, photon, plasma):
    with pytest.raises(ValueError):
        ter_mikaelian_factor(kinetic, photon, plasma)


def test_si_scale_at_30_and_100_mev():
    ep = plasma_energy_eV("silicon")
    assert ep == pytest.approx(31.0498, rel=3e-3)
    scales = (1 + np.array([30e6, 100e6]) / 510998.95) * ep
    assert scales[0] == pytest.approx(1.85e3, rel=0.03)
    assert scales[1] == pytest.approx(6.11e3, rel=0.03)
    np.testing.assert_allclose(ter_mikaelian_factor([30e6, 100e6], scales, ep), 0.5)


def test_bound_encloses_integrated_nonnegative_observables():
    energies = np.geomspace(500.0, 50000.0, 120)
    electron_T = np.linspace(1e6, 100e6, 45)[:, None]
    ep_si = plasma_energy_eV("silicon")
    bound = dielectric_omission_bound(100e6, float(energies[0]), ["silicon"])
    assert 0 < bound < 1
    factors = ter_mikaelian_factor(electron_T, energies[None, :], ep_si)
    weights = np.random.default_rng(439).uniform(0, 1, size=factors.shape)
    bias = float(np.sum(weights * (1 - factors)) / np.sum(weights))
    assert bias <= bound + 1e-15
    assert dielectric_omission_bound(100e6, 500.0, ["silicon", "silicon"]) == bound
    assert dielectric_omission_bound(100e6, 5e4, ["silicon"]) < bound
    with pytest.raises(ValueError, match="at least one"):
        dielectric_omission_bound(100e6, 500, [])
