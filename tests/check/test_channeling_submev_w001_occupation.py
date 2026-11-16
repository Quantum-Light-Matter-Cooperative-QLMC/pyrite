"""Deterministic checks of the W <001> occupation record helpers (#585)."""

import numpy as np
import pytest

from checks import channeling_submev_w001_occupation as occ
from pyrite.materials.channeling.axial import axial_profile
from pyrite.materials.crystal import HBAR2_2M_EV_ANG2
from pyrite.montecarlo.channeling import longitudinal_kinematics
from pyrite.montecarlo.channeling.axial_bloch import axial_states
from pyrite.montecarlo.channeling.dipole import dipole_rate_per_ang
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid

POINTS = 25
CELLS = 4


@pytest.fixture(scope="module")
def tungsten_30kev():
    profile = axial_profile("tungsten", (0, 0, 1), 293.15)
    gamma, _ = longitudinal_kinematics(30e3)
    period = profile.period_ang
    grid = RealSpaceGrid((CELLS * period[0], CELLS * period[1]), (CELLS * POINTS,) * 2)
    return profile, gamma, grid, occ.BlochBandObservable(profile, gamma, grid, CELLS)


def _bloch(profile, gamma, k):
    return axial_states(profile, gamma, tuple(k), max_order=occ.CATALOG_ORDER)


@pytest.mark.parametrize("fraction", [(0.0, 0.0), (0.05, 0.0), (0.13, 0.07), (0.6, -0.3)])
def test_plane_wave_populations_match_the_bloch_overlaps(tungsten_30kev, fraction):
    # An incommensurate tilt (0.13 g) and one outside the first zone (0.6 g).
    profile, gamma, grid, observable = tungsten_30kev
    g = np.asarray(profile.g_inv_ang)
    tilt = g * np.asarray(fraction)
    shift = np.rint(tilt / g).astype(int)
    solved = _bloch(profile, gamma, tilt - shift * g)
    row = np.flatnonzero(np.all(solved.orders == shift, axis=1))[0]
    expected = np.abs(solved.coefficients[row, :3]) ** 2
    p_1s, p_2p, _ = observable(grid.sample(occ.plane_wave(grid)), tuple(tilt))
    assert p_1s == pytest.approx(expected[0], rel=1e-9)
    assert p_2p == pytest.approx(expected[1] + expected[2], rel=1e-9, abs=1e-14)


def test_bloch_state_is_one_band_with_its_vertical_rate(tungsten_30kev):
    # psi = 2p band state at k = K + g (1, 2) / CELLS, carried by an arbitrary K.
    profile, gamma, grid, observable = tungsten_30kev
    g = np.asarray(profile.g_inv_ang)
    tilt = g * np.array([0.37, -0.11])
    k = tilt + g * np.array([1, 2]) / CELLS
    solved = _bloch(profile, gamma, k - np.rint(k / g) * g)
    psi = solved.wavefunction(grid.x, grid.y, 1) / CELLS  # cell -> supercell norm
    phi = np.exp(-1j * (tilt[0] * grid.x[:, None] + tilt[1] * grid.y[None, :])) * psi
    p_1s, p_2p, rate = observable(phi, tuple(tilt))
    assert p_1s == pytest.approx(0.0, abs=1e-10)
    assert p_2p == pytest.approx(1.0, rel=1e-9)
    c = solved.coefficients
    kvec = np.asarray(k - np.rint(k / g) * g) + solved.orders * g
    p = np.einsum("g,ga,g->a", c[:, 0].conj(), kvec, c[:, 1])
    gap = solved.energies_eV[1] - solved.energies_eV[0]
    x = 2.0 * HBAR2_2M_EV_ANG2 / gamma * np.linalg.norm(p) / gap
    assert rate == pytest.approx(dipole_rate_per_ang(gap, x, gamma), rel=1e-9)


def test_cumulative_photons_is_the_trapezoid_integral():
    depths = np.array([0.0, 10.0, 30.0])
    rates = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 2.0]])
    np.testing.assert_allclose(
        occ.cumulative_photons(depths, rates), [[0.0, 10.0, 30.0], [0.0, 10.0, 50.0]]
    )


def test_plane_wave_entry_is_normalized(tungsten_30kev):
    grid = tungsten_30kev[2]
    phi = grid.sample(occ.plane_wave(grid))
    assert grid.norm(phi) == pytest.approx(1.0, rel=1e-12)


def test_efold_depth_interpolates_and_refuses():
    depths = np.array([0.0, 10.0, 20.0])
    assert occ.efold_depth(depths, np.exp(-depths / 15.0)) == pytest.approx(15.0, rel=0.1)
    assert occ.efold_depth(depths, np.ones(3)) is None
    assert occ.efold_depth(depths, np.zeros(3)) is None
