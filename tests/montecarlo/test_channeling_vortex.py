"""Orbital-angular-momentum projection about an atomic string (#508)."""

from __future__ import annotations

import numpy as np
import pytest

from pyrite.materials.channeling import longitudinal_kinematics
from pyrite.materials.channeling.axial import axial_profile, axial_states
from pyrite.montecarlo.channeling import states
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid, SplitStepPropagator
from pyrite.montecarlo.channeling.vortex import CylindricalProjection, angular_momentum

GRID = RealSpaceGrid((20.0, 20.0), (64, 64))
CENTER = (10.0, 10.0)


@pytest.mark.parametrize(("ell", "p"), [(0, 0), (1, 0), (-2, 1), (3, 0)])
def test_centred_laguerre_gauss_is_a_delta_at_ell(ell, p):
    phi = GRID.sample(states.laguerre_gauss(ell, p, waist_ang=2.0, center_ang=CENTER))
    projection = CylindricalProjection(GRID, CENTER, 9.5, max_ell=6)
    populations = projection.project(phi)
    expected = np.where(projection.ells == ell, projection.disk_weight(phi), 0.0)
    np.testing.assert_allclose(populations, expected, atol=1e-12)
    assert populations.sum() == pytest.approx(GRID.norm(phi), abs=1e-9)
    assert angular_momentum(populations, projection.ells) == pytest.approx(ell, abs=1e-9)


def test_tilted_plane_wave_is_ell_balanced_about_any_centre():
    phi = np.full(GRID.shape, 1.0 / np.sqrt(GRID.area_ang2), dtype=complex)
    radius = 2.0
    projection = CylindricalProjection(GRID, (3.3, 7.1), radius, max_ell=12)
    populations = projection(phi, (1.7, -0.6))
    np.testing.assert_allclose(populations, populations[::-1], atol=1e-15)
    # |k_t| R ~ 3.6: |l| <= 12 holds the whole disk weight.
    assert populations.sum() == pytest.approx(np.pi * radius**2 / GRID.area_ang2, rel=1e-9)
    assert angular_momentum(populations, projection.ells) == pytest.approx(0.0, abs=1e-15)


def test_angular_momentum_matches_the_derivative_operator():
    state = states.superpose(
        [
            (1.0, states.laguerre_gauss(1, 0, waist_ang=1.5, center_ang=CENTER)),
            (0.5j, states.laguerre_gauss(-2, 0, waist_ang=1.5, center_ang=CENTER)),
            (0.3, states.gaussian(waist_ang=1.5, center_ang=CENTER)),
        ]
    )
    phi = GRID.sample(state)
    phi /= np.sqrt(GRID.norm(phi))
    x = GRID.x[:, None] - CENTER[0]
    y = GRID.y[None, :] - CENTER[1]
    spectrum = np.fft.fft2(phi)
    dx = np.fft.ifft2(1j * GRID.kx[:, None] * spectrum)
    dy = np.fft.ifft2(1j * GRID.ky[None, :] * spectrum)
    direct = np.sum(np.conj(phi) * -1j * (x * dy - y * dx)).real * GRID.cell_area_ang2
    projection = CylindricalProjection(GRID, CENTER, 9.5, max_ell=6)
    populations = projection.project(phi)
    assert angular_momentum(populations, projection.ells) == pytest.approx(direct, abs=1e-12)
    assert populations.sum() == pytest.approx(1.0, abs=1e-10)


def test_off_centre_vortex_spreads_over_ell():
    phi = GRID.sample(states.laguerre_gauss(1, 0, waist_ang=2.0, center_ang=(12.0, 10.0)))
    projection = CylindricalProjection(GRID, CENTER, 9.5, max_ell=8)
    populations = projection.project(phi)
    # A displaced vortex keeps <L_z> = l (zero mean momentum) but loses selectivity.
    assert angular_momentum(populations, projection.ells) / populations.sum() == pytest.approx(
        1.0, abs=1e-3
    )
    assert populations[projection.ells == 1][0] < 0.3 * populations.sum()


def test_string_symmetry_conserves_ell_mod_four():
    # Si <001>: one string per square cell, C4 about the string, so a centred
    # vortex only feeds l' = l (mod 4) under coherent propagation.
    profile = axial_profile("silicon", (0, 0, 1))
    cell = profile.period_ang[0]
    assert np.allclose(profile.strings[0].position_ang, 0.0, atol=1e-9)
    grid = RealSpaceGrid((4 * cell, 4 * cell), (64, 64))
    center = (2 * cell, 2 * cell)
    propagator = SplitStepPropagator(grid, 200e3, potential_eV=profile.on_grid((4, 4), grid.shape))
    phi = grid.sample(states.laguerre_gauss(1, 0, waist_ang=0.5, center_ang=center))
    projection = CylindricalProjection(grid, center, 0.5 * cell, max_ell=7)
    forbidden = np.mod(projection.ells - 1, 4) != 0
    for _, out in propagator.trajectory(phi, [100.0, 300.0], step_ang=1.0):
        populations = projection.project(out)
        assert populations[forbidden].max() < 1e-10 * populations.sum()
        assert populations[projection.ells == -3][0] > 1e-6 * populations.sum()


def test_projection_rejects_bad_geometry():
    with pytest.raises(ValueError, match="2R must fit"):
        CylindricalProjection(GRID, CENTER, 10.5)
    with pytest.raises(ValueError, match="grid shape"):
        CylindricalProjection(GRID, CENTER, 1.0).project(np.zeros((8, 8)))


@pytest.fixture(scope="module")
def tungsten_string():
    """W <001> at 300 keV: on a 6 x 6 supercell, the centre string and its 2p orbitals."""
    energy = 300e3
    profile = axial_profile("tungsten", (0, 0, 1))
    cell = profile.period_ang[0]
    gamma, _ = longitudinal_kinematics(energy)
    bound = axial_states(profile, gamma, max_order=12)
    grid = RealSpaceGrid((6 * cell, 6 * cell), (128, 128))
    center = (3 * cell, 3 * cell)
    x, y = np.meshgrid(grid.x - center[0], grid.y - center[1], indexing="ij")
    inside = (np.abs(x) <= cell / 2) & (np.abs(y) <= cell / 2)
    # The 2p pair is a flat, fully on-column band: its k = 0 Bloch function cut
    # to the centre cell is the on-column orbital.
    orbitals = [
        np.where(inside, bound.wavefunction(grid.x - center[0], grid.y - center[1], i), 0.0)
        for i in (1, 2)
    ]
    propagator = SplitStepPropagator(grid, energy, potential_eV=profile.on_grid((6, 6), grid.shape))
    return grid, center, cell, bound.energies_eV, orbitals, propagator


def _two_p_population(grid, orbitals, phi):
    return sum(abs(np.sum(np.conj(o) * phi) * grid.cell_area_ang2) ** 2 for o in orbitals)


@pytest.mark.slow
def test_atom_size_vortex_keeps_on_column_2p_and_helicity(tungsten_string):
    # Xin & Zheng (arXiv:1204.0021): a centred atom-size l = +-1 probe couples
    # to the on-column 2p state and keeps its helical phase beyond 50 nm.
    grid, center, cell, energies, orbitals, propagator = tungsten_string
    assert energies[1] == pytest.approx(energies[2], abs=1e-6)  # degenerate 2p pair
    assert energies[2] < 0.0 < energies[3]  # bound below the 29 eV barrier
    projection = CylindricalProjection(grid, center, 0.5 * cell, max_ell=3)
    for ell in (1, -1):
        phi = grid.sample(states.laguerre_gauss(ell, 0, waist_ang=0.35, center_ang=center))
        for depth, out in propagator.trajectory(phi, [500.0, 1000.0], step_ang=1.0):
            assert _two_p_population(grid, orbitals, out) > 0.9, depth
            populations = projection.project(out)
            same = populations[projection.ells == ell][0]
            assert same > 0.9
            assert populations[projection.ells == -ell][0] < 1e-12 * same


@pytest.mark.slow
@pytest.mark.parametrize(("offset_cells", "waist_ang"), [(0.5, 0.35), (0.0, 3.0)])
def test_off_centre_or_wide_vortex_loses_selectivity(tungsten_string, offset_cells, waist_ang):
    # Löffler & Schattschneider (arXiv:1111.6050): OAM about the beam axis is not OAM about
    # the string once the probe is off-centre or wider than the string spacing.
    grid, center, cell, _, orbitals, propagator = tungsten_string
    entry = (center[0] + offset_cells * cell, center[1])
    phi = grid.sample(states.laguerre_gauss(1, 0, waist_ang=waist_ang, center_ang=entry))
    for depth, out in propagator.trajectory(phi, [500.0], step_ang=1.0):
        assert _two_p_population(grid, orbitals, out) < 0.02, depth
